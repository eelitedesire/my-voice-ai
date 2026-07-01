"""Real-time streaming speaker diarization + identification.

Pipeline per session:

    audio chunk
      -> framed at 512 samples, Silero VAD per frame  (speech/silence)
      -> contiguous speech is a "turn"; sliding windows (window_sec / hop_sec)
         inside the turn are embedded with ECAPA
      -> each window embedding is scored (cosine) against every enrolled speaker
      -> per-speaker scores are EMA-smoothed; the active speaker is chosen with
         hysteresis (margin + consecutive-window requirement) to prevent flicker
      -> below id_threshold => "Unknown Speaker"
      -> sub-segments are emitted on speaker switch or on trailing silence

The diarizer is transport-agnostic: ``process()`` returns a list of event dicts.
Finalized segments carry their raw audio so the caller can run ASR on them.
"""
from __future__ import annotations

import numpy as np

from .config import settings
from .embeddings import embed
from .vad import StreamingVAD
from .enrollment import ProfileSnapshot

FRAME = StreamingVAD.FRAME  # 512 samples @ 16 kHz

UNKNOWN = "__unknown__"
UNKNOWN_NAME = "Unknown Speaker"


class StreamingDiarizer:
    def __init__(self, profiles: ProfileSnapshot):
        self.sr = settings.sample_rate
        self.profiles = profiles
        self.vad = StreamingVAD()

        # global sample clock (drives timestamps, in seconds since stream start)
        self._clock = 0

        # framing buffer for whatever hasn't filled a 512-sample frame yet
        self._frame_buf = np.zeros(0, dtype=np.float32)

        # ---- turn state ----
        self._in_speech = False
        self._silence_frames = 0
        self._turn_audio = np.zeros(0, dtype=np.float32)
        self._turn_start = 0            # global sample idx where the current turn began
        self._next_window = 0           # turn-relative sample idx at which to embed next

        # ---- sub-segment (single-speaker span within a turn) ----
        self._seg_start = 0             # global sample idx
        self._active = None             # active speaker label (id or UNKNOWN) or None

        # ---- hysteresis / smoothing ----
        self._ema = None                # np.array aligned to profiles order
        self._switch_label = None
        self._switch_count = 0

    # ---------- config-derived sizes ----------
    @property
    def _win_n(self) -> int:
        return int(settings.window_sec * self.sr)

    @property
    def _hop_n(self) -> int:
        return int(settings.hop_sec * self.sr)

    @property
    def _finalize_silence_frames(self) -> int:
        return max(1, int((settings.finalize_silence_ms / 1000.0) * self.sr / FRAME))

    # ---------- public API ----------

    def process(self, chunk: np.ndarray) -> list[dict]:
        """Feed a mono float32 chunk (any length). Return ordered event dicts."""
        events: list[dict] = []
        self._frame_buf = np.concatenate([self._frame_buf, chunk.astype(np.float32)])
        while self._frame_buf.size >= FRAME:
            frame = self._frame_buf[:FRAME]
            self._frame_buf = self._frame_buf[FRAME:]
            self._handle_frame(frame, events)
        return events

    def flush(self) -> list[dict]:
        """End of stream: finalize any open segment."""
        events: list[dict] = []
        if self._in_speech and self._active is not None:
            self._finalize_segment(events, end=self._clock)
        self._in_speech = False
        return events

    # ---------- internals ----------

    def _handle_frame(self, frame: np.ndarray, events: list[dict]) -> None:
        is_speech = self.vad.push(frame)
        is_speech = is_speech[0] if is_speech else False
        self._clock += FRAME

        if is_speech:
            self._silence_frames = 0
            if not self._in_speech:
                self._begin_turn(events)
            self._turn_audio = np.concatenate([self._turn_audio, frame])
            self._maybe_score_window(events)
        else:
            if self._in_speech:
                # keep appending a little silence so ASR gets word tails
                self._turn_audio = np.concatenate([self._turn_audio, frame])
                self._silence_frames += 1
                if self._silence_frames >= self._finalize_silence_frames:
                    self._end_turn(events)

    def _begin_turn(self, events: list[dict]) -> None:
        self._in_speech = True
        self._turn_audio = np.zeros(0, dtype=np.float32)
        self._turn_start = self._clock
        self._seg_start = self._clock
        self._next_window = self._win_n
        self._active = None
        self._ema = None
        self._switch_label = None
        self._switch_count = 0
        events.append({"type": "vad", "active": True})

    def _end_turn(self, events: list[dict]) -> None:
        if self._active is not None:
            self._finalize_segment(events, end=self._clock)
        self._in_speech = False
        self._turn_audio = np.zeros(0, dtype=np.float32)
        events.append({"type": "vad", "active": False})

    def _maybe_score_window(self, events: list[dict]) -> None:
        if len(self.profiles) == 0:
            # no enrolled speakers -> everything is Unknown, but still segment it
            self._route_label(UNKNOWN, 0.0, events)
            return
        while len(self._turn_audio) >= self._next_window:
            end = self._next_window
            start = max(0, end - self._win_n)
            window = self._turn_audio[start:end]
            self._next_window += self._hop_n
            self._score_and_route(window, events)

    def _score_and_route(self, window: np.ndarray, events: list[dict]) -> None:
        emb = embed(window)
        if emb is None:
            return
        raw = self._raw_scores(emb)             # (N,) cosine per speaker
        # EMA smoothing
        if self._ema is None:
            self._ema = raw.copy()
        else:
            a = settings.ema_alpha
            self._ema = a * raw + (1 - a) * self._ema

        best_i = int(np.argmax(self._ema))
        best_score = float(self._ema[best_i])
        if best_score >= settings.id_threshold:
            candidate = self.profiles.ids[best_i]
        else:
            candidate = UNKNOWN
        self._route_label(candidate, best_score, events, ema=self._ema)

    def _raw_scores(self, emb: np.ndarray) -> np.ndarray:
        mode = settings.scoring
        scores = np.empty(len(self.profiles), dtype=np.float32)
        for i in range(len(self.profiles)):
            if mode == "centroid":
                scores[i] = float(np.dot(emb, self.profiles.centroids[i]))
            else:  # "max" or "mean" over per-sample embeddings (more robust)
                sims = self.profiles.per_sample[i] @ emb
                scores[i] = float(sims.max() if mode == "max" else sims.mean())
        return scores

    def _current_active_score(self) -> float:
        if self._active is None or self._active == UNKNOWN or self._ema is None:
            return settings.id_threshold
        try:
            idx = self.profiles.ids.index(self._active)
        except ValueError:
            return settings.id_threshold
        return float(self._ema[idx])

    def _route_label(self, candidate: str, score: float, events: list[dict],
                     ema: np.ndarray | None = None) -> None:
        """Apply hysteresis, split sub-segments on confirmed switches, emit partials."""
        if self._active is None:
            self._active = candidate
            self._seg_start = self._turn_start
        elif candidate == self._active:
            self._switch_label = None
            self._switch_count = 0
        else:
            # only count as a switch if the challenger beats the incumbent by margin
            if score - self._current_active_score() >= settings.switch_margin:
                if candidate == self._switch_label:
                    self._switch_count += 1
                else:
                    self._switch_label = candidate
                    self._switch_count = 1
                if self._switch_count >= settings.min_switch_windows:
                    # confirmed speaker change: finalize the previous span
                    self._finalize_segment(events, end=self._clock)
                    self._active = candidate
                    self._seg_start = self._clock
                    self._switch_label = None
                    self._switch_count = 0
            else:
                self._switch_label = None
                self._switch_count = 0

        events.append({
            "type": "partial",
            **self._label_fields(self._active),
            "confidence": round(max(score, self._current_active_score()), 4),
            "start": round(self._seg_start / self.sr, 3),
            "end": round(self._clock / self.sr, 3),
        })

    def _finalize_segment(self, events: list[dict], end: int) -> None:
        if self._active is None:
            return
        dur = (end - self._seg_start) / self.sr
        if dur < settings.min_segment_sec:
            return
        # slice the turn audio for this sub-segment (for ASR)
        rel_start = max(0, self._seg_start - self._turn_start)
        rel_end = min(len(self._turn_audio), end - self._turn_start)
        seg_audio = self._turn_audio[rel_start:rel_end].copy()

        events.append({
            "type": "segment",
            **self._label_fields(self._active),
            "confidence": round(self._current_active_score(), 4),
            "start": round(self._seg_start / self.sr, 3),
            "end": round(end / self.sr, 3),
            "_audio": seg_audio,   # consumed by caller for transcription, then dropped
        })

    def _label_fields(self, label: str | None) -> dict:
        if label is None or label == UNKNOWN:
            return {"speaker": UNKNOWN_NAME, "speaker_id": None, "unknown": True}
        try:
            name = self.profiles.names[self.profiles.ids.index(label)]
        except ValueError:
            name = UNKNOWN_NAME
        return {"speaker": name, "speaker_id": label, "unknown": False}
