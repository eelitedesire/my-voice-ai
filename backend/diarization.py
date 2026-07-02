"""Real-time streaming speaker diarization + identification.

Pipeline per session:

    audio chunk
      -> framed at 512 samples, Silero VAD per frame  (speech/silence)
      -> contiguous speech is a "turn"; sliding windows (window_sec / hop_sec)
         inside the turn are embedded with ECAPA
      -> SEGMENTATION is acoustic: each window is compared (cosine) to a running
         centroid of the current segment; when it drops below change_sim_threshold
         for min_switch_windows consecutive windows, a speaker change is confirmed
         and the segment is split. This works for known AND unknown speakers, so two
         different unknown speakers in a row are kept apart.
      -> LABELING is separate: window embeddings are scored (EMA-smoothed) against
         enrolled speakers; a segment adopts the best match >= id_threshold, else it
         stays "Unknown Speaker". (A confirmed enrolled-label change is a second,
         complementary split trigger for similar-sounding enrolled voices.)
      -> sub-segments are emitted on a confirmed speaker change or on trailing silence

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

        # ---- current sub-segment = one acoustic speaker span within a turn ----
        self._seg_start = 0             # global sample idx
        self._active = None             # segment label (enrolled id or UNKNOWN) or None
        self._seg_centroid = None       # running mean embedding = acoustic identity
        self._seg_n = 0                 # windows absorbed into the centroid
        self._change_count = 0          # consecutive change-vote windows (hysteresis)
        self._ema = None                # enrolled-score EMA (labeling), reset per segment

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
        self._seg_centroid = None
        self._seg_n = 0
        self._change_count = 0
        self._ema = None
        events.append({"type": "vad", "active": True})

    def _end_turn(self, events: list[dict]) -> None:
        if self._active is not None:
            self._finalize_segment(events, end=self._clock)
        self._in_speech = False
        self._turn_audio = np.zeros(0, dtype=np.float32)
        events.append({"type": "vad", "active": False})

    def _maybe_score_window(self, events: list[dict]) -> None:
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
        raw = self._raw_scores(emb)             # (N,) cosine vs enrolled (empty if none)

        if self._seg_centroid is None:
            # first window of the turn -> segment starts at the speech onset
            self._begin_segment(emb, raw, self._turn_start)
        else:
            # --- labeling: EMA-smoothed enrolled scores for the CURRENT segment ---
            self._ema = settings.ema_alpha * raw + (1 - settings.ema_alpha) * self._ema
            best_score = float(self._ema.max()) if self._ema.size else 0.0
            cand = (self.profiles.ids[int(np.argmax(self._ema))]
                    if (self._ema.size and best_score >= settings.id_threshold) else UNKNOWN)

            # --- segmentation: acoustic change (works for known AND unknown) ---
            sim = float(np.dot(emb, self._seg_centroid))
            acoustic_change = sim < settings.change_sim_threshold
            # a second, complementary trigger for similar-sounding *enrolled* voices
            label_change = (self._active not in (None, UNKNOWN) and cand != UNKNOWN
                            and cand != self._active
                            and (best_score - self._current_active_score()) >= settings.switch_margin)

            if acoustic_change or label_change:
                self._change_count += 1
                if self._change_count >= settings.min_switch_windows:
                    self._finalize_segment(events, end=self._clock)   # close previous speaker
                    self._begin_segment(emb, raw, self._clock)        # open the new one
            else:
                self._change_count = 0
                self._absorb(emb)
                # sticky labeling: adopt/keep a known id; never downgrade known->unknown
                if cand != UNKNOWN:
                    self._active = cand
                elif self._active is None:
                    self._active = UNKNOWN

        best = float(self._ema.max()) if (self._ema is not None and self._ema.size) else 0.0
        events.append({
            "type": "partial",
            **self._label_fields(self._active),
            "confidence": round(best, 4),
            "start": round(self._seg_start / self.sr, 3),
            "end": round(self._clock / self.sr, 3),
        })

    def _begin_segment(self, emb: np.ndarray, raw: np.ndarray, start: int) -> None:
        """Start a new single-speaker segment anchored on this window's embedding."""
        self._seg_start = start
        self._seg_centroid = emb.copy()
        self._seg_n = 1
        self._change_count = 0
        self._ema = raw.copy()
        if raw.size and float(raw.max()) >= settings.id_threshold:
            self._active = self.profiles.ids[int(np.argmax(raw))]
        else:
            self._active = UNKNOWN

    def _absorb(self, emb: np.ndarray) -> None:
        """Fold a same-speaker window into the running (renormalized) centroid."""
        self._seg_n += 1
        c = self._seg_centroid + (emb - self._seg_centroid) / self._seg_n
        n = float(np.linalg.norm(c))
        self._seg_centroid = (c / n) if n > 1e-9 else c

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
