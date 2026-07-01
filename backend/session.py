"""Live session coordinator: binds the (unchanged) diarizer to streaming ASR.

Concurrency model (two executor-backed tasks per WebSocket):

  * ``feed()``     — fast path. Runs the diarizer on each audio chunk, buffers
                     audio, and maintains the *speaker block* timeline. Never runs
                     Whisper, so speaker/VAD events stay real-time.
  * ``asr_tick()`` — heavy path. Periodically runs the streaming decoder on the
                     open block (emitting refined partials) and finalizes any
                     blocks the diarizer has closed. Only this task touches Whisper,
                     so there is never concurrent decoding.

A **block** == one contiguous turn of a single identified speaker, exactly as the
diarizer defines it. Each block owns one ``StreamingTranscriber``; text therefore
can never mix two speakers, and speaker labels persist while text streams.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .diarization import StreamingDiarizer, UNKNOWN, UNKNOWN_NAME
from .streaming_asr import StreamingTranscriber
from .enrollment import ProfileSnapshot


class RollingAudio:
    """Absolute-sample-indexed audio buffer that drops committed history."""

    def __init__(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.start = 0  # absolute sample index of buf[0]

    def append(self, wav: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, wav.astype(np.float32)])

    def slice(self, a: int, b: int) -> np.ndarray:
        i = max(0, a - self.start)
        j = max(0, min(len(self.buf), b - self.start))
        return self.buf[i:j].copy() if j > i else np.zeros(0, dtype=np.float32)

    def drop_before(self, sample: int) -> None:
        if sample > self.start:
            cut = min(len(self.buf), sample - self.start)
            self.buf = self.buf[cut:]
            self.start += cut


@dataclass
class Block:
    id: int
    speaker_id: str | None
    speaker: str
    unknown: bool
    start_sample: int
    end_sample: int
    confidence: float
    transcriber: StreamingTranscriber = field(default_factory=StreamingTranscriber)
    fed_until: int = 0
    last_emitted: str = ""

    def fields(self) -> dict:
        return {
            "block_id": self.id,
            "speaker": self.speaker,
            "speaker_id": self.speaker_id,
            "unknown": self.unknown,
        }


class LiveSession:
    def __init__(self, profiles: ProfileSnapshot) -> None:
        self.sr = settings.sample_rate
        self.diar = StreamingDiarizer(profiles)
        self.roll = RollingAudio()
        self.clock = 0
        self.lock = threading.Lock()
        self.open_block: Block | None = None
        self.finalize_queue: list[Block] = []
        self._next_id = 0

    # -------------------------------------------------- fast path (diarization)
    def feed(self, wav: np.ndarray) -> list[dict]:
        with self.lock:
            self.roll.append(wav)
            self.clock += len(wav)
        ui: list[dict] = []
        for ev in self.diar.process(wav):
            t = ev["type"]
            if t == "vad":
                ui.append({"type": "vad", "active": ev["active"]})
                if not ev["active"]:
                    self._close_open_block(self.clock)
            elif t == "partial":
                self._on_partial(ev, ui)
            elif t == "segment":
                ev.pop("_audio", None)
                self._on_segment(ev, ui)
        return ui

    def flush(self) -> list[dict]:
        ui: list[dict] = []
        for ev in self.diar.flush():
            if ev["type"] == "segment":
                ev.pop("_audio", None)
                self._on_segment(ev, ui)
        self._close_open_block(self.clock)
        return ui

    def _open(self, ev: dict, start_s: int, ui: list[dict]) -> None:
        blk = Block(
            id=self._next_id, speaker_id=ev["speaker_id"], speaker=ev["speaker"],
            unknown=ev["unknown"], start_sample=start_s, end_sample=start_s,
            confidence=ev.get("confidence", 0.0), fed_until=start_s,
        )
        self._next_id += 1
        self.open_block = blk
        ui.append({"type": "block_open", "start": round(start_s / self.sr, 3),
                   "confidence": blk.confidence, **blk.fields()})

    def _on_partial(self, ev: dict, ui: list[dict]) -> None:
        spk = ev["speaker_id"] or UNKNOWN
        start_s = int(round(ev["start"] * self.sr))
        end_s = int(round(ev["end"] * self.sr))
        with self.lock:
            cur = self.open_block
            cur_spk = (cur.speaker_id or UNKNOWN) if cur else None
            if cur is None:
                self._open(ev, start_s, ui)
            elif cur_spk != spk:
                # speaker changed without an explicit segment (safety net)
                cur.end_sample = start_s
                self.finalize_queue.append(cur)
                self.open_block = None
                self._open(ev, start_s, ui)
            if self.open_block is not None:
                self.open_block.end_sample = max(self.open_block.end_sample, end_s)
                self.open_block.confidence = ev["confidence"]
        ui.append({"type": "now", "confidence": ev["confidence"],
                   "speaker": ev["speaker"], "speaker_id": ev["speaker_id"],
                   "unknown": ev["unknown"]})

    def _on_segment(self, ev: dict, ui: list[dict]) -> None:
        end_s = int(round(ev["end"] * self.sr))
        with self.lock:
            cur = self.open_block
            if cur is not None:
                cur.end_sample = max(cur.end_sample, end_s)
                # diarizer's segment speaker is authoritative
                cur.speaker, cur.speaker_id = ev["speaker"], ev["speaker_id"]
                cur.unknown, cur.confidence = ev["unknown"], ev["confidence"]
                self.finalize_queue.append(cur)
                self.open_block = None
            else:
                start_s = int(round(ev["start"] * self.sr))
                blk = Block(
                    id=self._next_id, speaker_id=ev["speaker_id"], speaker=ev["speaker"],
                    unknown=ev["unknown"], start_sample=start_s, end_sample=end_s,
                    confidence=ev["confidence"], fed_until=start_s,
                )
                self._next_id += 1
                self.finalize_queue.append(blk)

    def _close_open_block(self, end_sample: int) -> None:
        with self.lock:
            if self.open_block is not None:
                self.open_block.end_sample = max(self.open_block.end_sample, end_sample)
                self.finalize_queue.append(self.open_block)
                self.open_block = None

    # -------------------------------------------------- heavy path (streaming ASR)
    def asr_tick(self) -> list[dict]:
        evs: list[dict] = []
        # 1) finalize any closed blocks, in order
        while True:
            with self.lock:
                blk = self.finalize_queue.pop(0) if self.finalize_queue else None
                if blk is None:
                    break
                audio = self.roll.slice(blk.fed_until, blk.end_sample)
                blk.fed_until = blk.end_sample
            if audio.size:
                blk.transcriber.insert_audio(audio)
            res = blk.transcriber.finalize()
            evs.append(self._final_event(blk, res))
            with self.lock:
                self._trim()

        # 2) refine the currently open block
        with self.lock:
            blk = self.open_block
            audio = self.roll.slice(blk.fed_until, self.clock) if blk else None
            if blk is not None:
                blk.fed_until = self.clock
        if blk is not None and audio is not None:
            if audio.size:
                blk.transcriber.insert_audio(audio)
            if blk.transcriber.pending_sec >= settings.asr_min_chunk_sec:
                res = blk.transcriber.step()
                if res["text"] and res["text"] != blk.last_emitted:
                    blk.last_emitted = res["text"]
                    evs.append(self._partial_event(blk, res))
        return evs

    def _trim(self) -> None:
        earliest = self.clock
        if self.open_block is not None:
            earliest = min(earliest, self.open_block.fed_until)
        for b in self.finalize_queue:
            earliest = min(earliest, b.fed_until)
        self.roll.drop_before(earliest)

    def _partial_event(self, blk: Block, res: dict) -> dict:
        return {
            "type": "transcript_partial", "is_final": False,
            "start": round(blk.start_sample / self.sr, 3),
            "text": res["text"], "confidence": round(blk.confidence, 3),
            "asr_confidence": res["asr_confidence"], **blk.fields(),
        }

    def _final_event(self, blk: Block, res: dict) -> dict:
        return {
            "type": "transcript_final", "is_final": True,
            "start": round(blk.start_sample / self.sr, 3),
            "end": round(blk.end_sample / self.sr, 3),
            "text": res["text"], "confidence": round(blk.confidence, 3),
            "asr_confidence": res["asr_confidence"], **blk.fields(),
        }
