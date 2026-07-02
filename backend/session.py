"""Live session coordinator: binds the (unchanged) diarizer to streaming ASR.

Concurrency model (two executor-backed tasks per WebSocket):

  * ``feed()``     — fast path. Runs the diarizer on each audio chunk, buffers
                     audio, and maintains the *speaker block* timeline. Never runs
                     ASR, so speaker/VAD events stay real-time.
  * ``asr_tick()`` — heavy path. Periodically runs the live ASR engine on the
                     open block (emitting refined partials) and finalizes any
                     blocks the diarizer has closed. Only this task touches ASR,
                     so there is never concurrent decoding.

The ASR engine is Sherpa-ONNX streaming Zipformer (see ``make_transcriber``), used
for both live sessions and the offline file-upload/export path (``batch.py``).

A **block** == one contiguous turn of a single identified speaker, exactly as the
diarizer defines it. Each block owns its own transcriber; text therefore can never
mix two speakers, and speaker labels persist while text streams.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .diarization import StreamingDiarizer, UNKNOWN, UNKNOWN_NAME
from .enrollment import ProfileSnapshot


def make_transcriber():
    """Per-block live ASR transcriber (Sherpa-ONNX streaming Zipformer).

    Exposes insert_audio / step / finalize / pending_sec, so the rest of the
    session is engine-agnostic.
    """
    from .sherpa_asr import SherpaBlockTranscriber
    return SherpaBlockTranscriber()


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
    pending: bool = False        # opened but not yet confidently identified
    transcriber: object = field(default_factory=make_transcriber)
    fed_until: int = 0
    last_emitted: str = ""

    def fields(self) -> dict:
        if self.pending:
            return {"block_id": self.id, "speaker": "", "speaker_id": None,
                    "unknown": False, "pending": True}
        return {"block_id": self.id, "speaker": self.speaker,
                "speaker_id": self.speaker_id, "unknown": self.unknown, "pending": False}


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
                if ev["active"]:
                    self._begin_block(self.clock, ui)   # start transcribing immediately
                else:
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

    # ---- evidence-based block lifecycle (labels assigned lazily) ----
    def _pending_block(self, start_s: int) -> Block:
        blk = Block(id=self._next_id, speaker_id=None, speaker="", unknown=False,
                    pending=True, start_sample=start_s, end_sample=start_s,
                    confidence=0.0, fed_until=start_s)
        self._next_id += 1
        return blk

    def _labeled_block(self, ev: dict, start_s: int) -> Block:
        blk = Block(id=self._next_id, speaker_id=ev["speaker_id"], speaker=ev["speaker"],
                    unknown=ev["unknown"], pending=False, start_sample=start_s,
                    end_sample=start_s, confidence=ev.get("confidence", 0.0), fed_until=start_s)
        self._next_id += 1
        return blk

    def _enqueue_final(self, blk: Block) -> None:
        """Queue a block for finalization; a still-pending block becomes Unknown."""
        if blk.pending:
            blk.pending = False
            blk.unknown = True
            blk.speaker = UNKNOWN_NAME
            blk.speaker_id = None
        self.finalize_queue.append(blk)

    def _begin_block(self, start_s: int, ui: list[dict]) -> None:
        """Speech onset: open an unlabeled (pending) block and stream ASR now."""
        with self.lock:
            if self.open_block is not None:            # safety: close any straggler
                self.open_block.end_sample = max(self.open_block.end_sample, start_s)
                self._enqueue_final(self.open_block)
                self.open_block = None
            blk = self._pending_block(start_s)
            self.open_block = blk
        ui.append({"type": "block_open", "start": round(start_s / self.sr, 3),
                   "confidence": 0.0, **blk.fields()})

    def _on_partial(self, ev: dict, ui: list[dict]) -> None:
        named = not ev["unknown"]
        start_s = int(round(ev["start"] * self.sr))
        end_s = int(round(ev["end"] * self.sr))
        opened = relabel = None
        with self.lock:
            cur = self.open_block
            if cur is None:
                cur = self._labeled_block(ev, start_s) if named else self._pending_block(start_s)
                self.open_block = cur
                opened = cur
            elif named:
                if cur.pending:                         # evidence arrived -> label it
                    cur.pending = False
                    cur.speaker_id, cur.speaker, cur.unknown = ev["speaker_id"], ev["speaker"], False
                    relabel = cur
                elif cur.speaker_id != ev["speaker_id"]:
                    # different known speaker on a partial -> mid-turn change (defensive;
                    # the diarizer normally precedes this with a segment)
                    cur.end_sample = max(cur.end_sample, start_s)
                    self._enqueue_final(cur)
                    cur = self._labeled_block(ev, start_s)
                    self.open_block = cur
                    opened = cur
            # an UNKNOWN partial while a block is labeled is ignored (keep the label)
            if cur is not None:
                cur.end_sample = max(cur.end_sample, end_s)
                if not cur.pending:
                    cur.confidence = ev["confidence"]
        if opened is not None:
            ui.append({"type": "block_open", "start": round(opened.start_sample / self.sr, 3),
                       "confidence": opened.confidence, **opened.fields()})
        if relabel is not None:
            ui.append({"type": "block_label", "confidence": round(ev["confidence"], 3),
                       **relabel.fields()})
        ui.append({"type": "now", "confidence": ev["confidence"],
                   "speaker": ev["speaker"], "speaker_id": ev["speaker_id"], "unknown": ev["unknown"]})

    def _on_segment(self, ev: dict, ui: list[dict]) -> None:
        # A segment marks the end of one acoustic speaker span (the diarizer's
        # change-point detector fired, or the turn ended). Always finalize the
        # current block; the next partial opens a fresh block for the new speaker.
        named = not ev["unknown"]
        end_s = int(round(ev["end"] * self.sr))
        relabel = None
        with self.lock:
            cur = self.open_block
            if cur is not None:
                cur.end_sample = max(cur.end_sample, end_s)
                if named:
                    if cur.pending:
                        cur.pending = False
                        relabel = cur
                    cur.speaker_id, cur.speaker, cur.unknown = ev["speaker_id"], ev["speaker"], False
                    cur.confidence = ev["confidence"]
                    self.finalize_queue.append(cur)
                else:
                    # unknown span ends -> pending becomes Unknown; a known block keeps its label
                    self._enqueue_final(cur)
                self.open_block = None
            else:
                start_s = int(round(ev["start"] * self.sr))
                if named:
                    blk = self._labeled_block(ev, start_s); blk.end_sample = end_s
                    self.finalize_queue.append(blk)
                else:
                    blk = self._pending_block(start_s); blk.end_sample = end_s
                    self._enqueue_final(blk)
        if relabel is not None:
            ui.append({"type": "block_label", "confidence": round(ev["confidence"], 3),
                       **relabel.fields()})

    def _close_open_block(self, end_sample: int) -> None:
        with self.lock:
            if self.open_block is not None:
                self.open_block.end_sample = max(self.open_block.end_sample, end_sample)
                self._enqueue_final(self.open_block)     # pending -> Unknown at finalize
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
