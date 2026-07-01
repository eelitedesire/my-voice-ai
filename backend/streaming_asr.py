"""Incremental (streaming) speech-to-text on top of faster-whisper.

Whisper is not natively streaming, so we use the well-known **LocalAgreement-2**
policy (Macháček et al., "Whisper-Streaming"):

  * Keep a rolling audio buffer for the *current speaker block*.
  * Every tick, re-transcribe the buffer with word-level timestamps.
  * A word is **committed** (finalized, never changes) only once it appears
    identically as the next unconfirmed word in two consecutive hypotheses.
  * The still-unstable tail is shown as a **partial** and may be revised.
  * Once words are committed, their audio is dropped from the buffer so cost
    stays bounded no matter how long the block runs.

One ``StreamingTranscriber`` instance corresponds to exactly one speaker block,
which is how live transcription stays synchronized with diarization.
"""
from __future__ import annotations

import numpy as np

from .config import settings
from . import transcription


def _norm(word: str) -> str:
    return word.strip().lower().strip(".,!?;:\"'()[]").strip()


class HypothesisBuffer:
    """Commits the longest agreeing prefix across consecutive hypotheses."""

    def __init__(self) -> None:
        self.committed: list[transcription.Word] = []   # finalized words
        self.buffer: list[transcription.Word] = []      # previous hypothesis (tail)
        self.last_committed_end: float = 0.0

    def insert(self, words: list[transcription.Word]) -> None:
        # Only consider words at/after what we've already committed (small tolerance).
        self._incoming = [w for w in words if w.end > self.last_committed_end - 0.1]

    def flush(self) -> list[transcription.Word]:
        """Return newly committed words (agreement between prev + current)."""
        newly: list[transcription.Word] = []
        incoming = getattr(self, "_incoming", [])
        i = 0
        while i < len(incoming) and i < len(self.buffer):
            if _norm(incoming[i].word) == _norm(self.buffer[i].word) and _norm(incoming[i].word):
                w = incoming[i]
                newly.append(w)
                self.last_committed_end = w.end
                i += 1
            else:
                break
        self.committed.extend(newly)
        # everything from the agreed point onward becomes the new comparison buffer
        self.buffer = incoming[i:]
        self._incoming = []
        return newly

    def finalize(self) -> list[transcription.Word]:
        """Commit the remaining tail (called when the block ends)."""
        rest = self.buffer
        self.committed.extend(rest)
        if rest:
            self.last_committed_end = rest[-1].end
        self.buffer = []
        return rest


class StreamingTranscriber:
    def __init__(self, min_chunk_sec: float | None = None) -> None:
        self.sr = settings.sample_rate
        self.min_chunk = min_chunk_sec if min_chunk_sec is not None else settings.asr_min_chunk_sec
        self._audio = np.zeros(0, dtype=np.float32)   # buffer not yet committed
        self._buf_offset = 0.0                        # abs time (s) of _audio[0]
        self._hyp = HypothesisBuffer()

    # ----- feeding -----
    def insert_audio(self, wav: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, wav.astype(np.float32)])

    @property
    def pending_sec(self) -> float:
        return self._audio.size / self.sr

    # ----- decoding -----
    def step(self) -> dict:
        """Run one incremental pass. Returns {committed, partial, text}."""
        if self.pending_sec < self.min_chunk:
            return self._render([])
        words = transcription.transcribe_words(
            self._audio, beam_size=settings.asr_beam_size,
            initial_prompt=self._committed_text() or None,
        )
        # rebase word times to absolute
        for w in words:
            w.start += self._buf_offset
            w.end += self._buf_offset
        self._hyp.insert(words)
        newly = self._hyp.flush()
        self._trim_after_commit()
        return self._render(newly)

    def finalize(self) -> dict:
        """Flush remaining audio + commit the tail. Called at block end."""
        if self.pending_sec >= 0.2:
            words = transcription.transcribe_words(
                self._audio, beam_size=settings.asr_beam_size,
                initial_prompt=self._committed_text() or None,
            )
            for w in words:
                w.start += self._buf_offset
                w.end += self._buf_offset
            self._hyp.insert(words)
            self._hyp.flush()
        newly = self._hyp.finalize()
        return self._render(newly, final=True)

    # ----- internals -----
    def _trim_after_commit(self) -> None:
        cut_time = self._hyp.last_committed_end - self._buf_offset
        if cut_time <= 0:
            return
        cut = min(len(self._audio), int(cut_time * self.sr))
        if cut > 0:
            self._audio = self._audio[cut:]
            self._buf_offset += cut / self.sr

    def _committed_text(self) -> str:
        return "".join(w.word for w in self._hyp.committed).strip()

    def _partial_text(self) -> str:
        return "".join(w.word for w in self._hyp.buffer).strip()

    def _render(self, newly, final: bool = False) -> dict:
        committed = self._committed_text()
        partial = "" if final else self._partial_text()
        full = (committed + (" " + partial if partial else "")).strip()
        probs = [w.prob for w in self._hyp.committed]
        return {
            "committed": committed,
            "partial": partial,
            "text": full,
            "asr_confidence": round(float(np.mean(probs)), 3) if probs else None,
            "final": final,
        }
