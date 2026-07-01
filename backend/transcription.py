"""faster-whisper transcription wrapper (CPU / int8, singleton).

Transcribes finalized speaker segments. Kept intentionally small: the diarizer
decides *who* and *when*; Whisper only decides *what was said* for an already
speaker-attributed audio slice.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
import numpy as np

from .config import settings, MODELS_DIR

_lock = threading.Lock()
_model = None


@dataclass
class Word:
    start: float          # seconds, relative to the clip passed in
    end: float
    word: str             # includes Whisper's leading space
    prob: float


def get_model():
    """Expose the singleton model (loads on first call)."""
    return _load()


def _load():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel
        # Prefer a pre-downloaded local model dir (avoids re-download / flaky CDN).
        local_dir = MODELS_DIR / "whisper" / settings.whisper_model
        model_ref = str(local_dir) if (local_dir / "model.bin").exists() \
            else settings.whisper_model
        _model = WhisperModel(
            model_ref,
            device=settings.device,
            compute_type=settings.whisper_compute_type,
            download_root=str(MODELS_DIR / "whisper"),
        )
    return _model


def warmup() -> None:
    """Best-effort ASR warmup. A failure here (e.g. model download hiccup) must
    NOT stop the server: identification still works and ASR retries lazily."""
    if not settings.enable_transcription:
        return
    try:
        model = _load()
        model.transcribe(np.zeros(settings.sample_rate, dtype=np.float32), beam_size=1)
    except Exception as e:  # pragma: no cover
        print(f"[transcription] warmup skipped ({e}); will retry on first use")


def transcribe(wav: np.ndarray, beam_size: int = 1, initial_prompt: str | None = None) -> str:
    """Transcribe a 16 kHz mono float32 segment to text. Empty string if disabled,
    too short, or on transient model failure (diarization keeps working)."""
    if not settings.enable_transcription or wav.size < int(0.3 * settings.sample_rate):
        return ""
    try:
        model = _load()
        segments, _ = model.transcribe(
            np.ascontiguousarray(wav, dtype=np.float32),
            beam_size=beam_size,
            vad_filter=False,          # diarizer already handled VAD
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            language="en" if settings.whisper_model.endswith(".en") else None,
        )
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:  # pragma: no cover
        print(f"[transcription] failed: {e}")
        return ""


def transcribe_words(
    wav: np.ndarray, beam_size: int = 1, initial_prompt: str | None = None
) -> list[Word]:
    """Transcribe with word-level timestamps. Returns [] on failure/too-short.

    This is the primitive the streaming decoder builds on: it needs stable word
    boundaries to know which words are safe to commit vs. still being refined.
    """
    if not settings.enable_transcription or wav.size < int(0.2 * settings.sample_rate):
        return []
    try:
        model = _load()
        segments, _ = model.transcribe(
            np.ascontiguousarray(wav, dtype=np.float32),
            beam_size=beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            language="en" if settings.whisper_model.endswith(".en") else None,
        )
        words: list[Word] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append(Word(float(w.start), float(w.end), w.word, float(w.probability)))
        return words
    except Exception as e:  # pragma: no cover
        print(f"[transcription] word transcribe failed: {e}")
        return []
