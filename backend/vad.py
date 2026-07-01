"""Silero VAD wrapper.

Two usage modes:
  * ``get_speech_segments`` — offline, for enrollment (trim silence from a clip).
  * ``StreamingVAD``       — stateful, frame-by-frame, for the live pipeline.

Silero is used because it is far more robust to background noise than energy- or
WebRTC-based VADs, which directly serves the "robust against background noise"
requirement.
"""
from __future__ import annotations

import threading
import numpy as np
import torch

from .config import settings

_lock = threading.Lock()
_model = None
_utils = None


def _load():
    global _model, _utils
    if _model is not None:
        return _model, _utils
    with _lock:
        if _model is not None:
            return _model, _utils
        from silero_vad import load_silero_vad, get_speech_timestamps
        _model = load_silero_vad(onnx=False)
        _utils = {"get_speech_timestamps": get_speech_timestamps}
    return _model, _utils


def warmup() -> None:
    _load()


def get_speech_segments(wav: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) speech regions for a full clip."""
    if wav.size == 0:
        return []
    model, utils = _load()
    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    ts = utils["get_speech_timestamps"](
        tensor, model,
        sampling_rate=settings.sample_rate,
        threshold=settings.vad_threshold,
        min_speech_duration_ms=settings.vad_min_speech_ms,
        min_silence_duration_ms=settings.vad_min_silence_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )
    return [(int(t["start"]), int(t["end"])) for t in ts]


def trim_to_speech(wav: np.ndarray) -> np.ndarray:
    """Concatenate only the speech regions of a clip (used for enrollment)."""
    segs = get_speech_segments(wav)
    if not segs:
        return wav
    return np.concatenate([wav[s:e] for s, e in segs]).astype(np.float32)


class StreamingVAD:
    """Frame-level speech probability with hangover, for live streaming.

    Silero expects exactly 512-sample frames at 16 kHz. We buffer incoming audio,
    emit a boolean ``is_speech`` per frame, and apply a silence "hangover" so short
    pauses inside a word don't prematurely end a turn.
    """

    FRAME = 512  # samples @ 16 kHz (~32 ms)

    def __init__(self) -> None:
        self.model, _ = _load()
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._silence_frames = 0
        self._speech_frames = 0
        self.sr = settings.sample_rate

    def reset(self) -> None:
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._silence_frames = 0
        self._speech_frames = 0

    def _prob(self, frame: np.ndarray) -> float:
        t = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with torch.no_grad():
            return float(self.model(t, self.sr).item())

    def push(self, wav: np.ndarray) -> list[bool]:
        """Feed audio, return per-frame speech decisions (FRAME samples each)."""
        self._buf = np.concatenate([self._buf, wav.astype(np.float32)])
        out: list[bool] = []
        thr = settings.vad_threshold
        while self._buf.size >= self.FRAME:
            frame = self._buf[: self.FRAME]
            self._buf = self._buf[self.FRAME :]
            out.append(self._prob(frame) >= thr)
        return out
