"""Audio decoding, resampling and normalization helpers.

Internally the whole pipeline works with float32 mono waveforms at
``settings.sample_rate`` (16 kHz), represented as 1-D numpy arrays in [-1, 1].
"""
from __future__ import annotations

import io
import numpy as np

from .config import settings

TARGET_SR = settings.sample_rate


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian int16 PCM bytes -> float32 [-1, 1] mono array."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return np.clip(audio, -1.0, 1.0)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """High-quality resampling via torchaudio (sinc interpolation)."""
    if orig_sr == target_sr or wav.size == 0:
        return wav.astype(np.float32, copy=False)
    import torch
    import torchaudio.functional as AF
    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    out = AF.resample(t, orig_freq=orig_sr, new_freq=target_sr)
    return out.numpy().astype(np.float32)


def decode_file(raw: bytes, filename: str = "") -> np.ndarray:
    """Decode an uploaded audio file to 16 kHz float32 mono.

    soundfile (libsndfile) handles wav/flac/ogg and, on recent builds, mp3.
    torchaudio is tried as a fallback for other compressed formats. The web
    recorder always uploads WAV, so the primary path needs no extra codecs.
    """
    # Primary: soundfile.
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return resample(_to_mono(wav), sr, TARGET_SR)
    except Exception:
        pass
    # Fallback: torchaudio (uses ffmpeg/sox backend if available).
    try:
        import torch  # noqa: F401
        import torchaudio
        wav_t, sr = torchaudio.load(io.BytesIO(raw))
        wav = wav_t.mean(dim=0).numpy().astype(np.float32)
        return resample(wav, sr, TARGET_SR)
    except Exception as e:
        raise ValueError(
            f"Unsupported/undecodable audio '{filename}'. Use WAV/FLAC/OGG "
            f"(or install ffmpeg for mp3/m4a). Underlying error: {e}"
        )


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def rms_normalize(wav: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Loudness-normalize so different mics/levels are comparable before embedding.

    Robust to silence (returns input unchanged when energy is negligible).
    """
    if wav.size == 0:
        return wav
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-6:
        return wav
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    out = wav * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.999:  # prevent clipping introduced by gain
        out = out * (0.999 / peak)
    return out.astype(np.float32)
