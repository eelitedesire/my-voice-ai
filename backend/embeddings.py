"""SpeechBrain ECAPA-TDNN speaker embedding extractor.

Loaded once as a process-wide singleton. Produces L2-normalized 192-dim
embeddings suitable for cosine-similarity comparison. Loudness normalization is
applied first so embeddings are robust to microphone gain / recording level.
"""
from __future__ import annotations

import threading
import numpy as np
import torch

from .config import settings, MODELS_DIR
from .audio import rms_normalize

_lock = threading.Lock()
_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from speechbrain.inference.speaker import EncoderClassifier
        _model = EncoderClassifier.from_hparams(
            source=settings.embedding_model,
            savedir=str(MODELS_DIR / "ecapa"),
            run_opts={"device": settings.device},
        )
        _model.eval()
    return _model


def warmup() -> None:
    """Force model load + a dummy forward pass so first real request is fast."""
    model = _load()
    dummy = torch.zeros(1, settings.sample_rate)
    with torch.no_grad():
        model.encode_batch(dummy)


def embed(wav: np.ndarray) -> np.ndarray | None:
    """Return an L2-normalized embedding for a 16 kHz mono float32 waveform.

    Returns ``None`` if the clip is too short to embed reliably.
    """
    if wav is None or wav.size < int(settings.min_embed_sec * settings.sample_rate):
        return None
    wav = rms_normalize(wav)
    model = _load()
    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(tensor)  # (1, 1, D)
    vec = emb.squeeze().cpu().numpy().astype(np.float32)
    return l2_normalize(vec)


def embed_batch(wavs: list[np.ndarray]) -> list[np.ndarray | None]:
    """Embed several equal-purpose clips; skips clips that are too short."""
    return [embed(w) for w in wavs]


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return (vec / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-L2-normalized vectors (== dot product)."""
    return float(np.dot(a, b))
