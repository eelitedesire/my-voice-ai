"""Central configuration for the speaker recognition system.

Every tunable lives here and can be overridden via environment variables so the
system can be tuned in production without code changes. Values are read once at
import time; the live-tuning endpoint mutates the ``settings`` singleton in place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SPEAKERS_DIR = DATA_DIR / "speakers"
MODELS_DIR = BASE_DIR / "models"

for _d in (DATA_DIR, SPEAKERS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    # ---- Audio ----
    sample_rate: int = 16000            # everything internal runs at 16 kHz mono

    # ---- Models ----
    embedding_model: str = _env_str("EMBED_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    embedding_dim: int = 192
    whisper_model: str = _env_str("WHISPER_MODEL", "base.en")   # tiny/base/small(.en)
    whisper_compute_type: str = _env_str("WHISPER_COMPUTE", "int8")  # CPU-friendly
    device: str = _env_str("DEVICE", "cpu")

    # ---- Voice Activity Detection ----
    vad_threshold: float = _env_float("VAD_THRESHOLD", 0.5)
    vad_min_speech_ms: int = _env_int("VAD_MIN_SPEECH_MS", 250)
    vad_min_silence_ms: int = _env_int("VAD_MIN_SILENCE_MS", 350)
    vad_speech_pad_ms: int = _env_int("VAD_SPEECH_PAD_MS", 120)

    # ---- Streaming diarization windows ----
    window_sec: float = _env_float("WINDOW_SEC", 1.5)      # embedding window length
    hop_sec: float = _env_float("HOP_SEC", 0.75)           # window step
    min_embed_sec: float = _env_float("MIN_EMBED_SEC", 0.6)  # skip windows shorter than this

    # ---- Identification ----
    # Cosine similarity (embeddings are L2-normalized). ECAPA: same-spk ~0.5-0.75,
    # diff-spk ~0.0-0.3. Below id_threshold -> "Unknown Speaker".
    id_threshold: float = _env_float("ID_THRESHOLD", 0.35)
    # Score aggregation across a speaker's enrolled samples: "centroid" | "max" | "mean"
    scoring: str = _env_str("SCORING", "max")

    # ---- Stability / anti-flicker (hysteresis) ----
    # EMA smoothing factor for per-speaker similarity scores.
    ema_alpha: float = _env_float("EMA_ALPHA", 0.6)
    # A different speaker must beat the current one by this cosine margin...
    switch_margin: float = _env_float("SWITCH_MARGIN", 0.06)
    # ...for this many consecutive windows before the active speaker actually switches.
    min_switch_windows: int = _env_int("MIN_SWITCH_WINDOWS", 2)
    # Minimum duration before a segment can be finalized/emitted as a turn.
    min_segment_sec: float = _env_float("MIN_SEGMENT_SEC", 0.8)

    # ---- Transcription ----
    enable_transcription: bool = _env_str("ENABLE_ASR", "1") == "1"
    # Finalize (and transcribe) a turn after this much trailing silence.
    finalize_silence_ms: int = _env_int("FINALIZE_SILENCE_MS", 700)

    def public(self) -> dict:
        """Config safe to expose to the UI / tuning endpoint."""
        d = asdict(self)
        return d


settings = Settings()

# Fields the /config endpoint is allowed to mutate at runtime.
TUNABLE_FIELDS = {
    "vad_threshold", "id_threshold", "scoring", "ema_alpha", "switch_margin",
    "min_switch_windows", "min_segment_sec", "window_sec", "hop_sec",
    "min_embed_sec", "finalize_silence_ms", "enable_transcription",
}
