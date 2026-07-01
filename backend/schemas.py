"""Pydantic request/response models and WebSocket message contracts."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------- REST ----------

class SpeakerSummary(BaseModel):
    id: str
    name: str
    num_samples: int
    created_at: float
    updated_at: float


class SpeakerListResponse(BaseModel):
    speakers: list[SpeakerSummary]


class EnrollResponse(BaseModel):
    speaker: SpeakerSummary
    added_samples: int
    skipped_samples: int
    message: str


class ConfigUpdate(BaseModel):
    """Any subset of TUNABLE_FIELDS."""
    vad_threshold: Optional[float] = None
    id_threshold: Optional[float] = None
    scoring: Optional[str] = None
    ema_alpha: Optional[float] = None
    switch_margin: Optional[float] = None
    min_switch_windows: Optional[int] = None
    min_segment_sec: Optional[float] = None
    window_sec: Optional[float] = None
    hop_sec: Optional[float] = None
    min_embed_sec: Optional[float] = None
    finalize_silence_ms: Optional[int] = None
    enable_transcription: Optional[bool] = None


# ---------- WebSocket outbound messages (documented for the frontend) ----------
# {"type": "ready"}
# {"type": "partial", "speaker": "Alice", "speaker_id": "...", "confidence": 0.61,
#                     "start": 12.3, "end": 13.1, "unknown": false}
# {"type": "segment", "speaker": "Alice", "speaker_id": "...", "confidence": 0.63,
#                     "text": "hello there", "start": 12.3, "end": 15.0, "unknown": false}
# {"type": "vad", "active": true}
# {"type": "error", "message": "..."}
