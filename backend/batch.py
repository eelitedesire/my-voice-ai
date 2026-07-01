"""Offline processing of an uploaded recording.

Reuses the exact live pipeline (StreamingDiarizer + Whisper) over a whole file,
then returns timestamped, speaker-attributed, punctuated segments. Implemented as
a generator that yields progress so the UI can show a real progress bar.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .config import settings
from .audio import decode_file
from .diarization import StreamingDiarizer
from .enrollment import store
from . import transcription


def iter_process_file(raw: bytes, filename: str = "") -> Iterator[dict]:
    """Yield progress dicts; the final dict has stage=="done" + "segments"."""
    sr = settings.sample_rate
    yield {"stage": "decoding", "pct": 3}
    wav = decode_file(raw, filename)
    total_samples = max(1, len(wav))
    duration = len(wav) / sr
    yield {"stage": "diarizing", "pct": 8, "duration": round(duration, 2)}

    diar = StreamingDiarizer(store.snapshot())
    raw_segs: list[dict] = []
    step = sr  # 1s chunks
    for i in range(0, len(wav), step):
        for ev in diar.process(wav[i:i + step]):
            if ev["type"] == "segment":
                raw_segs.append(ev)
        yield {"stage": "diarizing", "pct": 8 + int(30 * (i + step) / total_samples)}
    for ev in diar.flush():
        if ev["type"] == "segment":
            raw_segs.append(ev)

    n = len(raw_segs)
    yield {"stage": "transcribing", "pct": 40, "done": 0, "total": n}
    segments: list[dict] = []
    for idx, ev in enumerate(raw_segs):
        audio = ev.pop("_audio")
        text = transcription.transcribe(
            audio, beam_size=settings.asr_beam_size_batch
        ).strip()
        if text:
            segments.append({
                "start": ev["start"], "end": ev["end"],
                "speaker": ev["speaker"], "speaker_id": ev["speaker_id"],
                "unknown": ev["unknown"], "text": text,
                "confidence": ev["confidence"], "asr_confidence": None,
            })
        yield {"stage": "transcribing",
               "pct": 40 + int(58 * (idx + 1) / max(1, n)),
               "done": idx + 1, "total": n}

    yield {"stage": "done", "pct": 100, "duration": round(duration, 2),
           "segments": segments}
