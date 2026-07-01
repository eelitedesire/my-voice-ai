"""Transcript exporters: TXT, JSON, SRT, VTT.

A *segment* is a plain dict with keys:
    start (float s), end (float s), speaker (str), speaker_id (str|None),
    unknown (bool), text (str), confidence (float|None), asr_confidence (float|None)

The same segment shape is produced by both the live session (finalized blocks)
and the batch file processor, so exports are identical across both paths.
"""
from __future__ import annotations

import json


def _clock(seconds: float, sep: str = ",") -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_txt(segments: list[dict]) -> str:
    """Human-readable, grouped like a meeting transcript."""
    out = []
    for seg in segments:
        out.append(_hms(seg["start"]))
        out.append(f"{seg['speaker']}:")
        out.append((seg.get("text") or "").strip())
        out.append("")
    return "\n".join(out).strip() + "\n"


def to_json(segments: list[dict]) -> str:
    payload = {
        "version": 1,
        "segment_count": len(segments),
        "segments": [
            {
                "index": i,
                "start": round(float(s["start"]), 3),
                "end": round(float(s.get("end", s["start"])), 3),
                "speaker": s["speaker"],
                "speaker_id": s.get("speaker_id"),
                "unknown": bool(s.get("unknown", False)),
                "text": (s.get("text") or "").strip(),
                "confidence": s.get("confidence"),
                "asr_confidence": s.get("asr_confidence"),
            }
            for i, s in enumerate(segments)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _clock(seg["start"], ",")
        end = _clock(seg.get("end", seg["start"] + 2), ",")
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{seg['speaker']}: {(seg.get('text') or '').strip()}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = _clock(seg["start"], ".")
        end = _clock(seg.get("end", seg["start"] + 2), ".")
        lines.append(f"{start} --> {end}")
        lines.append(f"<v {seg['speaker']}>{(seg.get('text') or '').strip()}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


EXPORTERS = {"txt": to_txt, "json": to_json, "srt": to_srt, "vtt": to_vtt}
MEDIA_TYPES = {
    "txt": "text/plain", "json": "application/json",
    "srt": "application/x-subrip", "vtt": "text/vtt",
}


def export(segments: list[dict], fmt: str) -> tuple[str, str]:
    """Return (content, media_type). Raises KeyError on unknown format."""
    fmt = fmt.lower()
    return EXPORTERS[fmt](segments), MEDIA_TYPES[fmt]
