"""Production-grade speaker diarization + recognition backend (SpeechBrain).

IMPORTANT: import ctranslate2 (faster-whisper's backend) *before* torch is ever
imported. On macOS, loading torch first and ctranslate2 second causes an OpenMP
runtime clash that segfaults the process. Doing this import here — before any
submodule pulls in torch — guarantees the safe ordering for every entry point.
"""
try:  # pragma: no cover - environment dependent
    import ctranslate2  # noqa: F401
except Exception:
    pass
