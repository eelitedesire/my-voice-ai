"""Live ASR engine: Sherpa-ONNX Streaming Zipformer (transducer).

The single ASR engine for the whole app (live sessions + file upload). It is a
drop-in for the per-block transcriber interface consumed by ``backend.session`` —
one ``SherpaBlockTranscriber`` per speaker block:

    insert_audio(wav)   feed 16 kHz mono float32 audio for this block
    step()   -> dict    incremental decode, returns the running hypothesis
    finalize() -> dict  flush + return the final hypothesis for the block
    pending_sec         seconds of audio fed since the last decode

Design notes:
  * The Zipformer is *natively* streaming, so there is no re-decoding and no
    LocalAgreement machinery — ``get_result`` returns the current hypothesis.
  * One shared ``OnlineRecognizer`` (singleton); each block owns its own decode
    *stream* (``create_stream``), so text can never leak across speakers.
  * Output is uppercase and unpunctuated by design — no post-processing is applied
    (per project decision to prioritise latency/throughput).

macOS note: pin ``sherpa-onnx==1.10.46`` on macOS 13 (Ventura); newer wheels bundle
an onnxruntime built for macOS >= 15 and fail to load here.
"""
from __future__ import annotations

import glob
import threading
from pathlib import Path

import numpy as np

from .config import settings

_lock = threading.Lock()
_recognizer = None


def _pick(model_dir: Path, kind: str) -> str:
    """Return the model file for `kind` (encoder/decoder/joiner), preferring int8."""
    cand = sorted(glob.glob(str(model_dir / f"{kind}-*.onnx")))
    if not cand:
        raise FileNotFoundError(f"No '{kind}-*.onnx' in {model_dir}")
    int8 = [c for c in cand if "int8" in c]
    return (int8 or cand)[0]


def _load():
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    with _lock:
        if _recognizer is not None:
            return _recognizer
        import sherpa_onnx
        model_dir = Path(settings.sherpa_model_dir)
        tokens = model_dir / "tokens.txt"
        if not tokens.exists():
            raise FileNotFoundError(
                f"Sherpa model not found in {model_dir}. Run scripts/download_sherpa.sh"
            )
        _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=_pick(model_dir, "encoder"),
            decoder=_pick(model_dir, "decoder"),
            joiner=_pick(model_dir, "joiner"),
            num_threads=settings.sherpa_num_threads,
            sample_rate=settings.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
            enable_endpoint_detection=False,  # the SpeechBrain diarizer owns boundaries
        )
    return _recognizer


def warmup() -> None:
    """Best-effort load + a tiny decode so the first block is fast. Never fatal."""
    try:
        rec = _load()
        s = rec.create_stream()
        s.accept_waveform(settings.sample_rate,
                          np.zeros(settings.sample_rate // 2, dtype=np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
    except Exception as e:  # pragma: no cover
        print(f"[sherpa_asr] warmup skipped ({e}); will retry on first use")


def transcribe_offline(wav: np.ndarray) -> str:
    """One-shot transcription of a whole clip (file-upload / export path).

    Runs the same streaming Zipformer over a complete segment on a throwaway
    stream. Returns "" for empty/too-short audio.
    """
    if wav is None or wav.size < int(0.2 * settings.sample_rate):
        return ""
    rec = _load()
    s = rec.create_stream()
    s.accept_waveform(settings.sample_rate, np.ascontiguousarray(wav, dtype=np.float32))
    s.accept_waveform(settings.sample_rate,
                      np.zeros(int(0.3 * settings.sample_rate), dtype=np.float32))
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    return (rec.get_result(s) or "").strip()


class SherpaBlockTranscriber:
    """One streaming decode stream, bound to a single speaker block."""

    def __init__(self) -> None:
        self.sr = settings.sample_rate
        self._rec = _load()
        self._stream = self._rec.create_stream()
        self._pending = 0            # samples fed since last decode
        self._finished = False

    # ----- feeding -----
    def insert_audio(self, wav: np.ndarray) -> None:
        if wav is None or wav.size == 0 or self._finished:
            return
        self._stream.accept_waveform(self.sr, np.ascontiguousarray(wav, dtype=np.float32))
        self._pending += wav.size

    @property
    def pending_sec(self) -> float:
        return self._pending / self.sr

    # ----- decoding -----
    def step(self) -> dict:
        self._decode()
        return self._render(final=False)

    def finalize(self) -> dict:
        if not self._finished:
            # a little tail padding flushes the model's right-context lookahead
            self._stream.accept_waveform(
                self.sr, np.zeros(int(0.3 * self.sr), dtype=np.float32))
            self._stream.input_finished()
            self._finished = True
        self._decode()
        return self._render(final=True)

    # ----- internals -----
    def _decode(self) -> None:
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        self._pending = 0

    def _render(self, final: bool) -> dict:
        text = (self._rec.get_result(self._stream) or "").strip()
        return {
            "committed": text if final else "",
            "partial": "" if final else text,
            "text": text,
            "asr_confidence": self._confidence(),
            "final": final,
        }

    def _confidence(self) -> float | None:
        try:
            probs = self._rec.ys_probs(self._stream)  # per-token log-probs
            if not probs:
                return None
            return round(float(np.exp(np.mean(probs))), 3)
        except Exception:
            return None
