#!/usr/bin/env python3
"""Phase 0 proof-of-concept: does Sherpa-ONNX Streaming Zipformer coexist with the
existing SpeechBrain diarizer, and does it deliver low-latency live partials —
WITHOUT changing diarization output?

This script is completely isolated:
  * it does NOT import or modify the running server, frontend, or Whisper path;
  * it only *reads* the existing SpeechBrain components (diarizer, embeddings, VAD)
    to run the real pipeline;
  * Sherpa is imported ONLY in --mode combined, so --mode baseline is a clean
    reference with sherpa never loaded in the process.

Usage:
  python poc/phase0_poc.py --mode baseline --out poc/baseline.json
  python poc/phase0_poc.py --mode combined --out poc/combined.json
  python poc/phase0_poc.py --mode compare  --a poc/baseline.json --b poc/combined.json
"""
from __future__ import annotations

import os
# Precaution against a duplicate OpenMP runtime (torch libomp vs onnxruntime).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import glob
import argparse
import resource
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCRATCH_VOICES = Path(
    "/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
    "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices"
)
MODEL_DIR = ROOT / "poc" / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
SR = 16000
CHUNK = 1600  # 100 ms feed granularity (simulates real-time streaming)


# ------------------------------------------------------------------ audio utils
def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        from backend.audio import resample
        wav = resample(wav, sr, SR)
    return wav.astype(np.float32)


def build_conversation() -> tuple[np.ndarray, list[str]]:
    """Daniel -> Samantha -> Fred(unknown), separated by 0.8 s silence."""
    gap = np.zeros(int(0.8 * SR), dtype=np.float32)
    parts, labels = [], []
    for name, f in [("Daniel", "dan_test.wav"),
                    ("Samantha", "sam_test.wav"),
                    ("Fred(unknown)", "fred_test.wav")]:
        parts.append(load_wav(SCRATCH_VOICES / f)); parts.append(gap)
        labels.append(name)
    return np.concatenate(parts), labels


def build_profiles():
    """Enroll Daniel + Samantha in-memory (does NOT touch the persistent store)."""
    from backend import embeddings
    from backend.vad import trim_to_speech
    from backend.embeddings import l2_normalize
    from backend.enrollment import ProfileSnapshot

    def enroll(files):
        embs = []
        for f in files:
            emb = embeddings.embed(trim_to_speech(load_wav(SCRATCH_VOICES / f)))
            if emb is not None:
                embs.append(emb)
        return np.stack(embs)

    dan = enroll(["danA.wav", "danB.wav"])
    sam = enroll(["samA.wav", "samB.wav"])
    ids = ["dan", "sam"]
    names = ["Daniel", "Samantha"]
    per_sample = [dan, sam]
    centroids = [l2_normalize(dan.mean(0)), l2_normalize(sam.mean(0))]
    return ProfileSnapshot(ids, names, centroids, per_sample)


# ------------------------------------------------------------------ diarization
def run_diarizer(diar, conv: np.ndarray) -> tuple[list[dict], float]:
    """Feed conversation in 100 ms chunks; return (segments, diar_cpu_seconds)."""
    segs: list[dict] = []
    t = 0.0
    for i in range(0, len(conv), CHUNK):
        chunk = conv[i:i + CHUNK]
        t0 = time.perf_counter()
        for ev in diar.process(chunk):
            if ev["type"] == "segment":
                ev.pop("_audio", None)
                segs.append(_seg(ev))
        t += time.perf_counter() - t0
    t0 = time.perf_counter()
    for ev in diar.flush():
        if ev["type"] == "segment":
            ev.pop("_audio", None)
            segs.append(_seg(ev))
    t += time.perf_counter() - t0
    return segs, t


def _seg(ev: dict) -> dict:
    return {
        "speaker": ev["speaker"], "speaker_id": ev["speaker_id"],
        "unknown": ev["unknown"], "start": round(ev["start"], 2),
        "end": round(ev["end"], 2), "confidence": round(ev["confidence"], 3),
    }


# ------------------------------------------------------------------ sherpa ASR
def make_sherpa():
    import sherpa_onnx

    def pick(kind):
        c = sorted(glob.glob(str(MODEL_DIR / f"{kind}-*.onnx")))
        i8 = [x for x in c if "int8" in x]
        return (i8 or c)[0]

    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
        num_threads=2, sample_rate=SR, feature_dim=80,
        decoding_method="greedy_search", provider="cpu",
        enable_endpoint_detection=False,
    )
    return rec


# ------------------------------------------------------------------ combined run
def run_combined(conv: np.ndarray):
    from backend.diarization import StreamingDiarizer
    profiles = build_profiles()
    diar = StreamingDiarizer(profiles)

    import sherpa_onnx  # noqa: F401  (loaded here, in-process with torch)
    rec = make_sherpa()
    stream = rec.create_stream()

    segs: list[dict] = []
    partials: list[dict] = []
    per_chunk_ms: list[float] = []
    diar_cpu = asr_cpu = 0.0
    audio_s = 0.0
    last_text = ""
    ttfw = None
    wall0 = time.perf_counter()

    for i in range(0, len(conv), CHUNK):
        chunk = conv[i:i + CHUNK]
        audio_s += len(chunk) / SR
        c0 = time.perf_counter()

        # --- SpeechBrain diarization (fast path) ---
        d0 = time.perf_counter()
        for ev in diar.process(chunk):
            if ev["type"] == "segment":
                ev.pop("_audio", None)
                segs.append(_seg(ev))
        diar_cpu += time.perf_counter() - d0

        # --- Sherpa streaming ASR (heavy path) ---
        a0 = time.perf_counter()
        stream.accept_waveform(SR, chunk)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        asr_cpu += time.perf_counter() - a0

        per_chunk_ms.append((time.perf_counter() - c0) * 1000.0)
        if text and text != last_text:
            last_text = text
            wall = time.perf_counter() - wall0
            partials.append({"audio_s": round(audio_s, 2),
                             "wall_s": round(wall, 3), "text": text})
            if ttfw is None:
                ttfw = {"audio_s": round(audio_s, 2), "wall_s": round(wall, 3),
                        "first_text": text}

    # flush sherpa
    a0 = time.perf_counter()
    stream.accept_waveform(SR, np.zeros(int(0.5 * SR), dtype=np.float32))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    final_text = rec.get_result(stream)
    asr_cpu += time.perf_counter() - a0

    # flush diarizer
    d0 = time.perf_counter()
    for ev in diar.flush():
        if ev["type"] == "segment":
            ev.pop("_audio", None)
            segs.append(_seg(ev))
    diar_cpu += time.perf_counter() - d0

    audio_dur = len(conv) / SR
    arr = np.array(per_chunk_ms)
    return {
        "segments": segs,
        "sherpa_final_text": final_text,
        "sherpa_partial_updates": len(partials),
        "sherpa_partials": partials,
        "ttfw": ttfw,
        "audio_duration_s": round(audio_dur, 2),
        "metrics": {
            "diar_cpu_s": round(diar_cpu, 3),
            "asr_cpu_s": round(asr_cpu, 3),
            "diar_rtf": round(diar_cpu / audio_dur, 4),
            "asr_rtf": round(asr_cpu / audio_dur, 4),
            "combined_rtf": round((diar_cpu + asr_cpu) / audio_dur, 4),
            "per_chunk_ms_mean": round(float(arr.mean()), 2),
            "per_chunk_ms_p95": round(float(np.percentile(arr, 95)), 2),
            "per_chunk_ms_max": round(float(arr.max()), 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        },
    }


def run_baseline(conv: np.ndarray):
    from backend.diarization import StreamingDiarizer
    profiles = build_profiles()
    diar = StreamingDiarizer(profiles)
    segs, diar_cpu = run_diarizer(diar, conv)
    audio_dur = len(conv) / SR
    return {
        "segments": segs,
        "audio_duration_s": round(audio_dur, 2),
        "metrics": {
            "diar_cpu_s": round(diar_cpu, 3),
            "diar_rtf": round(diar_cpu / audio_dur, 4),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        },
    }


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text())["segments"]
    b = json.loads(Path(b_path).read_text())["segments"]
    same = a == b
    print("\n=== DIARIZATION UNCHANGED CHECK ===")
    print(f"baseline segments: {len(a)} | combined segments: {len(b)}")
    for label, segs in [("BASELINE", a), ("COMBINED", b)]:
        print(f"  {label}:")
        for s in segs:
            print(f"    [{s['start']:.2f}-{s['end']:.2f}] {s['speaker']:16s} "
                  f"conf={s['confidence']:.3f} unknown={s['unknown']}")
    print(f"\nRESULT: {'IDENTICAL ✅' if same else 'DIFFERENT ❌'}")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "combined", "compare"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    args = ap.parse_args()

    if args.mode == "compare":
        return compare(args.a, args.b)

    conv, labels = build_conversation()
    print(f"[phase0] mode={args.mode}  audio={len(conv)/SR:.1f}s  speakers={labels}")
    t0 = time.perf_counter()
    result = run_baseline(conv) if args.mode == "baseline" else run_combined(conv)
    result["wall_total_s"] = round(time.perf_counter() - t0, 2)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[phase0] wrote {args.out}")

    print("\n=== RESULT (%s) ===" % args.mode)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("sherpa_partials", "segments")}, indent=2))
    print("\nDiarizer segments:")
    for s in result["segments"]:
        print(f"  [{s['start']:.2f}-{s['end']:.2f}] {s['speaker']:16s} "
              f"conf={s['confidence']:.3f} unknown={s['unknown']}")
    if args.mode == "combined":
        print("\nSherpa partial timeline (audio_s | text):")
        for p in result["sherpa_partials"]:
            print(f"  {p['audio_s']:5.2f}s  {p['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
