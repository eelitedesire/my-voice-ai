#!/usr/bin/env python3
"""Live-transcription latency profiler. MEASURES ONLY — changes no app config.

Produces:
  1. Sherpa decode compute cost per tick.
  2. ASR emission lag under REAL-TIME feeding = wall_time_when_a_token_appears
     minus the audio time of that token (perceived "behind real speech").
  3. A sweep over decode interval (asr_tick_sec) and client chunk size to quantify
     the improvement each knob would give.
  4. Turn-start delay: when the diarizer opens a block / first partial appears,
     relative to speech onset (the block-open gate = window_sec).

All timings are wall-clock against a real-time-paced audio feed, so they reflect
what a user actually perceives.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, time, statistics, resource
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOICES = Path("/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
              "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices")
SR = 16000

from backend.config import settings
from backend import sherpa_asr


def load(f):
    w, sr = sf.read(str(VOICES / f), dtype="float32")
    return (w if w.ndim == 1 else w.mean(1)).astype(np.float32)


def speech():
    """~8s of near-continuous speech (two utterances, small gap)."""
    gap = np.zeros(int(0.3 * SR), np.float32)
    return np.concatenate([load("dan_test.wav"), gap, load("sam_test.wav")])


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 3) if xs else None


def emission_lag(conv, chunk_ms, tick_s):
    """Real-time-paced single-stream decode; return (per-token lags, compute ms)."""
    rec = sherpa_asr._load()
    stream = rec.create_stream()
    chunk_n = int(chunk_ms / 1000 * SR)
    start = time.perf_counter()
    next_tick, prev_ntok = tick_s, 0
    audio_pos = 0.0
    lags, comp = [], []
    i = 0
    while i < len(conv):
        chunk = conv[i:i + chunk_n]; i += chunk_n
        stream.accept_waveform(SR, chunk)
        audio_pos += len(chunk) / SR
        # pace to real time
        while (time.perf_counter() - start) < audio_pos:
            time.sleep(0.001)
        if (time.perf_counter() - start) >= next_tick:
            t0 = time.perf_counter()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            comp.append((time.perf_counter() - t0) * 1000)
            ts = list(rec.timestamps(stream) or [])
            wall = time.perf_counter() - start
            for k in range(prev_ntok, len(ts)):
                lags.append(wall - ts[k])       # perceived lag for this token
            prev_ntok = len(ts)
            next_tick += tick_s
    return lags, comp


def turn_start(conv_full):
    """Audio time of speech-onset vs block_open vs first partial (block-open gate)."""
    from backend.enrollment import ProfileSnapshot
    from backend.session import LiveSession
    from backend import embeddings
    from backend.vad import trim_to_speech
    from backend.embeddings import l2_normalize

    def enr(fs):
        return np.stack([e for e in (embeddings.embed(trim_to_speech(load(f))) for f in fs) if e is not None])
    dan, sam = enr(["danA.wav", "danB.wav"]), enr(["samA.wav", "samB.wav"])
    prof = ProfileSnapshot(["dan", "sam"], ["Daniel", "Samantha"],
                           [l2_normalize(dan.mean(0)), l2_normalize(sam.mean(0))], [dan, sam])
    sess = LiveSession(prof)
    CH = 1600
    audio_s = 0.0
    onset = bopen = first_partial = None
    since_tick = 0.0
    for i in range(0, len(conv_full), CH):
        chunk = conv_full[i:i + CH]; audio_s += len(chunk) / SR
        for ev in sess.feed(chunk):
            if ev["type"] == "vad" and ev["active"] and onset is None:
                onset = audio_s
            if ev["type"] == "block_open" and bopen is None:
                bopen = audio_s
        since_tick += len(chunk) / SR
        if since_tick >= settings.asr_tick_sec:
            since_tick = 0.0
            for ev in sess.asr_tick():
                if ev["type"] == "transcript_partial" and first_partial is None:
                    first_partial = audio_s
    return onset, bopen, first_partial


def main():
    chunk_ms = round(settings.client_chunk_samples / SR * 1000)
    print("=== ACTIVE CONFIG ===")
    print(f"  client chunk      : {settings.client_chunk_samples} samples @16k = {chunk_ms} ms/msg")
    print(f"  asr_tick_sec      : {settings.asr_tick_sec} s  (decode interval; decode-THEN-sleep)")
    print(f"  asr_min_chunk_sec : {settings.asr_min_chunk_sec} s")
    print(f"  window_sec        : {settings.window_sec} s  (diarizer block-open gate — UNCHANGED)")
    print(f"  hop_sec           : {settings.hop_sec} s")
    print(f"  vad FRAME         : 512 samples = 32 ms")
    print(f"  endpoint detection: OFF   sherpa threads: {settings.sherpa_num_threads}")

    sherpa_asr.warmup()
    conv = speech()
    audio_dur = len(conv) / SR

    print("\n=== [1] Sherpa decode compute per tick (active config) ===")
    lags, comp = emission_lag(conv, chunk_ms=chunk_ms, tick_s=settings.asr_tick_sec)
    rtf = sum(comp) / 1000 / audio_dur
    print(f"  decode compute: mean {statistics.mean(comp):.1f} ms  "
          f"p95 {pct([c for c in comp],95):.1f} ms  (n={len(comp)})   ASR RTF: {rtf:.3f}")

    print("\n=== [2] ASR emission lag (perceived 'behind real speech') ===")
    print(f"  ACTIVE (chunk {chunk_ms}ms, tick {settings.asr_tick_sec}s): "
          f"mean {statistics.mean(lags):.3f}s  p50 {pct(lags,50)}s  p95 {pct(lags,95)}s")

    print("\n=== [3] Sweep decode interval + chunk size (lower = fresher) ===")
    print(f"  {'chunk':>7} {'tick':>6} | {'mean lag':>9} {'p50':>7} {'p95':>7}")
    grid = [(64, 0.45), (64, 0.30), (64, 0.20), (64, 0.15), (64, 0.10),
            (32, 0.20), (32, 0.10)]
    floor = None
    for cms, tk in grid:
        lg, _ = emission_lag(conv, chunk_ms=cms, tick_s=tk)
        m = statistics.mean(lg)
        print(f"  {cms:>5}ms {tk:>5.2f}s | {m:>8.3f}s {pct(lg,50):>6}s {pct(lg,95):>6}s")
        floor = m if floor is None else min(floor, m)
    print(f"  --> approx model-inherent floor (lookahead+compute): ~{floor:.3f}s")

    print("\n=== [4] Turn-start delay (block-open gate = window_sec) ===")
    gap = np.zeros(int(0.8 * SR), np.float32)
    full = np.concatenate([load("dan_test.wav"), gap, load("sam_test.wav")])
    onset, bopen, fp = turn_start(full)
    print(f"  speech onset (audio) : {onset}s")
    print(f"  block_open (audio)   : {bopen}s   -> gate ~= {round((bopen or 0)-(onset or 0),2)}s after onset")
    print(f"  first partial (audio): {fp}s   -> first-word delay ~= {round((fp or 0)-(onset or 0),2)}s")

    ttfw = round((fp or 0) - (onset or 0), 2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print("\n=== SUMMARY (active config) ===")
    print(f"  Time to First Word (turn start) : {ttfw} s")
    print(f"  Mean emission latency           : {statistics.mean(lags):.3f} s")
    print(f"  P95 emission latency            : {pct(lags,95)} s")
    print(f"  Decode compute / tick           : {statistics.mean(comp):.1f} ms")
    print(f"  ASR real-time factor            : {rtf:.3f}  ({'faster' if rtf<1 else 'slower'} than realtime)")
    print(f"  Peak RSS (torch+sherpa loaded)  : {rss:.0f} MB")


if __name__ == "__main__":
    raise SystemExit(main())
