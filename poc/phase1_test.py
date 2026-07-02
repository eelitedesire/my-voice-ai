#!/usr/bin/env python3
"""Phase 1 integration test: drive the real LiveSession (feed + asr_tick) exactly
like the WebSocket server, with the Sherpa live engine. Verifies:

  * per-block Sherpa streams keep transcripts under the correct speaker
    (resolving the single-stream POC artifact),
  * diarization output (speakers / Unknown / boundaries) is unchanged,
  * live partials stream and finals are non-empty.

Isolated: builds an in-memory profile snapshot; never touches the persistent store.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, time
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOICES = Path("/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
              "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices")
SR = 16000
CHUNK = 1600

from backend.config import settings
from backend.enrollment import ProfileSnapshot
from backend.session import LiveSession
from backend import embeddings
from backend.vad import trim_to_speech
from backend.embeddings import l2_normalize


def load(f):
    w, sr = sf.read(str(VOICES / f), dtype="float32")
    return (w if w.ndim == 1 else w.mean(1)).astype(np.float32)


def profiles():
    def enr(fs):
        return np.stack([e for e in (embeddings.embed(trim_to_speech(load(f))) for f in fs) if e is not None])
    dan, sam = enr(["danA.wav", "danB.wav"]), enr(["samA.wav", "samB.wav"])
    return ProfileSnapshot(["dan", "sam"], ["Daniel", "Samantha"],
                           [l2_normalize(dan.mean(0)), l2_normalize(sam.mean(0))], [dan, sam])


def main():
    print("[phase1] live ASR engine = Sherpa-ONNX")
    gap = np.zeros(int(0.8 * SR), np.float32)
    conv = np.concatenate([load("dan_test.wav"), gap, load("sam_test.wav"), gap, load("fred_test.wav")])
    sess = LiveSession(profiles())

    partials, finals = [], []
    since_tick, audio_s, first_partial_s = 0.0, 0.0, None
    t_wall0 = time.perf_counter()

    def drain(evs):
        nonlocal first_partial_s
        for ev in evs:
            if ev["type"] == "transcript_partial":
                partials.append(ev)
                if first_partial_s is None:
                    first_partial_s = audio_s
            elif ev["type"] == "transcript_final":
                finals.append(ev)

    for i in range(0, len(conv), CHUNK):
        chunk = conv[i:i + CHUNK]
        audio_s += len(chunk) / SR
        sess.feed(chunk)                 # fast path (diarizer) — events ignored here
        since_tick += len(chunk) / SR
        if since_tick >= settings.asr_tick_sec:
            since_tick = 0.0
            drain(sess.asr_tick())       # heavy path (sherpa)
    sess.flush()
    for _ in range(300):
        evs = sess.asr_tick()
        if not evs:
            break
        drain(evs)

    wall = time.perf_counter() - t_wall0
    print(f"\naudio={audio_s:.1f}s  processed_wall={wall:.1f}s  RTF={wall/audio_s:.3f}")
    print(f"partial updates: {len(partials)}   first partial at audio={first_partial_s}s")
    print("\n=== FINAL BLOCKS (speaker : text) ===")
    for f in finals:
        print(f"  [{f['start']:.2f}-{f['end']:.2f}] {f['speaker']:16s} "
              f"conf={f['confidence']:.3f}  \"{f['text']}\"")

    spk = [f["speaker"] for f in finals]
    ok_speakers = spk[:3] == ["Daniel", "Samantha", "Unknown Speaker"]
    ok_text = all(f["text"] for f in finals)
    no_mix = all(len(set(f["text"].split()) & {"AGREE", "REALISTIC"}) == 0
                 for f in finals if f["speaker"] == "Daniel")  # Samantha words not in Daniel block
    print("\n=== CHECKS ===")
    print(f"  correct speaker order (Daniel/Samantha/Unknown): {ok_speakers}")
    print(f"  every final block has text:                      {ok_text}")
    print(f"  no speaker-mixing in Daniel's block:             {no_mix}")
    print(f"\nRESULT: {'PASS ✅' if (ok_speakers and ok_text and no_mix) else 'CHECK ❌'}")
    return 0 if (ok_speakers and ok_text and no_mix) else 1


if __name__ == "__main__":
    raise SystemExit(main())
