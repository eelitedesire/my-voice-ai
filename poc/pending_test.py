#!/usr/bin/env python3
"""Verify evidence-based labeling: one block per turn, pending -> label (no
Unknown->Named split), pending -> Unknown only if never identified."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOICES = Path("/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
              "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices")
SR = 16000

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


gap = np.zeros(int(0.8 * SR), np.float32)
conv = np.concatenate([load("dan_test.wav"), gap, load("sam_test.wav"), gap, load("fred_test.wav")])
sess = LiveSession(profiles())

events = []
since = 0.0
for i in range(0, len(conv), 1600):
    ch = conv[i:i + 1600]
    events += [e for e in sess.feed(ch) if e["type"] in ("block_open", "block_label")]
    since += len(ch) / SR
    if since >= settings.asr_tick_sec:
        since = 0.0
        events += [e for e in sess.asr_tick() if e["type"] == "transcript_final"]
sess.flush()
for _ in range(300):
    evs = sess.asr_tick()
    if not evs:
        break
    events += [e for e in evs if e["type"] == "transcript_final"]

print("=== EVENT SEQUENCE (block lifecycle) ===")
for e in events:
    if e["type"] == "block_open":
        print(f"  block_open   #{e['block_id']}  pending={e.get('pending')}  speaker={e['speaker']!r}")
    elif e["type"] == "block_label":
        print(f"  block_label  #{e['block_id']}  -> {e['speaker']}")
    elif e["type"] == "transcript_final":
        print(f"  FINAL        #{e['block_id']}  {e['speaker']:16s} unknown={e['unknown']}  {e['text']!r}")

opens = [e for e in events if e["type"] == "block_open"]
finals = [e for e in events if e["type"] == "transcript_final"]
labels = [e for e in events if e["type"] == "block_label"]
spk = [f["speaker"] for f in finals]
print("\n=== CHECKS ===")
print(f"  blocks opened : {len(opens)}   (expected 3, one per turn)")
print(f"  all opened pending (no premature label): {all(o.get('pending') for o in opens)}")
print(f"  finals        : {len(finals)}  speakers={spk}")
print(f"  known speakers got a block_label: {sorted(l['speaker'] for l in labels)}")
print(f"  no Unknown->Named double blocks (3 finals, Daniel/Samantha/Unknown): "
      f"{spk == ['Daniel','Samantha','Unknown Speaker']}")
ok = len(opens) == 3 and all(o.get('pending') for o in opens) and spk == ['Daniel','Samantha','Unknown Speaker']
print(f"\nRESULT: {'PASS ✅' if ok else 'CHECK ❌'}")
