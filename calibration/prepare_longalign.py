"""Build the LongAlign-10k calibration cache used by jssa_calibrate.py.

Streams THUDM/LongAlign-10k in dataset order and keeps the first ``--num`` samples
whose token ``length`` is at least ``--min_tokens``, storing the user turn (long
document + instruction) and its length, one JSON object per line. The calibration
script concatenates consecutive entries into 128k-token sequences.

usage: python prepare_longalign.py [--out data/longalign_calib.jsonl]
"""
import argparse
import json
import os

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "data", "longalign_calib.jsonl"))
ap.add_argument("--num", type=int, default=64)
ap.add_argument("--min_tokens", type=int, default=8000)
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
ds = load_dataset("THUDM/LongAlign-10k", split="train", streaming=True)
n = 0
with open(a.out, "w") as f:
    for s in ds:
        if int(s.get("length", 0)) < a.min_tokens:
            continue
        user = next((m["content"] for m in s.get("messages", []) if m.get("role") == "user"),
                    None)
        if not user:
            continue
        f.write(json.dumps({"user": user, "length": int(s["length"])}, ensure_ascii=False) + "\n")
        n += 1
        if n >= a.num:
            break
print(f"wrote {n} samples to {a.out}")
