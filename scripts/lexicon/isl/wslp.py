"""Indian Sign Language word signs from the WSLP 2026 word-recognition training set -> an ISL lexicon.

Exploration-Lab/WSLP (Hugging Face), Shared_task_WR/Train-Dataset: ~4,400 clips of ~4,360 distinct words, each
labelled with its English word (train_WR.csv: uid, word). Each clip shows one sign, so the hand-visible span is the
sign (isharati.lexicon.asl_citizen.extract_trimmed). Licence CC BY-NC-ND 4.0: research use, poses are not published.
ISL is closely related to Pakistani Sign Language (the Indo-Pakistani sign language family), so the Urdu demo can
use these where no PSL sign exists, labelled as ISL.

  .venv/Scripts/python scripts/lexicon/isl/wslp.py [workers]
Output: data/lexicon_isl/wslp/{signs/<uid>.npy, entries.jsonl}
"""
import csv
import json
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
SRC = Path(r"D:\islam\WSLP\Shared_task_WR\Train-Dataset")
OUT = DATA / "lexicon_isl" / "wslp"


def members():
    z = zipfile.ZipFile(SRC / "train.zip")
    by_uid = {}
    for n in z.namelist():
        stem = Path(n).stem
        if n.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
            by_uid[stem] = n
    return by_uid


def one(job):
    uid, member = job
    npy = OUT / "signs" / f"{uid}.npy"
    if npy.exists() or npy.with_suffix(".failed").exists():
        return uid, "skip"
    from isharati.lexicon.asl_citizen import extract_trimmed
    with zipfile.ZipFile(SRC / "train.zip") as z, tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / Path(member).name
        dst.write_bytes(z.read(member))
        try:
            pose = extract_trimmed(dst)
        except Exception as e:  # a broken clip: record it and go on
            pose = None
            print(uid, "failed:", e, flush=True)
    if pose is None or len(pose) < 6:
        npy.with_suffix(".failed").write_text("no usable signing", encoding="utf-8")
        return uid, "failed"
    np.save(npy, pose.astype(np.float32))
    return uid, "ok"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    words = {r["uid"]: r["word"].strip() for r in csv.DictReader(open(SRC / "train_WR.csv", encoding="utf-8"))}
    by_uid = members()
    jobs = [(u, by_uid[u]) for u in words if u in by_uid]
    print(f"{len(jobs)} of {len(words)} labelled clips found in the zip", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        for uid, status in pool.map(one, jobs, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    entries = []
    for uid, word in words.items():
        if (OUT / "signs" / f"{uid}.npy").exists() and word:
            entries.append({"gloss": word.upper(), "sign_id": f"wslp_{uid}", "dataset": "wslp_isl",
                            "keypoints_path": f"wslp/signs/{uid}.npy", "review_status": "pending",
                            "is_religious": False, "is_letter": False})
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    print(f"{len(entries)} ISL signs -> {OUT / 'entries.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
