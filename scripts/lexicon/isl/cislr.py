"""Indian Sign Language word signs from CISLR (Joshi et al., EMNLP 2022; Exploration-Lab/CISLR, AFL-3.0) -> ISL lexicon.

CISLR has ~4,765 words in ~7,050 clips; prototype.csv names one reference clip per word, which becomes the lexicon
entry (the other clips, test.csv, stay free for evaluating a recognizer). Each clip shows one sign, so the
hand-visible span is the sign (isharati.lexicon.asl_citizen.extract_trimmed). Words in the "religious" category are
flagged. Together with WSLP (scripts/lexicon/isl/wslp.py) this gives ~7,100 distinct ISL words.

  .venv/Scripts/python scripts/lexicon/isl/cislr.py [workers]
Output: data/lexicon_isl/cislr/{signs/<uid>.npy, entries.jsonl}
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
SRC = Path(r"D:\islam\CISLR")
ZIP = SRC / "CISLR_v1.5-a_videos" / "CISLR_v1.5-a_videos.zip"
OUT = DATA / "lexicon_isl" / "cislr"


def one(job):
    uid, member = job
    npy = OUT / "signs" / f"{uid}.npy"
    if npy.exists() or npy.with_suffix(".failed").exists():
        return uid, "skip"
    from isharati.lexicon.asl_citizen import extract_trimmed
    with zipfile.ZipFile(ZIP) as z, tempfile.TemporaryDirectory() as tmp:
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
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    protos = list(csv.DictReader(open(SRC / "prototype.csv", encoding="utf-8")))
    names = {Path(n).stem: n for n in zipfile.ZipFile(ZIP).namelist() if n.endswith(".mp4")}
    jobs = [(p["uid"], names[p["uid"]]) for p in protos if p["uid"] in names]
    print(f"{len(jobs)} of {len(protos)} prototype clips found", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(one, jobs, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    entries = [{"gloss": p["gloss"].strip().upper(), "sign_id": f"cislr_{p['uid']}", "dataset": "cislr_isl",
                "keypoints_path": f"cislr/signs/{p['uid']}.npy", "review_status": "pending",
                "is_religious": "relig" in p["category"].lower(), "is_letter": False}
               for p in protos if (OUT / "signs" / f"{p['uid']}.npy").exists()]
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    print(f"{len(entries)} ISL signs -> {OUT / 'entries.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
