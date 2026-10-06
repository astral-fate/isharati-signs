"""Batch keypoint extraction. Resumable: existing .npy files are skipped (manifest rows rebuilt from them).

Example (KArSL on Kaggle: <signer>/<signer>/<split>/<sign_id>/<sample>/<frame>.jpg):
  python scripts/lexicon/arsl/karsl_extract.py --frames --videos data/kaggle/karsl-502 --out data/karsl_kp \
      --max-per-group 3 --workers 6 \
      --regex "(?P<signer>\\d{2})[\\\\/]\\d{2}[\\\\/](?:train|test)[\\\\/](?P<sign_id>\\d{4})[\\\\/]"
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from isharati.pose.keypoints import ExtractResult, extract, extract_image_dir

FIELDS = ["clip_id", "signer", "sign_id", "npy_path", "frames", "missing_ratio", "ok"]


def _work(job):
    """Runs in a worker process: extract one clip (or reuse its cached .npy) and return its manifest row."""
    src, npy, meta, frames, src_fps, row = job
    npy, meta = Path(npy), Path(meta)
    if npy.exists() and meta.exists():
        ratio, ok = meta.read_text().split(",")
        n = len(np.load(npy))
    else:
        try:
            r = extract_image_dir(src, src_fps=src_fps) if frames else extract(src)
        except Exception as e:  # one bad clip must not stop a 75k-clip batch; it is recorded as not ok
            print(f"failed: {src}: {type(e).__name__}: {e}", flush=True)
            r = ExtractResult(np.zeros((1, 50, 3), np.float32), 1.0, False)
        np.save(npy, r.pose)
        meta.write_text(f"{r.missing_ratio},{r.ok}")
        ratio, ok, n = str(r.missing_ratio), str(r.ok), len(r.pose)
    return {**row, "frames": n, "missing_ratio": ratio, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regex", required=True,
                    help="must define named groups 'signer' and 'sign_id' matched against the relative path")
    ap.add_argument("--glob", default="**/*.mp4")
    ap.add_argument("--frames", action="store_true",
                    help="inputs are directories of frame images (one directory per sample), e.g. KArSL on Kaggle")
    ap.add_argument("--src-fps", type=float, default=30.0, help="frame rate of image sequences (--frames only)")
    ap.add_argument("--max-per-group", type=int, default=0, help="keep at most N samples per (signer, sign_id); 0 = all")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    root, out = Path(a.videos), Path(a.out)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    rx = re.compile(a.regex)
    if a.frames:
        items = sorted({p.parent for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")})
    else:
        items = sorted(root.glob(a.glob))
    jobs, per_group = [], defaultdict(int)
    for v in items:
        # match against POSIX-style paths on every OS, so one regex works on Windows, Linux and Modal
        rel = v.relative_to(root).as_posix() + ("/" if a.frames else "")
        m = rx.search(rel)
        if not m:
            print("skip (regex did not match):", rel)
            continue
        group = (m.group("signer"), m.group("sign_id"))
        if a.max_per_group and per_group[group] >= a.max_per_group:
            continue
        per_group[group] += 1
        stem = rel.rstrip("/\\") if a.frames else rel.rsplit(".", 1)[0]
        clip_id = re.sub(r"[\\/. ()]", "_", stem)
        row = {"clip_id": clip_id, "signer": group[0], "sign_id": group[1], "npy_path": f"npy/{clip_id}.npy"}
        jobs.append((str(v), str(out / "npy" / f"{clip_id}.npy"), str(out / "npy" / f"{clip_id}.meta"),
                     a.frames, a.src_fps, row))
    print(f"{len(jobs)} clips to process with {a.workers} worker(s)", flush=True)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(_work, jobs, chunksize=4))
    else:
        rows = [_work(j) for j in jobs]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
