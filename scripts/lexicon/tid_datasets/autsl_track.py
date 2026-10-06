"""Holistic tracks (body, hands and face) of the fetched AUTSL clips, with scripts/face/track.py on the 25 fps grid.

The lexicon clip and its face are both cut from this one track (autsl_build.py), so they share their frames exactly.
Runs in the MediaPipe venv; k/n splits the clips over parallel processes.

--follow keeps rescanning the videos folder while autsl_fetch.py is still writing (until its fetch.done), the clips
split over processes by sample number so a rescan never moves a clip to another process. A clip track.py fails on
gets tracks/<sample>.failed (autsl_build.py rejects it).

  "<gathring data>/.venv/Scripts/python" scripts/lexicon/tid_datasets/autsl_track.py [k n] [--follow]
Output: F:/tid_datasets/autsl/tracks/<sample>.npz (track.py's format)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "face"))
from track import track  # noqa: E402

AUTSL = Path("F:/tid_datasets/autsl")


def follow(k, n):
    out_dir = AUTSL / "tracks"
    out_dir.mkdir(exist_ok=True)
    while True:
        done = (AUTSL / "fetch.done").exists()  # read before the scan: clips written before it are all seen
        todo = [v for v in sorted((AUTSL / "videos").glob("*/*_color.mp4"))
                if int(re.sub(r"\D", "", v.stem.split("_")[1])) % n == k
                and not (out_dir / f"{v.stem[:-6]}.npz").exists() and not (out_dir / f"{v.stem[:-6]}.failed").exists()]
        if not todo:
            if done:
                return
            time.sleep(10)
            continue
        for v in todo:
            sample = v.stem[:-len("_color")]
            out = out_dir / f"{sample}.npz"
            try:
                data, meta = track(v)
            except Exception as e:  # noqa: BLE001
                (out_dir / f"{sample}.failed").write_text(repr(e))
                print(f"{sample} FAILED {e!r}", flush=True)
                continue
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, meta=json.dumps(meta), **data)
            os.replace(tmp, out)
            print(f"{sample} {meta['frames']} frames", flush=True)


def main():
    if "--follow" in sys.argv:
        sys.argv.remove("--follow")
        k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
        return follow(k, n)
    k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    # every fetched clip on disk (the fetch may still be running: a rerun picks up the rest)
    # in rounds: every sign's first clip, then every sign's second, ... so a short run still covers all signs
    rank, rows = {}, []
    for v in sorted((AUTSL / "videos").glob("*/*_color.mp4")):
        rank[v.parent.name] = rank.get(v.parent.name, -1) + 1
        rows.append((rank[v.parent.name], v.parent.name, {"sample": v.stem[:-len("_color")], "video": v.relative_to(AUTSL)}))
    rows = [r for *_, r in sorted(rows, key=lambda x: x[:2])]
    out_dir = AUTSL / "tracks"
    out_dir.mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        if i % n != k:
            continue
        out = out_dir / f"{r['sample']}.npz"
        if out.exists():
            continue
        data, meta = track(AUTSL / r["video"])
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **data)
        os.replace(tmp, out)
        print(f"{i} {r['sample']} {meta['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
