"""ArabSign colour videos (Luqman, FG 2023; 6 signers x 50 sentences x ~30 repetitions) -> Isharati poses [T,50,3].

The per-signer archives (01-002.7z ... 06-004.7z, ~3.3 GB each) are done one at a time: unpacked to a temporary
folder, converted with MediaPipe Holistic (hands included, trimmed to the hand-visible span:
isharati.lexicon.asl_citizen.extract_trimmed), and the temporary copy deleted, so the 20 GB of video is never
unpacked at once.
The poses serve an Arabic sentence recognizer (back-translation) and multi-signer phrase signs («الله اكبر»).

  .venv/Scripts/python scripts/lexicon/arsl/arabsign_extract.py <archive.7z> [workers]
Output: data/ar_arabsign/poses/<signer>/<split>/<sentence>/<clip>.npy, and index.csv rows appended
"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
SRC = Path(r"D:\islam\ArabSign")
OUT = DATA / "ar_arabsign"


def convert(job):
    src, dst = job
    from isharati.lexicon.asl_citizen import extract_trimmed
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        pose = extract_trimmed(src)
    except Exception as e:  # a broken video: record it and go on
        pose = None
        print(src.name, "failed:", e, flush=True)
    if pose is None:
        dst.with_suffix(".failed").write_text("no usable signing", encoding="utf-8")
        return None
    np.save(dst, pose.astype(np.float16))
    return len(pose)


def main():
    """The archives are solid 7z: reaching a file means decoding everything before it, so a signer's archive is
    unpacked once (~3.5 GB temporary), converted by several workers, and the temporary copy deleted."""
    import py7zr
    from concurrent.futures import ProcessPoolExecutor
    archive = SRC / sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with py7zr.SevenZipFile(archive) as a:
        videos = [n for n in a.getnames() if n.endswith(".mp4")]
    todo = []
    for n in videos:  # 01/test/0001/01_0001_(10_03_21_20_37_10)_c.mp4
        signer, split, sentence = n.split("/")[:3]
        dst = OUT / "poses" / signer / split / sentence / (Path(n).stem + ".npy")
        if not dst.exists() and not dst.with_suffix(".failed").exists():
            todo.append((n, dst))
    print(archive.name, len(videos), "videos,", len(todo), "to convert", flush=True)
    if not todo:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="arabsign_", dir=str(OUT)))  # on D:, next to the output
    try:
        with py7zr.SevenZipFile(archive) as a:
            a.extractall(path=tmp)
        print(archive.name, "unpacked", flush=True)
        index = OUT / "index.csv"
        new_file = not index.exists()
        with open(index, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["signer", "split", "sentence", "clip", "frames", "path"])
            jobs = [(tmp / n, dst) for n, dst in todo]
            if workers <= 1:  # one process: nothing to break when memory is short (a pool dies with any worker)
                results = map(convert, jobs)
            else:
                pool = ProcessPoolExecutor(workers)
                results = pool.map(convert, jobs, chunksize=4)
            for k, ((n, dst), frames) in enumerate(zip(todo, results), 1):
                if frames:
                    signer, split, sentence = n.split("/")[:3]
                    w.writerow([signer, split, sentence, Path(n).stem, frames, str(dst.relative_to(OUT))])
                if k % 100 == 0:
                    f.flush()
                    print(f"{archive.name}: {k}/{len(todo)}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(archive.name, "done", flush=True)


if __name__ == "__main__":
    main()
