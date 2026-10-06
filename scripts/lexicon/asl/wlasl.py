"""ASL signs from WLASL (Li et al., WACV 2020; C-UDA licence, research use) for the words our ASL lexicon lacks.

WLASL_v0.3.json lists 2,000 glosses with ~21,000 clips hosted on sign dictionary sites and YouTube. Only glosses that
data/asl/lexicon/lexicon_all.jsonl does not have are fetched (~680), up to two clips each, preferring the dictionary
sites (direct downloads, no YouTube throttling). A clip's frame range, when given, is cut first; the hand-visible span
is then the sign (isharati.lexicon.asl_citizen.extract_trimmed).

  .venv/Scripts/python scripts/lexicon/asl/wlasl.py [workers]
Output: data/asl/lexicon/wlasl/{signs/*.npy, entries.jsonl}; videos in D:/islam/datasets_extra/WLASL/videos
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
WLASL = Path(r"D:\islam\datasets_extra\WLASL\start_kit\WLASL_v0.3.json")
VIDEOS = Path(r"D:\islam\datasets_extra\WLASL\videos")
OUT = DATA / "asl" / "lexicon" / "wlasl"
PER_GLOSS = 2
UA = {"User-Agent": "Mozilla/5.0 (research; isharati sign lexicon)"}


def fetch(inst):
    """Download one clip (direct file, or yt-dlp for pages), cut to its frame range. Returns the path or None."""
    vid = inst["video_id"]
    path = VIDEOS / f"{vid}.mp4"
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = inst["url"]
    raw = VIDEOS / f"{vid}_raw.mp4"
    try:
        if "youtube" in url or "youtu.be" in url or not url.lower().split("?")[0].endswith((".mp4", ".mov", ".m4v")):
            subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "-q", "-f", "mp4/best", "-o", str(raw), url],
                           cwd=r"D:\islam\yt-dlp", timeout=120, capture_output=True)
        else:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            raw.write_bytes(r.content)
    except Exception:
        return None
    if not raw.exists() or raw.stat().st_size < 1000:
        return None
    start, end = int(inst.get("frame_start", 1)), int(inst.get("frame_end", -1))
    if end > 0:  # WLASL frame ranges are 1-based and inclusive
        f = f"select=between(n\\,{start - 1}\\,{end - 1}),setpts=N/FRAME_RATE/TB"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", f, "-an", str(path)], timeout=120)
        raw.unlink(missing_ok=True)
    else:
        raw.rename(path)
    return path if path.exists() else None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    from isharati.lexicon.asl_citizen import extract_trimmed
    VIDEOS.mkdir(parents=True, exist_ok=True)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    ours = {json.loads(l)["gloss"].rstrip("0123456789").lower()
            for l in (DATA / "asl" / "lexicon" / "lexicon_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = []
    for e in json.load(open(WLASL, encoding="utf-8")):
        if e["gloss"].lower() in ours:
            continue
        insts = sorted(e["instances"], key=lambda i: ("youtu" in i["url"], i["instance_id"]))
        todo.append((e["gloss"], insts[:6]))  # try up to six, keep the first PER_GLOSS that work
    print(len(todo), "new glosses", flush=True)

    def work(item):
        gloss, insts = item
        kept = []
        for inst in insts:
            if len(kept) >= PER_GLOSS:
                break
            npy = OUT / "signs" / f"{inst['video_id']}.npy"
            if npy.exists():
                kept.append(inst["video_id"])
                continue
            path = fetch(inst)
            if path is None:
                continue
            try:
                pose = extract_trimmed(path)
            except Exception:
                pose = None
            if pose is not None and len(pose) >= 6:
                np.save(npy, pose.astype(np.float32))
                kept.append(inst["video_id"])
        return gloss, kept

    entries, done = [], 0
    with ThreadPoolExecutor(workers) as pool:
        for gloss, kept in pool.map(work, todo):
            done += 1
            for k, vid in enumerate(kept):  # the first clip is the entry; a second one is a template for evaluation
                entries.append({"gloss": gloss.upper() + ("" if k == 0 else str(k + 1)), "sign_id": f"wlasl_{vid}",
                                "dataset": "wlasl", "keypoints_path": f"wlasl/signs/{vid}.npy", "review_status": "pending",
                                "is_religious": False, "is_letter": False})
            if done % 50 == 0:
                print(f"{done}/{len(todo)} glosses, {len(entries)} clips", flush=True)
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    print(f"{len({e['gloss'].rstrip('0123456789') for e in entries})} new ASL glosses, {len(entries)} clips -> {OUT}", flush=True)


def entries_only():
    """Write entries.jsonl from the clips extracted so far (the fetch writes it only when it finishes)."""
    by_vid = {i["video_id"]: e["gloss"] for e in json.load(open(WLASL, encoding="utf-8")) for i in e["instances"]}
    seen, entries = {}, []
    for npy in sorted((OUT / "signs").glob("*.npy")):
        gloss = by_vid.get(npy.stem)
        if gloss:
            k = seen.get(gloss, 0)
            seen[gloss] = k + 1
            entries.append({"gloss": gloss.upper() + ("" if k == 0 else str(k + 1)), "sign_id": f"wlasl_{npy.stem}",
                            "dataset": "wlasl", "keypoints_path": f"wlasl/signs/{npy.name}", "review_status": "pending",
                            "is_religious": False, "is_letter": False})
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    print(f"{len(seen)} WLASL glosses, {len(entries)} clips -> entries.jsonl")


if __name__ == "__main__":
    entries_only() if sys.argv[1:2] == ["entries"] else main()
