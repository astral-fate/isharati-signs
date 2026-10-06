"""ASL signs from MS-ASL (Joze & Koller, BMVC 2019; C-UDA, research use) for words neither our lexicon nor WLASL has.

MS-ASL clips are segments of YouTube videos (url, start_time, end_time). Only the repository's test list is local, so
one clip per new word is taken from it (~140 words: thanks, twenty, how are you, next week ...). YouTube throttles
bursts, so this waits for the slow dictionary downloader to finish, then fetches one segment at a time with pauses.

  .venv/Scripts/python scripts/lexicon/asl/msasl.py
Output: data/asl/lexicon/msasl/{signs/*.npy, entries.jsonl}
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
REPO = Path(r"D:\islam\datasets_extra\msasl-video-downloader")
VIDEOS = Path(r"D:\islam\datasets_extra\msasl_videos")
OUT = DATA / "asl" / "lexicon" / "msasl"
WAIT_FOR = DATA / "lexicon_v2" / "arabic_dictionaries" / "logs" / "slow_tr.log"
COOKIES = Path(r"D:\islam\cookies\cookies_2.txt")


def main():
    from isharati.lexicon.asl_citizen import extract_trimmed
    while not (WAIT_FOR.exists() and "all downloads finished" in WAIT_FOR.read_text(encoding="utf-8", errors="replace")):
        time.sleep(600)  # one YouTube downloader at a time
    ours = {json.loads(l)["gloss"].rstrip("0123456789").lower()
            for l in (DATA / "asl" / "lexicon" / "lexicon_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    wlasl = {e["gloss"].lower() for e in json.load(open(r"D:\islam\datasets_extra\WLASL\start_kit\WLASL_v0.3.json", encoding="utf-8"))}
    clips = {}
    for c in json.load(open(REPO / "MSASL_test.json", encoding="utf-8")):
        word = c["clean_text"].lower()
        if word not in ours and word not in wlasl:
            clips.setdefault(word, []).append(c)
    print(len(clips), "new words", flush=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries = []
    for word, cs in clips.items():
        for c in cs[:3]:  # try up to three clips, keep the first that works
            vid = c["url"].split("v=")[-1][:11]
            name = f"{vid}_{int(c['start_time'] * 1000)}"
            npy = OUT / "signs" / f"{name}.npy"
            if not npy.exists():
                mp4 = VIDEOS / f"{name}.mp4"
                if not mp4.exists():
                    subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "-q", "--cookies", str(COOKIES),
                                    "--js-runtimes", "node", "--remote-components", "ejs:github",
                                    "-f", "bv*[height<=480][ext=mp4]/b[height<=480]",
                                    "--download-sections", f"*{c['start_time']:.2f}-{c['end_time'] + 0.2:.2f}",
                                    "-o", str(mp4), "https://" + c["url"]], cwd=r"D:\islam\yt-dlp", timeout=180, capture_output=True)
                    time.sleep(6)  # a pause between YouTube requests
                if not mp4.exists():
                    continue
                try:
                    pose = extract_trimmed(mp4)
                except Exception:
                    pose = None
                if pose is None or len(pose) < 6:
                    continue
                np.save(npy, pose.astype(np.float32))
            entries.append({"gloss": word.upper().replace("_", " "), "sign_id": f"msasl_{name}", "dataset": "msasl",
                            "keypoints_path": f"msasl/signs/{name}.npy", "review_status": "pending",
                            "is_religious": False, "is_letter": False})
            break
        (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    print(len(entries), "MS-ASL signs ->", OUT, flush=True)


if __name__ == "__main__":
    main()
