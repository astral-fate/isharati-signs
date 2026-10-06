"""Single-sign YouTube videos (one word per video, the word in the title) -> ASL lexicon entries + manifest.

Sources such as the Obscure ASL channel title every video "<Word> - ASL" (synonyms as "Toxic, Toxin - ASL").
Each video is downloaded, trimmed to the stretch where a hand is visible, converted to the Isharati pose contract
(isharati.lexicon.asl_citizen.extract_trimmed), and written as a lexicon entry. The gloss comes from the title; any
other names in it become aliases. Resumable: videos already converted are skipped.

  .venv/Scripts/python scripts/lexicon/asl/youtube_signs.py obscure_asl https://www.youtube.com/@obscureasl/videos https://www.youtube.com/@obscureasl/shorts
  .venv/Scripts/python scripts/lexicon/asl/youtube_signs.py single_clips https://youtu.be/hVu1uneKWEI https://youtu.be/qlqjQ5PvdLI

Output: data/asl/lexicon/<source>/{signs/*.npy, entries.jsonl}, data/asl/youtube/<source>/manifest.csv
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon.asl_citizen import extract_trimmed  # noqa: E402

YTDLP = Path(r"D:\islam\yt-dlp")
COOKIES = Path(r"D:\islam\cookies\cookies_2.txt")
RELIGIOUS = {"muslim", "islam", "quran", "koran", "allah", "god", "prophet", "mosque", "masjid", "prayer", "pray",
             "ramadan", "hajj", "zakat", "angel", "heaven", "hell", "faith", "religion"}


def glosses_from_title(title: str) -> list[str]:
    """'Toxic, Toxin - ASL' -> ['TOXIC', 'TOXIN']; 'muslim in ASL' -> ['MUSLIM']; '(anatomy)' notes are dropped."""
    t = re.sub(r"#\w+", " ", title)
    t = re.sub(r"(?i)\s*(-\s*)?(in\s+)?asl(\s+short)?\b.*$", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    names = [n.strip(" -–") for n in re.split(r"[,/]| or ", t) if n.strip(" -–")]
    return [n.upper() for n in names]


def list_videos(url: str) -> list[dict]:
    out = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--no-warnings", "--cookies", str(COOKIES),
                          "--print", "%(id)s\t%(title)s", url], cwd=YTDLP, capture_output=True, text=True,
                         encoding="utf-8")
    rows = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            vid, title = line.split("\t", 1)
            rows.append({"id": vid, "title": title, "url": f"https://www.youtube.com/watch?v={vid}"})
    if not rows and "watch" in url or "youtu.be" in url:
        info = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--skip-download", "--cookies",
                               str(COOKIES), "--js-runtimes", "node", "--remote-components", "ejs:github",
                               "--print", "%(id)s\t%(title)s", url], cwd=YTDLP, capture_output=True, text=True,
                              encoding="utf-8").stdout.strip().splitlines()
        rows = [{"id": l.split("\t")[0], "title": l.split("\t", 1)[1],
                 "url": f"https://www.youtube.com/watch?v={l.split(chr(9))[0]}"} for l in info if "\t" in l]
    return rows


def download(video: dict, folder: Path) -> Path | None:
    dst = folder / f"{video['id']}.mp4"
    if not dst.exists():
        subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--cookies", str(COOKIES), "--js-runtimes",
                        "node", "--remote-components", "ejs:github", "-f", "bv*[height<=720][ext=mp4]+ba/b[height<=720]/b",
                        "--merge-output-format", "mp4", "--sleep-requests", "1", "--sleep-interval", "3",
                        "--max-sleep-interval", "8", "-o", str(folder / "%(id)s.%(ext)s"), video["url"]],
                       cwd=YTDLP, capture_output=True)
    return dst if dst.exists() else None


def main():
    source, urls = sys.argv[1], sys.argv[2:]
    vid_dir = DATA / "asl" / "youtube" / source
    lex_dir = DATA / "asl" / "lexicon" / source
    (lex_dir / "signs").mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    videos = {v["id"]: v for u in urls for v in list_videos(u)}
    print(f"{source}: {len(videos)} videos", flush=True)

    manifest, entries = [], []
    for n, v in enumerate(videos.values()):
        names = glosses_from_title(v["title"])
        sid = f"{source}_{v['id']}"
        npy = lex_dir / "signs" / f"{sid}.npy"
        status, frames = "ok", 0
        if not names:
            status = "no gloss in title"
        elif not npy.exists():
            path = download(v, vid_dir)
            if path is None:
                status = "download failed"
            else:
                clip = extract_trimmed(path)
                if clip is None:
                    status = "no usable hands"
                else:
                    np.save(npy, clip)
        if status == "ok":
            frames = int(np.load(npy, mmap_mode="r").shape[0])
            religious = any(g.lower() in RELIGIOUS for g in names)
            for g in names:
                entries.append({"gloss": g, "sign_id": sid, "dataset": source,
                                "keypoints_path": f"{source}/signs/{sid}.npy", "review_status": "pending",
                                "is_religious": religious, "is_letter": False})
        manifest.append({"source": source, "video_id": v["id"], "url": v["url"], "title": v["title"],
                         "gloss": names[0] if names else "", "aliases": "; ".join(names[1:]), "frames_25fps": frames,
                         "status": status, "licence": "YouTube upload; channel owner's permission needed to redistribute"})
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(videos)}", flush=True)
    (lex_dir / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    with open(vid_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"{source}: {ok}/{len(manifest)} signs -> {lex_dir}; manifest -> {vid_dir / 'manifest.csv'}")
    for m in manifest:
        if m["status"] != "ok":
            print(f"  {m['status']}: {m['title']}")


if __name__ == "__main__":
    main()
