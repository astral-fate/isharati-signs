"""End-to-end check on video: texts -> the running app's Studio API -> glosses, stitched pose and signer face -> one MP4
per text with three panels side by side: the skeleton with the signer's face contours | the stylised avatar | a
realistic (Rocketbox) avatar, and the text and glosses as a caption.

The avatars are recorded frame by frame on the app's own /record page (the same avatar.js the site runs), so what
the video shows is what a visitor sees. Output: results/e2e_videos/<n>_<lang>.mp4 and index.json (text, glosses,
missing signs, frames with a face).

  PYTHONPATH=src python -m isharati.app.server           # port 8765, or --base
  py scripts/avatars/e2e_videos.py [--base URL] [--out DIR]
"""
import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [("ar", "بسم الله الرحمن الرحيم"), ("ar", "الصلاة في رمضان"),
         ("en", "Bismillah. Muslims fast in Ramadan."), ("en", "Make wudu before salah.")]
AVATARS = ["avatar.vrm", "rocketbox_female_10.vrm"]
W, H = 480, 600
FFMPEG = "ffmpeg"
TORSO = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (2, 5)]
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
        (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return json.loads(r.read())


def skeleton(frames, faces, edges, i, box):
    """The skeleton panel for frame i (pose and face share the pose's coordinate frame)."""
    img = np.full((H, W, 3), (26, 8, 4), np.uint8)
    x0, x1, y0, y1 = box
    s = min(W * 0.8 / (x1 - x0), H * 0.8 / (y1 - y0))
    px = lambda p: (int(W / 2 + (p[0] - (x0 + x1) / 2) * s), int(H / 2 + (p[1] - (y0 + y1) / 2) * s))  # noqa: E731
    f = np.array(frames[i])
    for a, b in TORSO:
        cv2.line(img, px(f[a]), px(f[b]), (211, 240, 126), 3, cv2.LINE_AA)
    for base, wr in ((8, 7), (29, 4)):
        if np.ptp(f[base:base + 21], axis=0).max() > 0.02:
            cv2.line(img, px(f[wr]), px(f[base]), (211, 240, 126), 2, cv2.LINE_AA)
            for a, b in HAND:
                cv2.line(img, px(f[base + a]), px(f[base + b]), (255, 184, 143), 1, cv2.LINE_AA)
    face = faces[i] if faces else None
    if face and any(q and q[0] is not None for q in face):
        for part, pairs in edges.items():
            colour = (178, 159, 255) if part == "lips" else (255, 255, 255) if part.endswith("iris") else (211, 240, 126)
            for a, b in pairs:
                qa, qb = face[a], face[b]
                if qa and qb and qa[0] is not None and qb[0] is not None:
                    cv2.line(img, px(qa), px(qb), colour, 1, cv2.LINE_AA)
    else:
        cv2.circle(img, px(f[0]), int(0.35 * s), (211, 240, 126), 2, cv2.LINE_AA)
    return img


def caption(img, lines):
    """Text lines at the bottom (rendered with PIL so Arabic shapes and runs right to left)."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, pil.height - 34 * len(lines) - 10, pil.width, pil.height], fill=(4, 8, 26))
    for k, line in enumerate(lines):
        d.text((14, pil.height - 34 * (len(lines) - k) - 4), line, font=font, fill=(232, 255, 250),
               direction="rtl" if any("\u0600" <= ch <= "\u06ff" for ch in line) else None)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    from playwright.sync_api import sync_playwright
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "results" / "e2e_videos"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = br.new_page(viewport={"width": W, "height": H})
        for n, (lang, text) in enumerate(TEXTS, 1):
            r = post(base, "/api/sign", {"text": text, "lang": lang, "fresh": True})
            pose = get(base, f"/pose/{r['id']}.json")
            frames, faces, edges = pose["frames"], pose.get("face"), pose.get("face_edges", {})
            pts = np.array(frames)[:, :8, :2].reshape(-1, 2)
            box = (pts[:, 0].min() - 0.6, pts[:, 0].max() + 0.6, pts[:, 1].min() - 0.8, pts[:, 1].max() + 0.3)
            panels = {}
            for m in AVATARS:
                pg.goto(f"{base}/record?model={m}&pose=/pose/{r['id']}.json")
                pg.wait_for_function("window.__ready || window.__error", timeout=180000)
                if pg.evaluate("window.__error"):
                    raise RuntimeError(pg.evaluate("window.__error"))
                panels[m] = [cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(
                    pg.evaluate(f"window.__frame({i})").split(",")[1]), np.uint8), cv2.IMREAD_COLOR), (W, H))
                    for i in range(len(frames))]
            video = out / f"{n}_{lang}.mp4"
            enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                    f"{3 * W}x{H}", "-r", str(pose["fps"]), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                    "yuv420p", "-crf", "20", str(video)], stdin=subprocess.PIPE)
            lines = [text, " ".join(r["glosses"]) + (f"   (no sign: {', '.join(r['missing_signs'])})" if r["missing_signs"] else "")]
            for i in range(len(frames)):
                row = np.hstack([skeleton(frames, faces, edges, i, box)] + [panels[m][i] for m in AVATARS])
                enc.stdin.write(caption(row, lines).tobytes())
            enc.stdin.close()
            enc.wait()
            with_face = sum(1 for b in pose.get("blend", []) if b and b[0] is not None)
            index.append({"video": video.name, "lang": lang, "text": text, "glosses": r["glosses"],
                          "missing": r["missing_signs"], "frames": len(frames), "frames_with_face": with_face})
            print(video.name, r["glosses"], f"face in {with_face}/{len(frames)} frames", flush=True)
        br.close()
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
