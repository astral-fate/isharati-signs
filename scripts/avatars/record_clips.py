"""Record each planned clip on its avatar (headless Chromium, the built /record page, one JPEG per frame) and encode
WebM (VP9) + MP4 (H.264) + a poster with ffmpeg. Needs web/dist built and the app running on --base.

  cd web && npm run build && cd ..
  PYTHONPATH=src py -m isharati.app.server            # another shell (port 8765)
  py scripts/avatars/record_clips.py [--base http://127.0.0.1:8765]
"""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "web" / "public" / "clips"
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
MAX_BYTES = 1_500_000


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def main():
    plan = json.loads((CLIPS / "poses" / "plan.json").read_text(encoding="utf-8"))
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for c in plan:
            page = browser.new_page(viewport={"width": 480, "height": 600})
            page.goto(f"{BASE}/record?model={c['avatar']}&pose=/clips/poses/{c['id']}.json")
            page.wait_for_function("window.__ready || window.__error", timeout=180_000)
            if page.evaluate("window.__error"):
                raise SystemExit(f"{c['id']}: {page.evaluate('window.__error')}")
            n = page.evaluate("window.__count")
            fps = str(json.loads((CLIPS / "poses" / f"{c['id']}.json").read_text(encoding="utf-8"))["fps"])
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(n):
                    data = page.evaluate(f"window.__frame({i})").split(",", 1)[1]
                    Path(tmp, f"{i:04d}.jpg").write_bytes(base64.b64decode(data))
                frames = str(Path(tmp, "%04d.jpg"))
                webm, mp4, poster = CLIPS / f"{c['id']}.webm", CLIPS / f"{c['id']}.mp4", CLIPS / f"{c['id']}.jpg"
                ffmpeg("-framerate", fps, "-i", frames, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38",
                       "-pix_fmt", "yuv420p", "-an", str(webm))
                ffmpeg("-framerate", fps, "-i", frames, "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4))
                shutil.move(str(Path(tmp, f"{n // 2:04d}.jpg")), str(poster))
            page.close()
            for f in (webm, mp4):
                if f.stat().st_size > MAX_BYTES:
                    raise SystemExit(f"{f.name} is {f.stat().st_size} bytes, over {MAX_BYTES}: raise -crf and re-run")
            out.append({**{k: c[k] for k in ("avatar", "lang", "sign_language", "glosses", "seconds")},
                        "webm": f"/clips/{webm.name}", "mp4": f"/clips/{mp4.name}", "poster": f"/clips/{poster.name}"})
            print(c["id"], n, "frames", webm.stat().st_size, mp4.stat().st_size)
        browser.close()
    (CLIPS / "clips.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in CLIPS.rglob("*") if f.is_file())
    print("clips folder:", total, "bytes")
    if total > 15_000_000:
        raise SystemExit("clips folder over 15 MB")


if __name__ == "__main__":
    main()
