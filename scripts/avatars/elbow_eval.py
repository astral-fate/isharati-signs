"""Does each avatar bend its elbows as the signer did? One sign is played on every avatar (headless Chromium, the
/record page, as retarget_eval.py) and per frame the elbow angle (shoulder->elbow vs elbow->wrist, 180 = straight)
of the signer's keypoints is compared with the avatar's bones (UpperArm, LowerArm, Hand). A screenshot of the same
frame is saved for each avatar.

  PYTHONPATH=src py -m isharati.app.server                         # port 8765
  py scripts/avatars/elbow_eval.py ar وضوء [--frames 0.3,0.5,0.7]

Output: results/elbow_eval_<gloss>.json and results/elbow_<gloss>_<model>_<frame>.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.types import FPS  # noqa: E402

MODELS = ["avatar.vrm", "rocketbox_female_06.vrm", "rocketbox_female_10.vrm", "rocketbox_male_15.vrm",
          "rocketbox_male_19.vrm", "rocketbox_male_21.vrm"]
ARMS = {"left": (5, 6, 7), "right": (2, 3, 4)}  # Isharati joints: shoulder, elbow, wrist
OUT = ROOT / "results"
PROBES = ROOT / "web" / "dist" / "clips" / "poses"


def angle(a, b, c):
    u, w = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))))


def main():
    from playwright.sync_api import sync_playwright
    lang, gloss = sys.argv[1], sys.argv[2]
    shots = [float(x) for x in (sys.argv[sys.argv.index("--frames") + 1] if "--frames" in sys.argv
                                else "0.3,0.5,0.7").split(",")]
    lex = load_lexicon(lang)
    e = lex.lookup(gloss)
    clip = lex.keypoints(e)
    probe = PROBES / f"_elbow_probe_{e.sign_id}.json"
    probe.write_text(json.dumps({"fps": FPS, "frames": np.round(clip, 4).tolist()}), encoding="utf-8")
    report = {"gloss": gloss, "sign_id": e.sign_id, "dataset": e.dataset, "frames": len(clip), "models": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 480, "height": 560})
        for model in MODELS:
            page.goto(f"http://127.0.0.1:8765/record?model={model}&pose=/clips/poses/{probe.name}")
            page.wait_for_function("window.__ready || window.__error", timeout=180000)
            if page.evaluate("window.__error"):
                report["models"][model] = {"error": page.evaluate("window.__error")}
                continue
            per = {s: [] for s in ARMS}
            for i in range(len(clip)):
                b = page.evaluate(f"window.__probe({i})")
                for side, (sh, el, wr) in ARMS.items():
                    sig = angle(clip[i][sh], clip[i][el], clip[i][wr])
                    av = angle(b[f"{side}UpperArm"], b[f"{side}LowerArm"], b[f"{side}Hand"])
                    per[side].append((sig, av))
            res = {}
            for side, rows in per.items():
                sig, av = np.array(rows).T
                bent = sig < 150  # frames where the signer's elbow is clearly bent
                res[side] = {"signer_median": round(float(np.median(sig)), 1),
                             "avatar_median": round(float(np.median(av)), 1),
                             "err_median": round(float(np.median(np.abs(sig - av))), 1),
                             "bent_frames": int(bent.sum()),
                             "err_when_bent": round(float(np.median(np.abs(sig - av)[bent])), 1) if bent.any() else None,
                             "avatar_straighter_by": round(float(np.median((av - sig)[bent])), 1) if bent.any() else None}
            report["models"][model] = res
            print(model, json.dumps(res), flush=True)
            for fr in shots:
                i = int(fr * (len(clip) - 1))
                page.evaluate(f"window.__probe({i})")
                page.screenshot(path=str(OUT / f"elbow_{e.sign_id}_{model.replace('.vrm', '')}_{i}.png"))
        browser.close()
    probe.unlink(missing_ok=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"elbow_eval_{e.sign_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
