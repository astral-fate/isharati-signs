"""How faithfully each avatar bends its fingers: per finger, the angle at the middle joint (knuckle->middle vs
middle->end segment) on the avatar against the same angle in the keypoints that drove it, median over the frames a
hand is up. 0 deg = straight finger; a fist is ~90-100 deg at that joint.

  PYTHONPATH=src py scripts/avatars/finger_eval.py [--base URL] [model.vrm ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.types import FPS  # noqa: E402

FINGERS = {"Thumb": (2, 3, 4, ("Metacarpal", "Proximal", "Distal")), "Index": (5, 6, 7, ("Proximal", "Intermediate", "Distal")),
           "Middle": (9, 10, 11, ("Proximal", "Intermediate", "Distal")), "Ring": (13, 14, 15, ("Proximal", "Intermediate", "Distal")),
           "Little": (17, 18, 19, ("Proximal", "Intermediate", "Distal"))}
SIGNS = [("ar", "الله"), ("en", "RAMADAN"), ("en", "MOTHER"), ("ar", "الصلاة"), ("en", "HAJJ")]
PROBES = ROOT / "web" / "dist" / "clips" / "poses"


def bend(a, b, c):
    u, w = b - a, c - b
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))))


def main():
    from playwright.sync_api import sync_playwright
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
    models = [a for a in sys.argv[1:] if a.endswith(".vrm")] or ["avatar.vrm", "rocketbox_female_10.vrm"]
    lex = {"en": load_lexicon("en"), "ar": load_lexicon("ar")}
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page()
        for m in models:
            errs = {f: [] for f in FINGERS}
            sample = []
            for lang, g in SIGNS:
                e = lex[lang].lookup(g)
                clip = lex[lang].keypoints(e)
                probe = PROBES / f"_finger_probe_{e.sign_id}.json"
                probe.write_text(json.dumps({"fps": FPS, "frames": np.round(clip, 4).tolist()}))
                pg.goto(f"{base}/record?model={m}&pose=/clips/poses/{probe.name}")
                pg.wait_for_function("window.__ready || window.__error", timeout=180000)
                for i in range(len(clip)):
                    o = pg.evaluate(f"window.__probe({i})")
                    for side, base_i in (("left", 8), ("right", 29)):
                        h = clip[i, base_i:base_i + 21]
                        if h[:, 1].max() > 1.5 or np.ptp(h, axis=0).max() < 1e-3:  # hand down or not tracked
                            continue
                        for f, (j1, j2, j3, segs) in FINGERS.items():
                            pts = [o.get(f"{side}{f}{s}") for s in segs]
                            if any(q is None for q in pts):
                                continue
                            kp = bend(h[j1], h[j2], h[j3])
                            av = bend(*[np.array(q) for q in pts])
                            errs[f].append(abs(kp - av))
                            if g == "الله" and side == "right" and i == len(clip) // 2:
                                sample.append((f, round(kp), round(av)))
                probe.unlink()
            print(m, {f: round(float(np.median(v)), 1) if v else None for f, v in errs.items()},
                  "| الله mid-frame right hand (keypoints deg, avatar deg):", sample, flush=True)
        b.close()


if __name__ == "__main__":
    main()
