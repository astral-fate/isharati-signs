"""The avatar's fingers against anatomy and against the keypoints, on a sample of signs from several sources.

Per hand-up frame and finger (index..little), at the middle joint (PIP: Proximal -> Intermediate -> Distal bones):
  implausible  share of frames with the joint bent > 30 deg sideways or > 25 deg backwards (a hinge does neither)
  fidelity     median |fold(avatar) - fold(keypoints)| over the frames where the keypoints themselves are plausible:
               the anatomical limits should remove noise, not flatten real handshapes
  PYTHONPATH=src py scripts/avatars/finger_avatar_eval.py TAG [--base URL] [--model avatar.vrm]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.types import FPS  # noqa: E402

SIGNS = [("ur", "PILLAR"), ("ur", "WHAT"), ("ur", "PRAYER"), ("en", "QURAN"), ("en", "MOTHER"), ("ar", "الله"),
         ("ar", "صوم"), ("ar", "الصلاة"), ("en", "WHAT"), ("ar", "رمضان")]
FINGERS = {"Index": (5, 6, 7), "Middle": (9, 10, 11), "Ring": (13, 14, 15), "Little": (17, 18, 19)}
PROBES = ROOT / "web" / "dist" / "clips" / "poses"


def pip(m, p, d, n, fwd):
    """(fold toward the palm, sideways, backwards) in degrees at joint p; n = palm normal (out of the palm), fwd = along
    the fingers. The hinge axis is the finger's side-to-side axis (its direction in the palm plane x n), as in
    avatar.js: a palm-normal-based fold direction breaks down when the knuckle is bent 90° and would call every fist
    "sideways"."""
    u, w = p - m, d - p
    u /= np.linalg.norm(u) + 1e-9
    w /= np.linalg.norm(w) + 1e-9
    in_palm = u - (u @ n) * n
    if in_palm @ fwd < 0:  # past a 90° knuckle: keep it pointing along the fingers (else the fold reads backwards)
        in_palm = -in_palm
    axis = np.cross(in_palm, n) if np.linalg.norm(in_palm) > 1e-2 else np.cross(fwd, n)
    axis /= np.linalg.norm(axis) + 1e-9
    to_fold = np.cross(axis, u)
    s = np.degrees(np.arcsin(np.clip(abs(w @ axis), 0, 1)))
    fold = np.degrees(np.arctan2(w @ to_fold, w @ u))
    return fold, s, max(-fold, 0)


def normal(wrist, mid, idx, lit, side):
    a, b = mid - wrist, idx - lit
    n = np.cross(a, b) if side == "left" else np.cross(b, a)
    return n / (np.linalg.norm(n) + 1e-9)


def main():
    from playwright.sync_api import sync_playwright
    tag = sys.argv[1]
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "avatar.vrm"
    lex = {}
    bad_kp = bad_av = frames = 0
    fid = []
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = None
        for lang, g in SIGNS:
            if pg is not None:  # a fresh page per sign: hundreds of probed frames stall the software renderer
                pg.close()
            pg = br.new_page()
            L = lex.setdefault(lang, load_lexicon(lang))
            e = L.lookup(g)
            if e is None:
                continue
            clip = L.keypoints(e)
            probe = PROBES / f"_fingeravatar_{e.sign_id}.json"
            probe.write_text(json.dumps({"fps": FPS, "frames": np.round(clip, 4).tolist()}))
            pg.goto(f"{base}/record?model={model}&pose=/clips/poses/{probe.name}")
            pg.wait_for_function("window.__ready || window.__error", timeout=180000)
            for i in range(len(clip)):
                o = pg.evaluate(f"window.__probe({i})")
                for side, b0 in (("left", 8), ("right", 29)):
                    h = np.array([[q[0], -q[1], -q[2]] for q in clip[i, b0:b0 + 21]])
                    if np.ptp(h, axis=0).max() < 0.05 or -h[:, 1].min() > 1.5:
                        continue
                    nk = normal(h[0], h[9], h[5], h[17], side)
                    fk_ = (h[9] - h[0]) / (np.linalg.norm(h[9] - h[0]) + 1e-9)
                    g3 = lambda k: np.array(o[k])  # noqa: E731
                    na = normal(g3(f"{side}Hand"), g3(f"{side}MiddleProximal"), g3(f"{side}IndexProximal"),
                                g3(f"{side}LittleProximal"), side)
                    fa_ = g3(f"{side}MiddleProximal") - g3(f"{side}Hand")
                    fa_ = fa_ / (np.linalg.norm(fa_) + 1e-9)
                    frames += 1
                    kb = ab = False
                    for f, (m, q, d) in FINGERS.items():
                        fk, sk, bk = pip(h[m], h[q], h[d], nk, fk_)
                        fa, sa, ba = pip(g3(f"{side}{f}Proximal"), g3(f"{side}{f}Intermediate"), g3(f"{side}{f}Distal"), na, fa_)
                        kb |= sk > 30 or bk > 25
                        ab |= sa > 30 or ba > 25
                        if sk <= 30 and bk <= 25:
                            fid.append(abs(fa - fk))
                    bad_kp += kb
                    bad_av += ab
            probe.unlink()
        br.close()
    out = {"tag": tag, "model": model, "hand_frames": frames, "implausible_keypoints": round(bad_kp / frames, 3),
           "implausible_avatar": round(bad_av / frames, 3), "fidelity_deg": round(float(np.median(fid)), 1)}
    print(json.dumps(out))
    (ROOT / "results" / f"finger_avatar_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
