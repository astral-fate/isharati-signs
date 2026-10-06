"""How anatomically plausible the finger keypoints are, per sign source. A finger's middle (PIP) and end (DIP) joints
are hinges: they fold toward the palm, not sideways, and barely backwards. For each finger segment after the first, the
bend from the previous segment is split into
  flex      folding toward the palm (palm normal side), in degrees
  side      bending out of the finger's own plane (should be ~0)
  back      folding toward the back of the hand (hyperextension; a few degrees at most)
Reported per source: share of hand-up frames with any PIP/DIP joint more than 30 deg sideways or 25 deg backwards, and
the median frame-to-frame change of the joint angles (jitter).

  PYTHONPATH=src py scripts/avatars/finger_plausibility.py [n signs per source]
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402

CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # index..little: MCP, PIP, DIP, tip


def joint_angles(h, side):
    """(flex, side, back) degrees for the PIP and DIP of the four fingers: shape (4, 2, 3)."""
    a, b = h[9] - h[0], h[5] - h[17]
    n = np.cross(a, b) if side == "left" else np.cross(b, a)  # palm normal (out of the palm), as avatar.js
    n /= np.linalg.norm(n) + 1e-9
    out = np.zeros((4, 2, 3))
    for f, (m, p, d, t) in enumerate(CHAINS):
        for k, (j0, j1, j2) in enumerate(((m, p, d), (p, d, t))):
            u, w = h[j1] - h[j0], h[j2] - h[j1]
            u /= np.linalg.norm(u) + 1e-9
            w /= np.linalg.norm(w) + 1e-9
            across = np.cross(u, n)                         # normal of the finger's flexion plane
            across /= np.linalg.norm(across) + 1e-9
            sidew = np.degrees(np.arcsin(np.clip(w @ across, -1, 1)))
            inplane = w - (w @ across) * across
            ang = np.degrees(np.arctan2(inplane @ n, inplane @ u))  # >0: toward the palm side (flexion; n points out of
            # the palm: in a fist the fingertips sit on its +n side)
            out[f, k] = (max(ang, 0), abs(sidew), max(-ang, 0))
    return out


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    stats = defaultdict(lambda: {"frames": 0, "bad": 0, "jitter": []})
    for lang in ("en", "ar", "ur", "tr"):
        try:
            lex = load_lexicon(lang)
        except Exception as e:
            print(lang, "skipped:", e)
            continue
        by_ds = defaultdict(list)
        for e in lex.sign_entries():
            by_ds[e.dataset].append(e)
        for ds, es in by_ds.items():
            for e in random.Random(0).sample(es, min(n_per, len(es))):
                try:
                    clip = lex.keypoints(e)
                except Exception:
                    continue
                for side, base in (("left", 8), ("right", 29)):
                    prev = None
                    for f in clip:
                        h = f[base:base + 21]
                        if np.ptp(h, axis=0).max() < 0.05 or h[:, 1].min() > 1.5:  # untracked or resting hand
                            prev = None
                            continue
                        j = joint_angles(h, side)
                        s = stats[f"{lang}:{ds}"]
                        s["frames"] += 1
                        s["bad"] += bool((j[..., 1] > 30).any() or (j[..., 2] > 25).any())
                        if prev is not None:
                            s["jitter"].append(float(np.median(np.abs(j[..., 0] - prev[..., 0]))))
                        prev = j
    print(f"{'source':38s} {'frames':>7s} {'implausible':>11s} {'jitter deg/frame':>17s}")
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["bad"] / max(1, kv[1]["frames"])):
        print(f"{k:38s} {s['frames']:7d} {s['bad'] / max(1, s['frames']):10.0%} {np.median(s['jitter']) if s['jitter'] else 0:17.1f}")


if __name__ == "__main__":
    main()
