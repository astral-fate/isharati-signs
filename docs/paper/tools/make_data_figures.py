"""Data-chapter figures (run from the project root with the project venv):
  keypoint_layout.pdf  the 50-joint Isharati skeleton on a real lexicon frame, joints numbered
  normalisation.pdf    one YouTube-ASL clip before (pixels) and after normalisation (neck origin, shoulder units)
  youtube_asl_stats.pdf frames per clip, hand visibility, frame sizes of the converted YouTube-ASL part
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from isharati.render_skeleton import BODY, HAND  # noqa: E402

FIG = ROOT / "docs" / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
TEAL, AMBER, RED, INK, GREY = "#14b8a6", "#f59e0b", "#dc2626", "#1d1d1b", "#9ca3af"
NAMES = ["0 nose", "1 neck", "2 R shoulder", "3 R elbow", "4 R wrist", "5 L shoulder", "6 L elbow", "7 L wrist"]


def draw(ax, f, label=True):
    for a, b in BODY:
        ax.plot([f[a, 0], f[b, 0]], [f[a, 1], f[b, 1]], color=INK, lw=1.4)
    for base, col in ((8, AMBER), (29, RED)):
        h = f[base:base + 21]
        if np.isfinite(h).all():
            for a, b in HAND:
                ax.plot([h[a, 0], h[b, 0]], [h[a, 1], h[b, 1]], color=col, lw=0.8)
            ax.scatter(h[:, 0], h[:, 1], s=3, color=col, zorder=3)
    ax.scatter(f[:8, 0], f[:8, 1], s=14, color=TEAL, zorder=4)
    if label:
        off = {0: (6, 0), 1: (-4, -14), 2: (-6, 8), 3: (-8, -10), 4: (6, -12), 5: (2, 12), 6: (6, -6), 7: (8, -12)}
        for j in range(8):
            ax.annotate(NAMES[j], f[j, :2], textcoords="offset points", xytext=off[j], fontsize=6.5,
                        ha="right" if off[j][0] < 0 else "left",
                        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.4})
        ax.annotate("8-28 left hand", f[20, :2], textcoords="offset points", xytext=(10, 14), fontsize=6.5,
                    color=AMBER, arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 0.4})
        ax.annotate("29-49 right hand", f[41, :2], textcoords="offset points", xytext=(-10, 22), fontsize=6.5,
                    color=RED, ha="right", arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.4})


def layout():
    p = np.load(ROOT / "data" / "asl" / "out" / json.loads((ROOT / "docs/paper/data/ask_en.json").read_text(
        encoding="utf-8"))["id"] / "pose.npy")
    d = json.loads((ROOT / "docs/paper/data/ask_en.json").read_text(encoding="utf-8"))
    # the frame with the hands furthest apart and both hands detected, so every joint is visible
    ok = [t for t in range(len(p)) if np.isfinite(p[t]).all() and np.abs(p[t, 8:50]).sum() > 0
          and p[t, [4, 7], 1].max() < 0.6]
    t = max(ok, key=lambda t: np.linalg.norm(p[t, 4, :2] - p[t, 7, :2]))
    f = p[t]
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    draw(ax, f)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(2.3, -1.4); ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.5, ls=":"); ax.axvline(0, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("x (shoulder widths)"); ax.set_ylabel("y (shoulder widths, down)")
    fig.savefig(FIG / "keypoint_layout.pdf"); plt.close(fig)


def normalisation():
    z = np.load(ROOT / "data" / "youtube_asl" / "poses" / "raw_keypoints_1.npz")
    clip = "BSRkKugmny0.006331-006508"
    norm = z[clip].astype(np.float32)
    raw = json.loads((ROOT / "docs" / "paper" / "data" / "yt_raw_frame.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9))
    r = np.full((50, 2), np.nan)
    ids = [0, None, 12, 14, 16, 11, 13, 15]
    for j, i in enumerate(ids):
        if i is not None:
            r[j] = raw["pose"][i]
    r[1] = (r[2] + r[5]) / 2
    if raw["right"]:
        r[29:50] = raw["right"]
    if raw["left"]:
        r[8:29] = raw["left"]
    draw(axes[0], r, label=False)
    lo, hi = np.nanmin(r[:8], 0), np.nanmax(r[:8], 0)
    pad = 0.6 * (hi - lo).max()
    axes[0].set_xlim(lo[0] - pad, hi[0] + pad); axes[0].set_ylim(hi[1] + pad, lo[1] - pad)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("y (px)")
    axes[0].set_title("MediaPipe output (pixels)", fontsize=8)
    draw(axes[1], norm[0], label=False)
    axes[1].set_xlim(-2.2, 2.2); axes[1].set_ylim(2.6, -1.2); axes[1].set_aspect("equal")
    axes[1].axhline(0, color=GREY, lw=0.5, ls=":"); axes[1].axvline(0, color=GREY, lw=0.5, ls=":")
    axes[1].set_title("Isharati format (neck origin, shoulder units)", fontsize=8)
    fig.savefig(FIG / "normalisation.pdf"); plt.close(fig)


def yt_stats():
    rows = list(csv.DictReader(open(ROOT / "data/youtube_asl/manifest/raw_keypoints_1.csv", encoding="utf-8")))
    fr = np.array([int(r["frames"]) for r in rows])
    hand = np.array([max(float(r["left_hand"]), float(r["right_hand"])) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"wspace": 0.45})
    axes[0].hist(np.clip(fr, 0, 600), bins=60, color=TEAL)
    axes[0].axvline(np.median(fr), color=RED, lw=1)
    axes[0].set_xlabel("frames per clip (~30 fps)"); axes[0].set_ylabel("clips")
    axes[0].text(np.median(fr) + 10, axes[0].get_ylim()[1] * 0.85, f"median {int(np.median(fr))}", fontsize=7, color=RED)
    axes[1].hist(hand, bins=40, color=TEAL)
    axes[1].set_xlabel("share of frames with a hand detected"); axes[1].set_ylabel("clips")
    # manifest width/height are stored as (height, width) by the converter: swap back
    sizes = Counter(f"{r['height']}x{r['width']}" for r in rows).most_common(5)
    axes[2].barh(range(len(sizes)), [n for _, n in sizes], color=TEAL)
    axes[2].set_yticks(range(len(sizes)), [s for s, _ in sizes]); axes[2].invert_yaxis()
    axes[2].set_xlabel("clips"); axes[2].set_title("frame size (w x h)", fontsize=8)
    fig.savefig(FIG / "youtube_asl_stats.pdf"); plt.close(fig)
    print("clips", len(rows), "videos", len({r["video_id"] for r in rows}), "median", np.median(fr),
          "hand>=0.5", round(float((hand >= 0.5).mean()), 3), sizes)


if __name__ == "__main__":
    layout(); normalisation(); yt_stats()
