"""The MediaPipe landmark diagrams (33 body, 21 hand), numbered and named, as schematic layouts; the landmarks
Isharati keeps are highlighted. Run from the project root: .venv/Scripts/python docs/paper/tools/make_mediapipe_figure.py
"""
import json
import struct
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "savefig.bbox": "tight", "savefig.dpi": 300})
TEAL, GREY, INK, RED = "#14b8a6", "#b8bcc4", "#1d1d1b", "#dc2626"

POSE = ["nose", "left eye (inner)", "left eye", "left eye (outer)", "right eye (inner)", "right eye", "right eye (outer)",
        "left ear", "right ear", "mouth (left)", "mouth (right)", "left shoulder", "right shoulder", "left elbow",
        "right elbow", "left wrist", "right wrist", "left pinky", "right pinky", "left index", "right index",
        "left thumb", "right thumb", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
        "left heel", "right heel", "left foot index", "right foot index"]
POSE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
              (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
              (30, 32), (27, 31), (28, 32)]
KEPT_POSE = {0, 11, 12, 13, 14, 15, 16}   # + the neck, computed as the mid-point of the shoulders
HAND = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP", "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# schematic layouts in the MediaPipe definitions: a front-facing body (the subject's right is the viewer's left) and an
# open right hand, palm towards the viewer; x right, y up
BODY_XY = [(0, 8.6), (0.15, 8.85), (0.3, 8.88), (0.45, 8.85), (-0.15, 8.85), (-0.3, 8.88), (-0.45, 8.85), (0.7, 8.65),
           (-0.7, 8.65), (0.18, 8.3), (-0.18, 8.3), (1.4, 7.4), (-1.4, 7.4), (1.9, 5.6), (-1.9, 5.6), (2.2, 3.9),
           (-2.2, 3.9), (2.5, 3.45), (-2.5, 3.45), (2.25, 3.2), (-2.25, 3.2), (1.95, 3.5), (-1.95, 3.5), (0.85, 3.9),
           (-0.85, 3.9), (0.9, 2.0), (-0.9, 2.0), (0.95, 0.35), (-0.95, 0.35), (0.8, 0.0), (-0.8, 0.0), (1.3, -0.1),
           (-1.3, -0.1)]
HAND_XY = [(0, 0), (-0.38, 0.22), (-0.66, 0.5), (-0.86, 0.78), (-1.02, 1.04), (-0.32, 1.0), (-0.38, 1.42), (-0.41, 1.68),
           (-0.44, 1.94), (0, 1.06), (0, 1.52), (0, 1.81), (0, 2.08), (0.3, 1.0), (0.34, 1.42), (0.37, 1.67), (0.39, 1.9),
           (0.56, 0.88), (0.66, 1.16), (0.72, 1.36), (0.76, 1.56)]


def legend(ax, names, kept, x0, y0, dy, ncol=1, rows=None, bold=True):
    rows = rows or len(names)
    for i, n in enumerate(names):
        c, r = divmod(i, rows)
        ax.text(x0 + c * 2.6, y0 - r * dy, f"{i:>2}  {n}", fontsize=6.3, family="serif",
                color="#0f766e" if i in kept else "#555555", fontweight="bold" if bold and i in kept else "normal")


def main():
    body, hand = np.array(BODY_XY), np.array(HAND_XY)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.02})
    for a, c in POSE_EDGES:
        k = a in KEPT_POSE and c in KEPT_POSE
        ax.plot(*body[[a, c]].T, color=TEAL if k else GREY, lw=2.2 if k else 1.2, zorder=1)
    for i, (x, y) in enumerate(body):
        k = i in KEPT_POSE
        ax.scatter(x, y, s=34 if k else 16, color=TEAL if k else "#e5e7eb", edgecolor=INK, lw=0.5, zorder=2)
        if i not in range(1, 11):              # the face points are numbered in the inset list only
            ax.text(x + (0.18 if x >= 0 else -0.18), y, str(i), fontsize=6, ha="left" if x >= 0 else "right",
                    va="center", zorder=3)
    neck = body[[11, 12]].mean(0)
    ax.scatter(*neck, s=36, marker="D", color=RED, zorder=3)
    ax.text(neck[0], neck[1] + 0.3, "neck = mid-shoulders", fontsize=6, color=RED, ha="center")
    ax.text(0, 9.3, "1-10: eyes, ears, mouth", fontsize=6, ha="center", color="#555555")
    legend(ax, POSE, KEPT_POSE, 3.3, 9.0, 0.29, rows=33)
    ax.set_xlim(-3.0, 6.4); ax.set_ylim(-0.6, 9.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Body: 33 landmarks", fontsize=9)
    for a, c in HAND_EDGES:
        bx.plot(*hand[[a, c]].T, color=TEAL, lw=2, zorder=1)
    for i, (x, y) in enumerate(hand):
        bx.scatter(x, y, s=36, color=TEAL, edgecolor=INK, lw=0.5, zorder=2)
        bx.text(x + (0.08 if x >= 0 else -0.08), y + 0.05, str(i), fontsize=6.5,
                ha="left" if x >= 0 else "right", zorder=3)
    legend(bx, [h.replace("_", " ") for h in HAND], set(range(21)), 1.2, 2.2, 0.118, rows=21, bold=False)
    bx.set_xlim(-1.3, 2.9); bx.set_ylim(-0.25, 2.35); bx.set_aspect("equal"); bx.axis("off")
    bx.set_title("Hand: 21 landmarks (each hand)", fontsize=9)
    fig.savefig(FIG / "mediapipe_landmarks.pdf")
    print("ok")


if __name__ == "__main__":
    main()
