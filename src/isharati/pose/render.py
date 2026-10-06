"""Pose [T,50,3] -> skeleton mp4 (always available, CPU only)."""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from isharati.pose.keypoints import L_EL, L_SH, L_WR, NECK, NOSE, R_EL, R_SH, R_WR
from isharati.types import FPS

BODY = [(NECK, NOSE), (NECK, R_SH), (R_SH, R_EL), (R_EL, R_WR), (NECK, L_SH), (L_SH, L_EL), (L_EL, L_WR)]
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
BG, TEAL, AMBER, RED, WHITE = (27, 42, 58), (20, 184, 166), (245, 158, 11), (220, 38, 38), (230, 236, 240)


class SkeletonRenderer:
    def __init__(self, size: int = 512):
        self.size = size

    def _px(self, p):
        s = self.size
        x = float(p[0]) if (len(p) > 0 and np.isfinite(p[0])) else 0.0
        y = float(p[1]) if (len(p) > 1 and np.isfinite(p[1])) else 0.0
        return int(s / 2 + x * s / 5), int(s * 0.35 + y * s / 5)

    def _frame(self, f: np.ndarray) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), np.uint8)
        img[:] = BG
        for a, b in BODY:
            if np.all(np.isfinite(f[a])) and np.all(np.isfinite(f[b])):
                cv2.line(img, self._px(f[a]), self._px(f[b]), WHITE, 3, cv2.LINE_AA)
        for base, wrist, col in ((8, L_WR, AMBER), (29, R_WR, RED)):
            if np.all(np.isfinite(f[wrist])) and np.all(np.isfinite(f[base])):
                cv2.line(img, self._px(f[wrist]), self._px(f[base]), WHITE, 2, cv2.LINE_AA)
            for a, b in HAND:
                if np.all(np.isfinite(f[base + a])) and np.all(np.isfinite(f[base + b])):
                    cv2.line(img, self._px(f[base + a]), self._px(f[base + b]), col, 2, cv2.LINE_AA)
        for j in range(8):
            if np.all(np.isfinite(f[j])):
                cv2.circle(img, self._px(f[j]), 4, TEAL, -1, cv2.LINE_AA)
        return img

    def render(self, pose: np.ndarray, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pose_clean = np.nan_to_num(pose, nan=0.0).astype(np.float32)
        with imageio.get_writer(out_path, fps=FPS, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, quality=7) as w:
            for f in pose_clean:
                w.append_data(self._frame(f))
        return out_path
