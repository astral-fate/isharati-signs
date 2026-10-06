"""Create a composite image of the 6 available avatars."""
import cv2
import numpy as np
from pathlib import Path

CLIPS_DIR = Path(r"D:\islam\isharati-signs-landing\web\public\clips")
OUT_PATH = Path(r"D:\islam\isharati\docs\paper\figures\avatar_roster.jpg")

AVATARS = [
    ("avatar_en.jpg", "Cartoon / Anime (Default)", "ASL (English)"),
    ("rocketbox_female_06_en.jpg", "Rocketbox Female 06", "ASL (English)"),
    ("rocketbox_female_10_ar.jpg", "Rocketbox Female 10", "ArSL (Arabic)"),
    ("rocketbox_male_15_tr.jpg", "Rocketbox Male 15", "TİD (Turkish)"),
    ("rocketbox_male_19_ur.jpg", "Rocketbox Male 19", "ISL (Urdu)"),
    ("rocketbox_male_21_ar.jpg", "Rocketbox Male 21", "ArSL (Arabic)"),
]

tiles = []
target_h, target_w = 400, 300

for fname, name, lang in AVATARS:
    img_path = CLIPS_DIR / fname
    if not img_path.exists():
        print(f"Missing: {img_path}")
        continue
    img = cv2.imread(str(img_path))
    # Resize keeping aspect ratio, crop or pad to target
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    # Center crop
    rh, rw = resized.shape[:2]
    start_y = (rh - target_h) // 2
    start_x = (rw - target_w) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    # Add label banner at bottom
    banner_h = 50
    banner = np.ones((banner_h, target_w, 3), dtype=np.uint8) * 20
    cv2.putText(banner, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, lang, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA)
    tile = np.vstack([cropped, banner])
    tiles.append(tile)

if len(tiles) == 6:
    # 2 rows of 3
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    # Add border/gap between columns and rows
    gap = 8
    sep_v = np.ones((row1.shape[0], gap, 3), dtype=np.uint8) * 40
    row1_spaced = np.hstack([tiles[0], sep_v, tiles[1], sep_v, tiles[2]])
    row2_spaced = np.hstack([tiles[3], sep_v, tiles[4], sep_v, tiles[5]])
    sep_h = np.ones((gap, row1_spaced.shape[1], 3), dtype=np.uint8) * 40
    roster = np.vstack([row1_spaced, sep_h, row2_spaced])
    cv2.imwrite(str(OUT_PATH), roster, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Roster saved to: {OUT_PATH} ({roster.shape})")
else:
    print(f"Only found {len(tiles)} tiles.")
