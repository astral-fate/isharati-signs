"""Extract actual frames from each Tafsir dataset video and render MediaPipe keypoints on them."""
import cv2
import numpy as np
from pathlib import Path
import json

FIGURES = Path(r"D:\islam\isharati\docs\paper\figures")

# MediaPipe Pose connections (33 landmarks)
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # shoulders, elbows, wrists
    (11,23),(12,24),(23,24),  # torso
    (0,1),(1,2),(2,3),(3,7),  # right eye
    (0,4),(4,5),(5,6),(6,8),  # left eye
    (9,10),  # mouth
]

# Hand connections (21 landmarks each)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

SOURCES = [
    {
        "label": "King Fahd Complex",
        "video": Path(r"D:\islam\gathring data\kfc_videos\R1 - الإصدار الأول\R1 - الإصدار الأول.mp4"),
        "keypoints": Path(r"D:\islam\gathring data\kfc_videos\R1 - الإصدار الأول\R1 - الإصدار الأول.keypoints.npz"),
        "frame_sec": 120,  # 2 minutes in where signer is actively signing
    },
    {
        "label": "Al-Kharj Curriculum",
        "video": Path(r"D:\islam\gathring data\quran_sign_language_videos\001 - سورة الفاتحة\001 - سورة الفاتحة.mp4"),
        "keypoints": Path(r"D:\islam\gathring data\quran_sign_language_videos\001 - سورة الفاتحة\001 - سورة الفاتحة.keypoints.npz"),
        "frame_sec": 15,
    },
]


def find_good_frame(pose, left_hand, right_hand, start_idx, search_range=200):
    """Find a frame near start_idx where the signer's hands are visible."""
    for offset in range(search_range):
        idx = start_idx + offset
        if idx >= len(pose):
            break
        p = pose[idx]
        lh = left_hand[idx]
        rh = right_hand[idx]
        # Check if at least shoulders + one hand are visible
        shoulders_ok = not (np.isnan(p[11,0]) or np.isnan(p[12,0]))
        lh_ok = np.sum(~np.isnan(lh[:, 0])) > 10
        rh_ok = np.sum(~np.isnan(rh[:, 0])) > 10
        if shoulders_ok and (lh_ok or rh_ok):
            return idx
    return start_idx


def draw_on_frame(img, pose_kp, lh_kp, rh_kp):
    """Draw MediaPipe keypoints on a BGR image. Coords are normalized [0,1]."""
    h, w = img.shape[:2]
    overlay = img.copy()

    def px(kp, idx):
        x, y = float(kp[idx, 0]), float(kp[idx, 1])
        if np.isnan(x) or np.isnan(y):
            return None
        return (int(x * w), int(y * h))

    # Pose (body)
    for i, j in POSE_CONNECTIONS:
        p1, p2 = px(pose_kp, i), px(pose_kp, j)
        if p1 and p2:
            cv2.line(overlay, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
    for i in range(33):
        p = px(pose_kp, i)
        if p:
            cv2.circle(overlay, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

    # Left hand
    for i, j in HAND_CONNECTIONS:
        p1, p2 = px(lh_kp, i), px(lh_kp, j)
        if p1 and p2:
            cv2.line(overlay, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
    for i in range(21):
        p = px(lh_kp, i)
        if p:
            cv2.circle(overlay, p, 3, (255, 200, 0), -1, cv2.LINE_AA)

    # Right hand
    for i, j in HAND_CONNECTIONS:
        p1, p2 = px(rh_kp, i), px(rh_kp, j)
        if p1 and p2:
            cv2.line(overlay, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    for i in range(21):
        p = px(rh_kp, i)
        if p:
            cv2.circle(overlay, p, 3, (0, 200, 255), -1, cv2.LINE_AA)

    return overlay


def process_source(src):
    label = src["label"]
    print(f"\n=== {label} ===")

    # Load keypoints
    data = np.load(str(src["keypoints"]), allow_pickle=True)
    pose = data["pose"][:, :, :3]  # (N, 33, 3) - drop visibility
    lh = data["left_hand"]  # (N, 21, 3)
    rh = data["right_hand"]  # (N, 21, 3)
    meta = json.loads(str(data["meta"]))
    kp_fps = meta["fps"]
    print(f"  Keypoints: {len(pose)} frames at {kp_fps} fps")

    # Find a good frame
    start_kp = int(src["frame_sec"] * kp_fps)
    good_kp = find_good_frame(pose, lh, rh, start_kp)
    good_sec = good_kp / kp_fps
    print(f"  Good frame at kp_idx={good_kp} ({good_sec:.1f}s)")

    # Read video frame
    cap = cv2.VideoCapture(str(src["video"]))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_frame_idx = int(good_sec * vid_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("  ERROR: Could not read video frame")
        return None
    print(f"  Video frame {vid_frame_idx}, shape={frame.shape}")

    # Render keypoints
    frame_kp = draw_on_frame(frame.copy(), pose[good_kp], lh[good_kp], rh[good_kp])

    # Save side by side
    gap = 10
    h, w = frame.shape[:2]
    composite = np.ones((h, w*2 + gap, 3), dtype=np.uint8) * 30
    composite[:, :w] = frame
    composite[:, w+gap:] = frame_kp

    safe = label.replace(" ", "_")
    out_path = FIGURES / f"tafsir_{safe}.png"
    cv2.imwrite(str(out_path), composite)
    print(f"  Saved: {out_path.name}")
    return out_path, composite


if __name__ == "__main__":
    results = []
    for src in SOURCES:
        if src["video"].exists() and src["keypoints"].exists():
            r = process_source(src)
            if r:
                results.append(r)
        else:
            print(f"SKIP: {src['label']} - files missing")

    if results:
        # Stack vertically with labels
        panels = []
        for (path, img), src in zip(results, SOURCES):
            bar_h = 50
            bar = np.ones((bar_h, img.shape[1], 3), dtype=np.uint8) * 30
            cv2.putText(bar, f"{src['label']}  |  Original (left) vs Rendered Keypoints (right)",
                       (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(np.vstack([bar, img]))

        # Resize all to same width
        max_w = max(p.shape[1] for p in panels)
        resized = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.ones((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8) * 30
                p = np.hstack([p, pad])
            resized.append(p)

        separator = np.ones((8, max_w, 3), dtype=np.uint8) * 60
        final = resized[0]
        for p in resized[1:]:
            final = np.vstack([final, separator, p])

        final_path = FIGURES / "tafsir_datasets_composite.png"
        cv2.imwrite(str(final_path), final)
        print(f"\nFinal composite: {final_path}")

    print("Done!")
