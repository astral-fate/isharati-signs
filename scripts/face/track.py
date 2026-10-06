"""Stage 1 of the face pass: a full MediaPipe Holistic track of one source video, with the face.

The sign lexicons kept 50 points per frame (8 upper-body + 2x21 hand) and dropped the face. This re-runs the video
through the Holistic Landmarker task (models/holistic_landmarker.task) on the same 25 fps grid and the same crop as
the lexicon builder, so the body and hand points line up with the stored clips (scripts/face/attach.py finds each
clip in this track), and keeps per frame:

  pose (T,33,4) x,y,z,visibility   left_hand / right_hand (T,21,3)   face (T,478,3) float16
  blend (T,52) float16             the face blendshape scores (ARKit names in meta["blend_names"]): brows, eyes,
                                   cheeks, jaw, mouth shapes; what drives an avatar's face
  time (T,) seconds in the source video

Coordinates are normalised to the crop, as the builders' MediaPipe calls saw it (x by crop width, y by crop height), or
to the whole frame with --frame-coords (gathring data/extract_keypoints.py, the Qur'an curriculum); meta["aspect"]
(width / height of that frame of reference) corrects them to square units. NaN = not detected. left_hand is the signer's own.

--fps: the grid's rate (default 25); the source's own rate gives every frame (new_arabic_sources kept every frame).
--until: stop after this many seconds of video (new_arabic_sources' builder only ran the first 90 s).

  "<gathring data>/.venv/Scripts/python" scripts/face/track.py VIDEO OUT.npz [--crop x0,y0,x1,y1 pixels] [--fps F] [--until S]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FPS = 25  # isharati.types.FPS: the lexicon clips' frame rate
MODEL = Path(r"D:\islam\gathring data\models\holistic_landmarker.task")
PARTS = {"pose": ((33, 4), np.float32), "left_hand": ((21, 3), np.float32), "right_hand": ((21, 3), np.float32),
         "face": ((478, 3), np.float16), "blend": ((52,), np.float16)}


def _fill(row, landmarks, box=(0.0, 0.0, 1.0, 1.0)):
    """box: where the crop sits in the coordinate frame wanted (the crop itself by default)."""
    if not landmarks:
        return
    if isinstance(landmarks[0], list):  # some result fields are per-person lists
        landmarks = landmarks[0]
    bx0, by0, bx1, by1 = box
    for j, lm in enumerate(landmarks[:row.shape[0]]):
        row[j, :3] = (bx0 + lm.x * (bx1 - bx0), by0 + lm.y * (by1 - by0), lm.z)
        if row.shape[1] == 4:
            row[j, 3] = lm.visibility if lm.visibility is not None else np.nan


def track(video: Path, crop=None, frame_coords=False, fps=FPS, until=None):
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = crop or (0, 0, w, h)
    # frame_coords: coordinates as fractions of the whole frame, like gathring data/extract_keypoints.py (the Qur'an
    # curriculum clips), instead of fractions of the crop (the lexicon builders' own MediaPipe calls)
    box = (x0 / w, y0 / h, x1 / w, y1 / h) if frame_coords else (0.0, 0.0, 1.0, 1.0)
    # the builders' grid: n source frames -> round(n * 25 / fps) picks spread evenly (islamic_terms.extract)
    picks = np.linspace(0, n_frames - 1, int(round(n_frames * fps / src_fps))).round().astype(int)
    if until is not None:  # the same grid, cut short
        picks = picks[picks < until * src_fps]
    data = {k: np.full((len(picks), *shape), np.nan, dtype) for k, (shape, dtype) in PARTS.items()}
    names = []
    options = vision.HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                               running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True)
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        i, k = 0, 0
        while k < len(picks):
            ok, frame = cap.read()
            if not ok:
                break
            while k < len(picks) and picks[k] == i:  # a source frame can be picked twice when upsampling
                rgb = np.ascontiguousarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                r = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                int(k * 1000 / fps))
                _fill(data["pose"][k], r.pose_landmarks, box)
                _fill(data["left_hand"][k], r.left_hand_landmarks, box)
                _fill(data["right_hand"][k], r.right_hand_landmarks, box)
                _fill(data["face"][k], r.face_landmarks, box)
                bs = r.face_blendshapes[0] if r.face_blendshapes and isinstance(r.face_blendshapes[0], list) \
                    else r.face_blendshapes
                if bs:
                    data["blend"][k] = [c.score for c in bs[:52]]
                    names = names or [c.category_name for c in bs[:52]]
                k += 1
            i += 1
    cap.release()
    data["time"] = (picks / src_fps).astype(np.float32)
    meta = {"video": str(video), "src_fps": src_fps, "width": w, "height": h, "crop": [x0, y0, x1, y1],
            "aspect": (w if frame_coords else x1 - x0) / (h if frame_coords else y1 - y0), "frame_coords": frame_coords,
            "frames": int(k), "fps": fps, "blend_names": names}
    return data, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--crop", help="x0,y0,x1,y1 in pixels (default: the whole frame)")
    ap.add_argument("--frame-coords", action="store_true", help="coordinates relative to the whole frame, not the crop")
    ap.add_argument("--fps", type=float, default=FPS, help="grid rate (default 25)")
    ap.add_argument("--until", type=float, help="stop after this many seconds of video")
    a = ap.parse_args()
    if a.out.exists():
        print("exists:", a.out)
        return
    data, meta = track(a.video, tuple(int(v) for v in a.crop.split(",")) if a.crop else None, a.frame_coords, a.fps, a.until)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + ".tmp.npz")  # atomic: a killed run never leaves a half file
    np.savez_compressed(tmp, meta=json.dumps(meta), **data)
    os.replace(tmp, a.out)
    face = float(np.mean(~np.isnan(data["blend"][:, 0])))
    print(f"{a.video.name}: {meta['frames']} frames at {a.fps:g} fps, face found in {face:.0%} -> {a.out}")


if __name__ == "__main__":
    main()
