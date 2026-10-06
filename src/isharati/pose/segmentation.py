"""Sign Language Continuous Pose Segmentation.

Uses the RoPE Transformer model from sign-language-processing/segmentation
to segment continuous sign language into discrete, word-level SIGN gestures
and SENTENCE phrases from MediaPipe Holistic poses.
"""
from pathlib import Path
from typing import Dict, List
import numpy as np

_MODEL = None
_MODEL_DIR = Path(r"D:\islam\sign-language-segmentation\sign_language_segmentation\dist\2026")

# the package's labels (sign_language_segmentation.utils.bio.BIO)
UNK, O, B, I = 0, 1, 2, 3


def get_segmentation_model():
    """Lazy load the pre-trained RoPE Transformer segmentation model."""
    global _MODEL
    if _MODEL is None:
        from sign_language_segmentation.bin import load_model
        _MODEL = load_model(model_dir=str(_MODEL_DIR), device="cpu")
    return _MODEL


def labels_to_segments(preds) -> List[dict]:
    """Per-frame labels (UNK=0, O=1, B=2, I=3) -> [{"start", "end"}] with inclusive frame indices.

    B opens a sign (closing one that runs straight into it), I continues it (or opens one if its B was
    missed), O and UNK close it."""
    segs = []
    seg_start = None
    for i, p in enumerate(np.asarray(preds).tolist()):
        if p == B:
            if seg_start is not None:
                segs.append({"start": seg_start, "end": i - 1})
            seg_start = i
        elif p == I:
            if seg_start is None:
                seg_start = i
        else:  # O or UNK
            if seg_start is not None:
                segs.append({"start": seg_start, "end": i - 1})
                seg_start = None
    if seg_start is not None:
        segs.append({"start": seg_start, "end": len(preds) - 1})
    return segs


def likeliest_probs_to_segments(log_probs) -> List[dict]:
    """[T, 4] per-frame (log-)probabilities -> segments of the argmax labels."""
    if hasattr(log_probs, "cpu"):
        log_probs = log_probs.cpu().numpy()
    return labels_to_segments(np.asarray(log_probs).argmax(axis=1))


def filter_segments(segs: List[dict], min_len: int = 3, gap: int = 0) -> List[dict]:
    """Merge segments separated by <= gap frames, then drop those shorter than min_len frames."""
    if not segs:
        return segs
    if gap > 0:
        merged = [dict(segs[0])]
        for s in segs[1:]:
            if s["start"] - merged[-1]["end"] - 1 <= gap:
                merged[-1]["end"] = s["end"]
            else:
                merged.append(dict(s))
        segs = merged
    return [s for s in segs if s["end"] - s["start"] + 1 >= min_len]


def prepare_input(pose: np.ndarray, fps: float = 25.0):
    """[T,50,3] Isharati pose (or a ready [T,50,6] array) -> (pose features [T,50,6], frame times [T]).

    [T,50,3] is z-scored over its non-zero values and concatenated with its velocity (units/second)."""
    T = len(pose)
    frame_times = np.arange(T, dtype=np.float32) / fps
    if pose.ndim == 3 and pose.shape[1] == 50 and pose.shape[2] == 3:
        clean = np.nan_to_num(pose, nan=0.0).astype(np.float32)
        valid = (clean != 0)
        if valid.any():
            mean = clean[valid].mean()
            std = clean[valid].std() + 1e-6
            norm_pose = np.where(valid, (clean - mean) / std, 0.0).astype(np.float32)
        else:
            norm_pose = clean
        if T > 1:
            vel = np.concatenate([np.zeros_like(norm_pose[:1]),
                                  np.diff(norm_pose, axis=0) / np.diff(frame_times)[:, None, None]], axis=0)
        else:
            vel = np.zeros_like(norm_pose)
        return np.concatenate([norm_pose, vel], axis=-1).astype(np.float32), frame_times
    if pose.ndim == 3 and pose.shape[1] == 50 and pose.shape[2] == 6:
        return pose.astype(np.float32), frame_times
    raise ValueError(f"Expected pose shape [T, 50, 3] or [T, 50, 6], got {pose.shape}")


def pose_log_probs(pose: np.ndarray, fps: float = 25.0) -> Dict[str, np.ndarray]:
    """Run the model: {"sign": [T,4], "sentence": [T,4]} per-frame log-probabilities."""
    import torch

    pose_data, frame_times = prepare_input(pose, fps)
    with torch.inference_mode():
        out = get_segmentation_model()(torch.from_numpy(pose_data).unsqueeze(0),
                                       timestamps=torch.from_numpy(frame_times).unsqueeze(0))
    return {k: out[k][0].cpu().numpy() for k in ("sign", "sentence")}


def segment_pose_array(
    pose: np.ndarray,
    fps: float = 25.0,
    min_frames: int = 4,
    merge_gap: int = 1
) -> Dict[str, List[dict]]:
    """Segment an [T, 50, 3] (Isharati contract) or [T, 50, 6] pose array.

    Args:
        pose: Array of shape [T, 50, 3] (Isharati contract) or [T, 50, 6].
        fps: Frames per second of the source video (default 25.0).
        min_frames: Minimum frame length of a valid sign (default 4 = 160ms at 25fps).
        merge_gap: Merge adjacent segments separated by <= N frames (default 1).

    Returns:
        dict with:
          "signs": list of {"start_frame": int, "end_frame": int, "start_s": float, "end_s": float, "duration_s": float}
          "sentences": list of {"start_frame": int, "end_frame": int, "start_s": float, "end_s": float, "duration_s": float}
    """
    if len(pose) < min_frames:
        return {"signs": [], "sentences": []}

    log_probs = pose_log_probs(pose, fps)
    signs = filter_segments(likeliest_probs_to_segments(log_probs["sign"]), min_len=min_frames, gap=merge_gap)
    sentences = filter_segments(likeliest_probs_to_segments(log_probs["sentence"]),
                                min_len=max(min_frames * 2, 8), gap=merge_gap * 2)

    def to_dict_list(segs):
        res = []
        for s in segs:
            st_s = round(s["start"] / fps, 3)
            en_s = round(s["end"] / fps, 3)
            res.append({
                "start_frame": int(s["start"]),
                "end_frame": int(s["end"]),
                "start_s": st_s,
                "end_s": en_s,
                "duration_s": round(en_s - st_s, 3),
            })
        return res

    return {"signs": to_dict_list(signs), "sentences": to_dict_list(sentences)}
