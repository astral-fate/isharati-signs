import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _black_video(path: Path, frames: int = 10, fps: int = 25):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(frames):
        w.write(np.zeros((64, 64, 3), np.uint8))
    w.release()


def test_extract_on_video_without_person_is_not_ok_and_has_no_nan(tmp_path):
    from isharati.pose.keypoints import extract
    v = tmp_path / "a.mp4"
    _black_video(v)
    r = extract(v)
    assert r.pose.shape[1:] == (50, 3) and r.pose.shape[0] >= 1
    assert r.pose.dtype == np.float32
    assert not np.isnan(r.pose).any()
    assert r.ok is False
    assert r.missing_ratio == 1.0


def test_extract_script_writes_manifest(tmp_path):
    vids = tmp_path / "videos"
    _black_video(vids / "01" / "0071" / "s1.mp4")
    _black_video(vids / "02" / "0072" / "s1.mp4")
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/lexicon/arsl/karsl_extract.py", "--videos", str(vids), "--out", str(out),
           "--regex", r"(?P<signer>\d{2})[\\/](?P<sign_id>\d{4})[\\/]"]
    subprocess.run(cmd, check=True, cwd=Path(__file__).parents[1])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert len(rows) == 2
    assert {r["sign_id"] for r in rows} == {"0071", "0072"}
    assert all(r["ok"] == "False" for r in rows)
    assert all((out / r["npy_path"]).exists() for r in rows)


def _frame_dir(path: Path, frames: int = 6):
    path.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        cv2.imwrite(str(path / f"x_{i + 1:04d}.jpg"), np.zeros((64, 64, 3), np.uint8))


def test_extract_image_dir_without_person_is_not_ok(tmp_path):
    from isharati.pose.keypoints import extract_image_dir
    _frame_dir(tmp_path / "s1")
    r = extract_image_dir(tmp_path / "s1", src_fps=30)
    assert r.ok is False and r.pose.shape[1:] == (50, 3) and not np.isnan(r.pose).any()
    assert r.pose.shape[0] == 5   # 6 frames at 30 fps -> 5 at 25 fps


def test_extract_script_frames_mode_one_row_per_sample_dir(tmp_path):
    root = tmp_path / "karsl"
    _frame_dir(root / "01" / "01" / "train" / "0071" / "01_01_0071_a")
    _frame_dir(root / "01" / "01" / "train" / "0071" / "01_01_0071_b")
    _frame_dir(root / "03" / "03" / "test" / "0072" / "03_03_0072_a")
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/lexicon/arsl/karsl_extract.py", "--frames", "--videos", str(root), "--out", str(out),
           "--regex", r"(?P<signer>\d{2})[\\/]\d{2}[\\/](?:train|test)[\\/](?P<sign_id>\d{4})[\\/]"]
    subprocess.run(cmd, check=True, cwd=Path(__file__).parents[1])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert len(rows) == 3
    assert sorted((r["signer"], r["sign_id"]) for r in rows) == [("01", "0071"), ("01", "0071"), ("03", "0072")]


def test_frame_paths_are_naturally_sorted(tmp_path):
    from isharati.pose.keypoints import frame_paths
    for i in (1, 2, 10, 11, 3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
    assert [p.stem for p in frame_paths(tmp_path)] == ["1", "2", "3", "10", "11"]


def test_corrupt_frames_do_not_crash_extraction(tmp_path):
    from isharati.pose.keypoints import extract_image_dir
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "0001.jpg").write_bytes(b"not a jpeg")
    r = extract_image_dir(tmp_path / "s")
    assert r.ok is False and not np.isnan(r.pose).any()


def test_extract_script_caps_samples_per_signer_and_sign_and_runs_in_parallel(tmp_path):
    root = tmp_path / "karsl"
    for s in ("a", "b", "c"):
        _frame_dir(root / "01" / "01" / "train" / "0071" / f"01_01_0071_{s}")
    _frame_dir(root / "02" / "02" / "train" / "0071" / "02_02_0071_a")
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/lexicon/arsl/karsl_extract.py", "--frames", "--videos", str(root), "--out", str(out),
           "--max-per-group", "2", "--workers", "2",
           "--regex", r"(?P<signer>\d{2})[\\/]\d{2}[\\/](?:train|test)[\\/](?P<sign_id>\d{4})[\\/]"]
    subprocess.run(cmd, check=True, cwd=Path(__file__).parents[1])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert sorted(r["signer"] for r in rows) == ["01", "01", "02"]


def test_extract_script_regex_uses_forward_slashes_on_every_os(tmp_path):
    root = tmp_path / "karsl"
    _frame_dir(root / "01" / "01" / "train" / "0071" / "01_01_0071_(1_2)_c")
    out = tmp_path / "out"
    rx = r"(?P<signer>\d{2})/\d{2}/(?:train|test)/(?P<sign_id>\d{4})/"
    cmd = [sys.executable, "scripts/lexicon/arsl/karsl_extract.py", "--frames", "--videos", str(root), "--out", str(out),
           "--regex", rx]
    subprocess.run(cmd, check=True, cwd=Path(__file__).parents[1])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert [(r["signer"], r["sign_id"]) for r in rows] == [("01", "0071")]
