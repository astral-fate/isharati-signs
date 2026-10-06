"""The Qur'an sign-language dataset publisher: Holistic -> Isharati conversion, manifest rows, per-surah packing."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pub():
    spec = importlib.util.spec_from_file_location("publish_quran_sign", ROOT / "scripts" / "hub" / "publish_quran_sign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def holistic(T=30, fps=15.0, hands=True):
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.3, 0.7, (T, 33, 4)).astype(np.float32)
    pose[:, 11, :3], pose[:, 12, :3] = (0.6, 0.4, 0.0), (0.4, 0.4, 0.0)      # shoulders 0.2 apart, same depth
    lh = rng.uniform(0.3, 0.7, (T, 21, 3)).astype(np.float32)
    rh = rng.uniform(0.3, 0.7, (T, 21, 3)).astype(np.float32)
    if not hands:
        lh[:] = np.nan
    lh[3] = np.nan                                                           # one undetected frame
    return {"pose": pose, "left_hand": lh, "right_hand": rh, "face": np.zeros((T, 478, 3), np.float32),
            "time": (100 + np.arange(T) / fps).astype(np.float32), "frame": np.arange(T)}


def test_to_isharati_resamples_to_25_fps_and_normalises(pub):
    p = pub.to_isharati(holistic(T=30, fps=15.0))                           # 2 s at 15 fps
    assert p.dtype == np.float16 and p.shape[1:] == (50, 3)
    assert len(p) == int((29 / 15.0) * 25) + 1
    assert np.isfinite(p.astype(np.float32)).all()
    assert np.allclose(p[:, 1].astype(np.float32), 0, atol=1e-3)            # neck-centred
    width = np.linalg.norm(p[:, 2].astype(np.float32) - p[:, 5].astype(np.float32), axis=-1)
    assert np.allclose(width, 1, atol=1e-2)                                  # shoulder-width units (3-D, as normalize)


def test_to_isharati_without_shoulders_is_none(pub):
    z = holistic()
    z["pose"][:] = np.nan
    assert pub.to_isharati(z) is None


def test_manifest_rows_normalise_both_layouts(pub):
    yt = {"001 - x.mp4": "abc"}
    rec = {"id": "001_001", "type": "recitation", "surah": 1, "surah_name": "الفاتحة", "video": "001 - x.mp4",
           "fps": 25.0, "start": 12.7, "end": 15.0, "clip_frames": 60, "detection_rate": {"pose": 1.0},
           "ayah": 1, "text_uthmani": "بِسْمِ ٱللَّهِ", "words": [{"word": "بسم", "start": 12.7, "end": 13.0, "clip_index": [0, 8]}]}
    taf = {"id": "002_tafsir_002-003", "type": "tafsir", "surah": 2, "video": "002 - y.mp4", "fps": 30.0,
           "start": 72.2, "end": 86.8, "ayahs": [[2, 2], [2, 3]],
           "ayah_texts": [{"surah": 2, "ayah": 2, "text_uthmani": "ذَٰلِكَ"}, {"surah": 2, "ayah": 3, "text_uthmani": "ٱلَّذِينَ"}],
           "text": "ذلك القرآن العظيم", "words": []}
    a, b = pub.manifest_row(rec, yt), pub.manifest_row(taf, yt)
    assert a["ayahs"] == [[1, 1]] and a["tafsir_text"] is None and a["words"][0]["frames"] == [0, 8]
    assert a["youtube_url"] == "https://www.youtube.com/watch?v=abc&t=12s" and a["video_title"] == "001 - x"
    assert b["ayahs"] == [[2, 2], [2, 3]] and b["text_uthmani"] == "ذَٰلِكَ ٱلَّذِينَ"
    assert b["tafsir_text"] == "ذلك القرآن العظيم" and b["youtube_id"] is None and b["youtube_url"] is None


def test_build_packs_per_surah_and_counts_match(pub, tmp_path, monkeypatch):
    src = tmp_path / "gathered"
    ds = src / "dataset" / "quran_sign_language_videos"
    (ds / "clips").mkdir(parents=True)
    rows = []
    for i, (s, a) in enumerate([(1, 1), (1, 2), (114, 1)]):
        sid = f"{s:03d}_{a:03d}"
        # surah 114's video has no info.json (no YouTube id): its frame size comes from the clip's own metadata
        meta = {"meta": json.dumps({"width": 1280, "height": 720})} if s == 114 else {}
        np.savez(ds / "clips" / f"{sid}.npz", **holistic(), **meta)
        rows.append({"id": sid, "type": "recitation", "surah": s, "surah_name": f"s{s}", "video": f"{s:03d} - v.mp4",
                     "fps": 15.0, "start": float(i), "end": float(i) + 2, "clip_frames": 30, "ayah": a,
                     "text_uthmani": "x", "words": [], "detection_rate": {"pose": 1.0}})
    rows.append({"id": "114_002", "type": "recitation", "surah": 114, "surah_name": "s114", "video": "114 - v.mp4",
                 "fps": 15.0, "start": 9.0, "end": 10.0, "clip_frames": 0, "ayah": 2, "text_uthmani": "y", "words": []})
    (ds / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    vid = src / "quran_sign_language_videos" / "001 - v"
    vid.mkdir(parents=True)
    (vid / "001 - v.info.json").write_text(json.dumps({"id": "yt001", "width": 1280, "height": 720}), encoding="utf-8")

    out = pub.build("curriculum", src_root=src, out_root=tmp_path / "out")
    man = [json.loads(l) for l in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["id"] for m in man] == ["001_001", "001_002", "114_001", "114_002"]
    assert [m["isharati"] for m in man] == [True, True, True, False]          # 114_002 has no keypoints file
    assert man[0]["youtube_id"] == "yt001" and man[2]["youtube_id"] is None
    surahs = json.loads((out / "surahs.json").read_text(encoding="utf-8"))
    assert [(s["surah"], s["segments"], s["ayahs"]) for s in surahs] == [(1, 2, 2), (114, 2, 2)]
    hol = np.load(out / "holistic" / "001.npz")
    assert hol["001_002/face"].shape == (30, 478, 3) and hol["001_002/time"].shape == (30,)
    ish = np.load(out / "isharati" / "114.npz")
    assert list(ish.files) == ["114_001"] and ish["114_001"].shape[1:] == (50, 3)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "permission" in readme and "112 of 114 surahs are not in this source" in readme
