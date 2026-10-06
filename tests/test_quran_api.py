"""Qur'an dataset API: coverage, ayah lookup, poses, path safety, missing sources."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _source(root, repo, rows):
    d = root / repo
    (d / "isharati").mkdir(parents=True)
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    by = {}
    for r in rows:
        by.setdefault(r["surah"], []).append(r)
    surahs = [{"surah": s, "surah_name": rs[0]["surah_name"], "segments": len(rs),
               "ayahs": len({tuple(a) for r in rs for a in r["ayahs"]}),
               "recitation": sum(r["type"] == "recitation" for r in rs), "tafsir": sum(r["type"] == "tafsir" for r in rs),
               "intro": 0, "seconds": 10.0} for s, rs in sorted(by.items())]
    (d / "surahs.json").write_text(json.dumps(surahs, ensure_ascii=False), encoding="utf-8")
    for s, rs in by.items():
        np.savez(d / "isharati" / f"{s:03d}.npz",
                 **{r["id"]: np.full((5, 50, 3), 0.5, np.float16) for r in rs if r["isharati"]})


def _row(id, typ, surah, ayahs, isharati=True, yt="abc"):
    return {"id": id, "type": typ, "surah": surah, "surah_name": f"s{surah}", "ayahs": ayahs, "text_uthmani": "نص",
            "tafsir_text": "تفسير" if typ == "tafsir" else None, "words": [], "start": 10.0, "end": 14.0,
            "fps_source": 25.0, "frames": 100, "detection_rate": {}, "youtube_id": yt,
            "youtube_url": f"https://www.youtube.com/watch?v={yt}&t=10s" if yt else None, "video_title": "v",
            "isharati": isharati}


@pytest.fixture
def root(tmp_path):
    _source(tmp_path, "quran-sign-kfc", [_row("114_001", "recitation", 114, [[114, 1]]),
                                         _row("114_tafsir_001-002", "tafsir", 114, [[114, 1], [114, 2]], isharati=False)])
    _source(tmp_path, "tafsir-mukhtasar-sign", [_row("002_tafsir_255-255", "tafsir", 2, [[2, 255]], yt=None)])
    return tmp_path                                  # curriculum missing on purpose


def test_quran_data_coverage_ayahs_nearest_pose(root):
    from isharati.quran_data import QuranData
    q = QuranData(root)
    cov = q.coverage()
    assert [s["key"] for s in cov["sources"]] == ["kfc", "mukhtasar"]        # curriculum not loaded, not fatal
    assert len(cov["surahs"]) == 114
    assert cov["surahs"][113]["kfc"] == {"recitation": 1, "tafsir": 1}
    assert cov["surahs"][0]["kfc"] == {"recitation": 0, "tafsir": 0}
    seg = q.segments(114, 1)
    assert [(s["source"], s["type"]) for s in seg] == [("kfc", "recitation"), ("kfc", "tafsir")]
    assert q.segments(1, 1) == [] and q.nearest(1, 1) == [2, 255]
    p = q.pose("kfc", "114_001")
    assert p["fps"] == 25 and len(p["frames"]) == 5 and abs(p["frames"][0][0][0] - 0.5) < 1e-3
    assert q.pose("kfc", "114_tafsir_001-002") is None and q.pose("curriculum", "114_001") is None


@pytest.fixture
def client(root, monkeypatch):
    monkeypatch.setenv("ISHARATI_QURAN_DIR", str(root))
    from isharati import quran_data
    monkeypatch.setattr(quran_data, "_DATA", None)
    from isharati.app import server
    return TestClient(server.app)


def test_endpoints(client):
    assert len(client.get("/api/quran/coverage").json()["surahs"]) == 114
    r = client.get("/api/quran/ayah", params={"surah": 114, "ayah": 2}).json()
    assert [s["id"] for s in r["segments"]] == ["114_tafsir_001-002"] and r["segments"][0]["isharati"] is False
    r = client.get("/api/quran/ayah", params={"surah": 1, "ayah": 1}).json()
    assert r["segments"] == [] and r["nearest"] == [2, 255]
    assert client.get("/api/quran/ayah", params={"surah": 115, "ayah": 1}).status_code == 400
    assert client.get("/api/quran/pose/kfc/114_001.json").json()["fps"] == 25
    assert client.get("/api/quran/pose/nope/114_001.json").status_code == 404
    assert client.get("/api/quran/pose/kfc/..%2F..%2Fx.json").status_code == 404
    assert client.get("/api/quran/pose/kfc/114_tafsir_001-002.json").status_code == 404


def test_hub_mode_retries_failed_source(root, monkeypatch):
    import huggingface_hub
    from isharati import quran_data
    calls = {"n": 0}

    def fake(repo_id, filename, repo_type=None):
        repo = repo_id.split("/")[1]
        if repo == "tafsir-mukhtasar-sign" and calls["n"] < 1:      # first attempt fails (still uploading)
            calls["n"] += 1
            raise OSError("still uploading")
        return str(root / repo / filename)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    clock = [1000.0]
    monkeypatch.setattr(quran_data, "_now", lambda: clock[0])
    q = quran_data.QuranData()
    assert [s["key"] for s in q.coverage()["sources"]] == ["kfc"]
    clock[0] += 30
    assert [s["key"] for s in q.coverage()["sources"]] == ["kfc"]    # not retried inside 60 s
    clock[0] += 31
    assert [s["key"] for s in q.coverage()["sources"]] == ["kfc", "mukhtasar"]
    assert q.segments(2, 255)[0]["source"] == "mukhtasar"


def test_unknown_id_never_loads_a_pack(root, monkeypatch):
    from isharati.quran_data import QuranData
    q = QuranData(root)
    monkeypatch.setattr(q, "_pack", lambda *a: pytest.fail("pack loaded"))
    assert q.pose("kfc", "114_999") is None
    assert q.pose("kfc", "114_tafsir_001-002") is None               # in manifest but isharati false


def test_failed_pack_backs_off_60s(root, monkeypatch):
    from isharati import quran_data
    clock = [500.0]
    monkeypatch.setattr(quran_data, "_now", lambda: clock[0])
    q = quran_data.QuranData(root)
    f = root / "quran-sign-kfc" / "isharati" / "114.npz"
    data = f.read_bytes()
    f.unlink()
    assert q.pose("kfc", "114_001") is None
    f.write_bytes(data)
    clock[0] += 30
    assert q.pose("kfc", "114_001") is None                          # inside the backoff
    clock[0] += 31
    assert q.pose("kfc", "114_001") is not None


def test_slow_pack_load_does_not_block_coverage(root, monkeypatch):
    import threading
    from isharati import quran_data
    q = quran_data.QuranData(root)
    started, release = threading.Event(), threading.Event()
    real = np.load

    def slow(*a, **k):
        started.set()
        release.wait(10)
        return real(*a, **k)
    monkeypatch.setattr(quran_data.np, "load", slow)
    out = {}
    t = threading.Thread(target=lambda: out.update(p=q.pose("kfc", "114_001")))
    t.start()
    assert started.wait(5)
    done = threading.Event()
    threading.Thread(target=lambda: (q.coverage(), done.set())).start()
    assert done.wait(2)                                              # coverage not stuck behind the pack load
    release.set()
    t.join(5)
    assert out["p"] is not None


def test_pose_regex_and_ayah_zero(client):
    assert client.get("/api/quran/pose/kfc/abc.json").status_code == 404
    assert client.get("/api/quran/ayah", params={"surah": 1, "ayah": 0}).status_code == 400
