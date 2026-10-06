"""Signed hadith dataset API: coverage, browse list, per-hadith samples, poses, path safety, a missing/growing dataset."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

# trimmed from hub_build/hadith-sign/manifest.jsonl (texts shortened, holistic fields kept as they are published)
NAWAWI_12 = ("عَنْ أَبِي هُرَيْرَةَ رَضِيَ اللهُ عَنْهُ قَالَ: قَالَ رَسُولُ اللَّهِ صلى الله عليه و سلم مِنْ حُسْنِ إسْلَامِ "
             "الْمَرْءِ تَرْكُهُ مَا لَا يَعْنِيهِ . حَدِيثٌ حَسَنٌ، رَوَاهُ التِّرْمِذِيُّ")
MUSLIM_2495 = ("حَدَّثَنَا يَحْيَى بْنُ أَيُّوبَ، عَنْ أَبِي هُرَيْرَةَ، أَنَّ رَسُولَ اللَّهِ صلى الله عليه وسلم قَالَ ‏ \"‏ "
               "إِذَا جَاءَ رَمَضَانُ فُتِّحَتْ أَبْوَابُ الْجَنَّةِ ‏\"‏ ‏.‏")


def _hadith(ref, name, text, gloss=None, also_in=(), conflict=False, read=(5.3, 17.0)):
    coll, num = ref.split(":")
    return {"ref": ref, "collection": coll, "collection_name": name, "number": num, "also_in": list(also_in),
            "text_ar": text, "text_en": "Part of the perfection of one's Islam…" if coll == "nawawi" else None,
            "text_tr": None, "read_start": read[0], "read_end": read[1], "matched_words": 7, "coverage": 0.6,
            "asr_text": None, "via": "translation", "onscreen_text": None, "title_conflict": conflict, "gloss": gloss}


def _row(id, source, sl, label, hadiths, isharati=True, start=0.0, end=17.0):
    return {"id": id, "source": source, "source_name": {"eray_40hadis": "Eray Demir: İşaret Dili ile 40 Hadis",
                                                         "arsl_misc": "أحاديث متفرقة بلغة الإشارة"}[source],
            "kind": "clip", "sign_language": sl, "spoken_language": "tr" if sl == "TİD" else "ar", "channel": "c",
            "video_id": id, "title": "t", "youtube_url": f"https://www.youtube.com/watch?v={id}&t={int(start)}s",
            "start": start, "end": end, "frames": 510, "fps": 30.0, "stride": 1, "detection_rate": {},
            "label_source": label, "title_ref": None, "hadiths": hadiths, "keypoints": f"holistic/{source}-000.npz",
            "isharati": isharati, "isharati_file": f"isharati/{source}-000.npz" if isharati else None}


ROWS = [
    _row("xVlmf7cW2LE", "eray_40hadis", "TİD", "translation_match",
         [_hadith("nawawi:12", "الأربعون النووية", NAWAWI_12, gloss={"ArSL": [{"text": "حسن", "oov": False},
                                                                              {"text": "إسلام", "oov": True}]})]),
    _row("y1074REjsEM", "arsl_misc", "ArSL", "speech_match",
         [_hadith("muslim:2495", "صحيح مسلم", MUSLIM_2495, also_in=["bukhari:1898"],
                  gloss={"ArSL": [{"text": "رمضان", "oov": False}]})], end=24.94),
    _row("QXs2MiJT67k", "eray_40hadis", "TİD", "translation_match",
         [_hadith("muslim:2495", "صحيح مسلم", MUSLIM_2495)]),
    _row("6MiyfHEVhxU", "eray_40hadis", "TİD", "none", []),                                       # unlabelled
    _row("CONFLICT001", "eray_40hadis", "TİD", "title+speech_match",
         [_hadith("bukhari:1", "صحيح البخاري", "نص", conflict=True)]),                           # title disagrees
    _row("NOPOSE00001", "eray_40hadis", "TİD", "translation_match",
         [_hadith("bukhari:2", "صحيح البخاري", "نص")], isharati=False),                          # no Isharati pose
]


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "hadith-sign"
    (d / "isharati").mkdir(parents=True)
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS), encoding="utf-8")
    packs: dict[str, dict] = {}
    for r in ROWS:
        if r["isharati"]:
            packs.setdefault(r["isharati_file"], {})[r["id"]] = np.full((5, 50, 3), 0.5, np.float16)
    for name, arrays in packs.items():
        np.savez(d / name, **arrays)
    return tmp_path


def test_hadith_data_coverage_list_item_pose(root):
    from isharati.hadith_data import HadithData
    q = HadithData(root)
    cov = q.coverage()
    assert cov["samples"] == 3 and cov["hadith"] == 2
    assert cov["skipped"] == {"unlabelled": 1, "title_conflict": 1, "no_pose": 1}
    assert [c["collection"] for c in cov["collections"]] == ["nawawi", "muslim"]              # Nawawi's Forty first
    assert {s["sign_language"]: (s["samples"], s["hadith"]) for s in cov["sign_languages"]} == {"ArSL": (1, 1),
                                                                                                "TİD": (2, 2)}
    assert {s["key"]: s["samples"] for s in cov["sources"]} == {"eray_40hadis": 2, "arsl_misc": 1}
    lst = q.hadith_list()
    assert [h["ref"] for h in lst] == ["nawawi:12", "muslim:2495"]
    assert lst[0]["text"] == "مِنْ حُسْنِ إسْلَامِ الْمَرْءِ تَرْكُهُ مَا لَا يَعْنِيهِ"                # no isnad, no grading
    assert lst[1]["text"].startswith("إِذَا جَاءَ رَمَضَانُ") and lst[1]["samples"] == 2
    assert [h["ref"] for h in q.hadith_list(sign_language="ArSL")] == ["muslim:2495"]
    assert [h["ref"] for h in q.hadith_list(collection="nawawi")] == ["nawawi:12"]
    item = q.item("muslim:2495")
    assert [s["id"] for s in item["samples"]] == ["y1074REjsEM", "QXs2MiJT67k"]                # ArSL first
    assert item["text_ar"] == MUSLIM_2495 and item["samples"][0]["youtube_url"].endswith("&t=0s")
    assert item["samples"][0]["gloss"]["ArSL"][0]["text"] == "رمضان"
    assert q.item("bukhari:1898")["ref"] == "muslim:2495"                                       # a parallel narration
    assert q.item("bukhari:1") is None and q.item("bukhari:2") is None                        # flagged rows stay out
    p = q.pose("xVlmf7cW2LE")
    assert p["fps"] == 25 and len(p["frames"]) == 5 and abs(p["frames"][0][0][0] - 0.5) < 1e-3
    assert q.pose("6MiyfHEVhxU") is None and q.pose("CONFLICT001") is None and q.pose("nope") is None


def test_root_may_be_the_dataset_folder(root):
    from isharati.hadith_data import HadithData
    assert HadithData(root / "hadith-sign").coverage()["hadith"] == 2


@pytest.fixture
def client(root, monkeypatch):
    monkeypatch.setenv("ISHARATI_HADITH_DIR", str(root))
    from isharati import hadith_data
    monkeypatch.setattr(hadith_data, "_DATA", None)
    from isharati.app import server
    return TestClient(server.app)


def test_endpoints(client):
    assert client.get("/api/hadith/coverage").json()["hadith"] == 2
    assert [h["ref"] for h in client.get("/api/hadith/list").json()["hadith"]] == ["nawawi:12", "muslim:2495"]
    r = client.get("/api/hadith/list", params={"collection": "muslim", "sign_language": "TİD"}).json()["hadith"]
    assert [(h["ref"], h["samples"]) for h in r] == [("muslim:2495", 1)]
    assert client.get("/api/hadith/list", params={"sign_language": "ASL"}).status_code == 400
    assert client.get("/api/hadith/list", params={"collection": "../x"}).status_code == 400
    item = client.get("/api/hadith/item", params={"ref": "nawawi:12"}).json()
    assert item["collection_name"] == "الأربعون النووية" and item["samples"][0]["read_start"] == 5.3
    assert client.get("/api/hadith/item", params={"ref": "nawawi:99"}).status_code == 404
    assert client.get("/api/hadith/item", params={"ref": "x"}).status_code == 400
    assert client.get("/api/hadith/pose/xVlmf7cW2LE.json").json()["fps"] == 25
    assert client.get("/api/hadith/pose/6MiyfHEVhxU.json").status_code == 404                  # unlabelled
    assert client.get("/api/hadith/pose/..%2F..%2Fx.json").status_code == 404


def test_hub_mode_retries_then_reloads_growing_dataset(root, monkeypatch):
    import huggingface_hub
    from isharati import hadith_data
    calls = {"n": 0}

    def fake(repo_id, filename, repo_type=None):
        assert repo_id == "FatimahEmadEldin/hadith-sign" and repo_type == "dataset"
        assert not filename.startswith("holistic/")
        if calls["n"] < 1:                                       # first attempt fails (still uploading)
            calls["n"] += 1
            raise OSError("still uploading")
        return str(root / "hadith-sign" / filename)
    monkeypatch.delenv("ISHARATI_HADITH_OWNER", raising=False)
    monkeypatch.delenv("ISHARATI_QURAN_OWNER", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    clock = [1000.0]
    monkeypatch.setattr(hadith_data, "_now", lambda: clock[0])
    q = hadith_data.HadithData()
    assert q.coverage()["hadith"] == 0
    clock[0] += 30
    assert q.coverage()["hadith"] == 0                           # not retried inside 60 s
    clock[0] += 31
    assert q.coverage()["hadith"] == 2
    assert q.pose("xVlmf7cW2LE") is not None
    m = root / "hadith-sign" / "manifest.jsonl"                  # republished with one more labelled sample
    extra = _row("UUK_PfQ4aJ0", "eray_40hadis", "TİD", "translation_match",
                 [_hadith("muslim:6030", "صحيح مسلم", "نص الحديث")])
    m.write_text(m.read_text(encoding="utf-8") + "\n" + json.dumps(extra, ensure_ascii=False), encoding="utf-8")
    clock[0] += 600
    assert q.coverage()["hadith"] == 2                           # inside the 30 min reload interval
    clock[0] += 1800
    assert q.coverage()["hadith"] == 3


def test_failed_pack_backs_off_60s(root, monkeypatch):
    from isharati import hadith_data
    clock = [500.0]
    monkeypatch.setattr(hadith_data, "_now", lambda: clock[0])
    q = hadith_data.HadithData(root)
    f = root / "hadith-sign" / "isharati" / "eray_40hadis-000.npz"
    data = f.read_bytes()
    f.unlink()
    assert q.pose("xVlmf7cW2LE") is None
    f.write_bytes(data)
    clock[0] += 30
    assert q.pose("xVlmf7cW2LE") is None                         # inside the backoff
    clock[0] += 31
    assert q.pose("xVlmf7cW2LE") is not None


def test_pack_name_from_manifest_cannot_leave_the_dataset(root, monkeypatch):
    from isharati import hadith_data
    q = hadith_data.HadithData(root)
    q._by_id["xVlmf7cW2LE"] = {**q._by_id["xVlmf7cW2LE"], "isharati_file": "../../secret.npz"}
    monkeypatch.setattr(q, "_pack", lambda *a: pytest.fail("pack loaded"))
    assert q.pose("xVlmf7cW2LE") is None


def test_short_text():
    from isharati.hadith_data import short_text
    assert short_text("قال رسول الله صلى الله عليه وسلم \"إنما الأعمال بالنيات\"") == "إنما الأعمال بالنيات"
    assert short_text("أ" * 200).endswith("…") and len(short_text("أ" * 200)) == 121
    assert short_text(None) == ""
