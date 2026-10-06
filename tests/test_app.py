"""The web app: avatar list, page, and replay of a saved answer (no model calls)."""
import json

import numpy as np
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    from isharati.app import server
    monkeypatch.setattr(server, "OUT", tmp_path)
    return server, TestClient(server.app)


def test_avatars_listed_with_the_default_first(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    names = c.get("/api/avatars").json()
    assert names[0] == "avatar.vrm"
    assert {"rocketbox_female_06.vrm", "rocketbox_male_15.vrm", "rocketbox_male_19.vrm"} <= set(names)


def test_page_and_static_assets(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    assert c.get("/").status_code == 200
    assert c.get("/static/avatar.js").status_code == 200
    assert c.get("/static/..%2Fserver.py").status_code == 404


def test_saved_answer_is_replayed_without_calling_the_pipeline(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    q = "What are the pillars of Islam?"
    run = tmp_path / server.question_id(q, "en")
    run.mkdir()
    report = {"status": "signed", "version": server.REPORT_VERSION, "stamp": server.stamp("en"), "answer": "Islam is built on five.",
              "glosses": ["ISLAM", "FIVE"]}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    np.save(run / "pose.npy", np.zeros((4, 50, 3), np.float32))
    monkeypatch.setattr(server, "pipeline", lambda lang: (_ for _ in ()).throw(AssertionError("pipeline must not run")))
    r = c.post("/api/ask", json={"question": q, "lang": "en"}).json()
    assert r["cached"] and r["glosses"] == ["ISLAM", "FIVE"] and r["id"] == run.name
    pose = c.get(f"/pose/{run.name}.json").json()
    assert pose["fps"] == 25 and len(pose["frames"]) == 4


def test_turkish_and_urdu_replay(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    for lang, q in (("tr", "Ramazan nedir?"), ("ur", "رمضان کیا ہے؟")):
        run = tmp_path / server.question_id(q, lang)
        run.mkdir()
        report = {"status": "signed", "version": server.REPORT_VERSION, "stamp": server.stamp(lang), "answer": "x",
                  "glosses": ["a"]}
        (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
        monkeypatch.setattr(server, "pipeline", lambda lang: (_ for _ in ()).throw(AssertionError("no pipeline")))
        r = c.post("/api/ask", json={"question": q, "lang": lang})
        assert r.status_code == 200 and r.json()["cached"]


def test_unknown_language_rejected(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    assert c.post("/api/ask", json={"question": "hi", "lang": "fr"}).status_code == 400


class _TextOnly:
    """A pipeline stand-in: signing a text must never call the answering step."""
    def __init__(self):
        self.signed = []

    def run(self, q):
        raise AssertionError("text to sign must not answer")

    def sign_text(self, text):
        self.signed.append(text)
        return {"status": "signed", "mode": "text", "answer": text, "sources": [], "glosses": ["PRAYER"], "id": "tenabc"}


def test_sign_text_skips_answering(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    fake = _TextOnly()
    monkeypatch.setattr(server, "pipeline", lambda lang: fake)
    r = c.post("/api/sign", json={"text": "  Prayer is light.  ", "lang": "en"})
    assert r.status_code == 200 and r.json()["mode"] == "text" and fake.signed == ["Prayer is light."]


def test_sign_text_limits_and_errors(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    assert c.post("/api/sign", json={"text": "", "lang": "en"}).status_code == 400
    assert c.post("/api/sign", json={"text": "x" * 1001, "lang": "en"}).status_code == 400
    assert c.post("/api/sign", json={"text": "hi", "lang": "fr"}).status_code == 400

    class NoSigns:
        def sign_text(self, text):
            raise ValueError("nothing in the lexicon to sign")
    monkeypatch.setattr(server, "pipeline", lambda lang: NoSigns())
    r = c.post("/api/sign", json={"text": "zzz qqq", "lang": "en"})
    assert r.status_code == 422 and "sign" in r.json()["detail"]


def test_signed_text_replays(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    text = "Prayer is light."
    run = tmp_path / server.question_id(text, "en", "text")
    run.mkdir()
    report = {"status": "signed", "mode": "text", "version": server.REPORT_VERSION, "stamp": server.stamp("en"),
              "answer": text, "glosses": ["PRAYER"]}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(server, "pipeline", lambda lang: (_ for _ in ()).throw(AssertionError("no pipeline")))
    r = c.post("/api/sign", json={"text": text, "lang": "en"})
    assert r.status_code == 200 and r.json()["cached"] and r.json()["mode"] == "text"


def test_stats_reflects_the_wslp_flag(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    stats = {"languages": [{"code": "en", "signs": 10, "glosses": 9},
                         {"code": "ur", "signs": 7, "glosses": 6, "signs_by_source": {"cislr": 3, "wslp": 4},
                          "glosses_by_source": {"cislr": 2, "wslp": 4}}],
             "licences": [], "created": "x"}
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(server, "STATS", f)
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    on = c.get("/api/stats").json()
    assert {l["code"]: l["signs"] for l in on["languages"]} == {"en": 10, "ur": 7} and on["avatars"] >= 6
    assert {l["code"]: l["glosses"] for l in on["languages"]} == {"en": 9, "ur": 6}
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    off = c.get("/api/stats").json()
    assert {l["code"]: l["signs"] for l in off["languages"]}["ur"] == 3
    assert {l["code"]: l["glosses"] for l in off["languages"]} == {"en": 9, "ur": 2}


def test_generated_stats_file_is_complete():
    from isharati.app import server
    s = json.loads(server.STATS.read_text(encoding="utf-8"))
    assert [l["code"] for l in s["languages"]] == ["en", "ar", "tr", "ur"]
    for l in s["languages"]:
        assert l["signs"] > 0 and l["glosses"] > 0 and l["passages"] > 0 and 0 < l["coverage"] < 1
    assert 0 < s["distinct_passages"] <= sum(l["passages"] for l in s["languages"])
    assert any("WSLP" in r["source"] for r in s["licences"])


def test_rule_glossed_text_is_not_replayed(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    text = "Prayer is light."
    run = tmp_path / server.question_id(text, "en", "text")
    run.mkdir()
    report = {"status": "signed", "version": server.REPORT_VERSION, "stamp": server.stamp("en"), "answer": text,
              "glosses": ["PRAYER"], "glosser": "rule", "mode": "text"}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    fake = _TextOnly()
    monkeypatch.setattr(server, "pipeline", lambda lang: fake)
    r = c.post("/api/sign", json={"text": text, "lang": "en"})
    assert r.status_code == 200 and fake.signed == [text] and not r.json().get("cached")


def _dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "clips").mkdir()
    (d / "index.html").write_text("<div id=root>react</div>", encoding="utf-8")
    (d / "record.html").write_text("recorder", encoding="utf-8")
    (d / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (d / "clips" / "clips.json").write_text("[]", encoding="utf-8")
    return d


def test_react_build_served_with_classic_fallback(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WEB_DIST", _dist(tmp_path))
    assert "react" in c.get("/").text
    assert "Isharati" in c.get("/classic").text
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/clips/clips.json").json() == []
    assert c.get("/record").text == "recorder"
    assert c.get("/assets/..%2F..%2Findex.html").status_code == 404
    assert c.get("/assets/missing.js").status_code == 404


def test_classic_page_when_not_built(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WEB_DIST", tmp_path / "nothing")
    assert "Isharati" in c.get("/").text


def test_ask_with_no_recorded_signs_is_a_422_not_a_500(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)

    class NoSigns:
        def run(self, q):
            raise ValueError("nothing in the lexicon to sign")
    monkeypatch.setattr(server, "pipeline", lambda lang: NoSigns())
    r = c.post("/api/ask", json={"question": "Kur'an nedir?", "lang": "tr"})
    assert r.status_code == 422 and "recorded sign" in r.json()["detail"]
