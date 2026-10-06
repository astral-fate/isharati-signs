"""Turkish (TİD) and Urdu (ISL) lexicons, glossers and back-translation sampling."""
import json

import numpy as np
import pytest


def _write(root, folder, file, rows):
    d = root / folder
    (d / "signs").mkdir(parents=True, exist_ok=True)
    out = []
    for i, (gloss, sid) in enumerate(rows):
        np.save(d / "signs" / f"{sid}.npy", np.full((6, 50, 3), i, np.float32))
        out.append({"gloss": gloss, "sign_id": sid, "dataset": folder, "keypoints_path": f"{folder}/signs/{sid}.npy",
                    "review_status": "pending", "is_religious": False, "is_letter": False})
    (d / file).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")


@pytest.fixture
def tid_root(tmp_path):
    _write(tmp_path, "dictionaries", "entries_tid_dictionary.jsonl",
           [("namaz", "n1"), ("oruç", "o1"), ("iman", "i1"), ("ışık", "k1"), ("abdest", "a1")])
    return tmp_path


@pytest.fixture
def isl_root(tmp_path):
    _write(tmp_path, "cislr", "entries.jsonl", [("PRAYER", "c1"), ("BOOK", "c2")])
    _write(tmp_path, "wslp", "entries.jsonl", [("PRAYER", "w1"), ("MOSQUE", "w2")])
    return tmp_path


def test_tr_norm_turkish_casing():
    from isharati.text.turkish import tr_norm
    assert tr_norm("İMAN") == "iman" and tr_norm("IŞIK") == "ışık" and tr_norm("Namaz!") == "namaz"


def test_tid_lexicon_matches_turkish_case_and_suffixes(tid_root):
    from isharati.lexicon.turkish import tid_lexicon
    lex = tid_lexicon(tid_root)
    assert lex.match_token("İMAN").sign_id == "i1"
    assert lex.match_token("IŞIK").sign_id == "k1"
    assert lex.match_token("namazı").sign_id == "n1"      # suffix: first five letters
    assert lex.match_token("abdesti").sign_id == "a1"
    assert lex.lookup("Oruç").sign_id == "o1"
    assert lex.match_token("kitap") is None
    assert lex.keypoints(lex.lookup("namaz")).shape == (6, 50, 3)


def test_isl_lexicon_wslp_flag(isl_root, monkeypatch):
    from isharati.lexicon.isl import isl_lexicon, wslp_enabled
    on = isl_lexicon(isl_root, wslp=True)
    assert on.lookup("PRAYER").sign_id == "c1"             # CISLR wins a tie
    assert on.lookup("MOSQUE").sign_id == "w2"
    off = isl_lexicon(isl_root, wslp=False)
    assert off.lookup("MOSQUE") is None and len(off.sign_entries()) == 2
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    assert not wslp_enabled()
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    assert wslp_enabled()
    monkeypatch.delenv("ISHARATI_URDU_WSLP")
    assert wslp_enabled()                                   # on by default while the Space is private


class Reply:
    """A fake LLM client: records the messages and returns a fixed reply."""
    def __init__(self, raw):
        self.raw, self.seen = raw, []

    def chat(self, msgs):
        self.seen.append(msgs)
        return self.raw


def test_turkish_glosser_uses_tid_vocabulary(tid_root):
    from isharati.glossing.llm import LLMGlosser, TID_SPEC
    from isharati.lexicon.turkish import tid_lexicon
    c = Reply('{"glosses": [{"text": "NAMAZ", "oov": false}, {"text": "İman", "oov": false},'
              ' {"text": "kitap", "oov": false}]}')
    r = LLMGlosser(tid_lexicon(tid_root), c, TID_SPEC).gloss("Namaz imanın direğidir.")
    assert r.glosses == ["namaz", "iman"] and r.oov == ["kitap"]   # the lexicon decides, not the model
    system, user = c.seen[0][0]["content"], c.seen[0][1]["content"]
    assert "Turkish Sign Language" in system and "namaz" in user and "ışık" in user


def test_urdu_glosser_uses_english_isl_glosses(isl_root):
    from isharati.glossing.llm import ISL_SPEC, LLMGlosser
    from isharati.lexicon.isl import isl_lexicon
    c = Reply('{"glosses": [{"text": "PRAYER", "oov": false}, {"text": "جماعت", "oov": true}]}')
    r = LLMGlosser(isl_lexicon(isl_root, wslp=False), c, ISL_SPEC).gloss("نماز جماعت کے ساتھ")
    assert r.glosses == ["PRAYER"] and r.oov == ["جماعت"]
    assert "Indian Sign Language" in c.seen[0][0]["content"]
    assert "mosque" not in c.seen[0][1]["content"].lower()           # WSLP off: its signs are never offered


def test_asl_glosser_unchanged_alias():
    from isharati.glossing import asl, llm
    assert asl.parse_glosses is llm.parse_glosses and asl.SYSTEM == llm.ASL_SPEC.system
    assert issubclass(asl.ASLLLMGlosser, llm.LLMGlosser)


def test_sampled_recognizer_keeps_answer_signs(lexicon):
    from isharati.backtranslate import SampledRecognizer
    rec = SampledRecognizer(lexicon, n_distractors=2).for_glosses(["صلاة", "غير موجود"])
    labels = [g for g, _ in rec.templates]
    assert "صلاة" in labels and len(labels) <= 3


def test_stamp_and_ids_cover_new_languages(monkeypatch, tmp_path):
    from isharati import pipeline
    assert pipeline.LANGS == ("en", "ar", "tr", "ur")
    assert pipeline.question_id("Ramazan nedir?", "tr").startswith("tr")
    assert pipeline.question_id("رمضان کیا ہے؟", "ur").startswith("ur")
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    on = pipeline.stamp("ur")
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    assert pipeline.stamp("ur") != on                     # turning WSLP off invalidates saved Urdu answers


def test_fallback_glosser_uses_rules_when_the_llm_fails():
    from isharati.glossing.llm import FallbackGlosser
    from isharati.llm import GlossError
    from isharati.types import GlossItem, GlossResult

    class Down:
        def gloss(self, text):
            raise GlossError("all LLM backends failed")

    class Rules:
        def gloss(self, text):
            return GlossResult([GlossItem("PRAYER")], "rule")

    r = FallbackGlosser(Down(), Rules()).gloss("Prayer is light.")
    assert r.glosses == ["PRAYER"] and r.backend == "rule"


def test_text_ids_never_collide_with_answer_ids():
    from isharati.pipeline import question_id
    q = "What is Ramadan?"
    assert question_id(q, "en") != question_id(q, "en", "text")
    assert question_id(q, "en", "text").startswith("ten") and question_id(q, "en", "text").isalnum()
    assert question_id(q, "en") == question_id(q)                  # earlier answer ids still replay
