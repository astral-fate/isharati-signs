"""ASL path: English normalisation, lexicon lookup, glossing and stitching on a tiny synthetic lexicon."""
import json

import numpy as np
import pytest

from isharati.text.english import candidates, normalize_en
from isharati.glossing.asl import ASLLLMGlosser, ASLRuleGlosser
from isharati.lexicon.asl import ASLLexicon, gloss_key
from isharati.pose.stitch import StitchPoser
from isharati.lexicon.arabic import MissingLetterSign
from isharati.types import N_JOINTS


def test_normalize_and_candidates():
    assert normalize_en("Allah's Prophet!") == "allah prophet"
    assert "pray" in candidates("prayed") and "prayer" in candidates("prayers")
    assert "be" in candidates("is") and "person" in candidates("people")
    assert "stop" in candidates("stopping")


def test_gloss_key_drops_variant_digits():
    assert gloss_key("EAT1") == "eat" and gloss_key("THANK_YOU") == "thank you"


@pytest.fixture
def lex(tmp_path):
    rows = []
    for i, g in enumerate(["GOD", "PRAY1", "PRAY2", "DAY", "FIVE", "PEOPLE", "a", "l", "h"]):
        np.save(tmp_path / f"s{i}.npy", np.full((5, N_JOINTS, 3), i, np.float32))
        rows.append({"gloss": g, "sign_id": f"s{i}", "dataset": "t", "keypoints_path": f"s{i}.npy",
                     "review_status": "pending" if g == "GOD" else "approved", "is_religious": g == "GOD",
                     "is_letter": len(g) == 1})
    (tmp_path / "lexicon.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return ASLLexicon.load(tmp_path / "lexicon.jsonl")


def test_lookup_and_variants(lex):
    assert lex.lookup("pray").gloss == "PRAY1"          # first variant wins
    assert lex.match_token("praying").gloss == "PRAY1"
    assert lex.match_token("days").gloss == "DAY"
    assert [e.gloss for e in lex.fingerspell("Allah")] == ["a", "l", "l", "a", "h"]
    with pytest.raises(MissingLetterSign):
        lex.fingerspell("Zakat")


def test_rule_glosser_drops_function_words_and_fingerspells(lex):
    r = ASLRuleGlosser(lex).gloss("People pray five times a day to God, Allah.")
    assert r.glosses == ["PEOPLE", "PRAY1", "FIVE", "DAY", "GOD"]
    assert r.oov == ["times", "Allah"]


def test_llm_glosser_lexicon_decides(lex):
    class Fake:
        def chat(self, msgs):
            return '{"glosses": [{"text": "PEOPLE", "oov": false}, {"text": "praying", "oov": true}, ' \
                   '{"text": "Allah", "oov": false}]}'
    r = ASLLLMGlosser(lex, Fake()).gloss("People pray to Allah")
    assert r.glosses == ["PEOPLE", "PRAY1"] and r.oov == ["Allah"]


def test_stitch_with_fingerspelling(lex):
    r = ASLRuleGlosser(lex).gloss("pray Allah")
    pose, segs = StitchPoser(lex, n_transition=2).pose(r)
    kinds = [s.kind for s in segs if s.kind != "transition"]
    assert kinds == ["sign"] + ["fingerspell"] * 5
    assert pose.shape[1:] == (N_JOINTS, 3) and len(pose) == 6 * 5 + 5 * 2


def test_missing_signs_are_marked_not_fingerspelled(lex):
    from isharati.pose.poser import MissingAwarePoser, MISSING_FRAMES
    r = ASLRuleGlosser(lex).gloss("People pray to Allah")  # "Allah" has no sign
    pose, segs = MissingAwarePoser(lex, n_transition=2).pose(r)
    named = [(s.label, s.kind) for s in segs if s.kind != "transition"]
    assert named == [("PEOPLE", "sign"), ("PRAY1", "sign"), ("Allah", "missing")]
    missing = [s for s in segs if s.kind == "missing"][0]
    assert missing.end - missing.start == MISSING_FRAMES
    # the video starts and ends with the hands down, and the segments cover every frame
    assert segs[0].kind == segs[-1].kind == "transition" and segs[-1].end == len(pose)
    assert pose[-1][7, 1] > pose[-1][5, 1] + 1.5  # left wrist well below the left shoulder at the end


def test_parse_glosses_tolerates_broken_json():
    from isharati.glossing.asl import parse_glosses
    broken = '{"glosses": [{"text": "GOD", "oov": false} {"text": "Islam", "oov": true}, {"text": "pray'
    assert parse_glosses(broken) == [{"text": "GOD", "oov": False}, {"text": "Islam", "oov": True}]


def test_support_rejects_added_facts():
    from isharati.retrieval.engine import MIN_SUPPORT, Passage, _cited_sentences, _support
    p = Passage("h1", "hadith", "x", "Whoever fasts Ramadan out of faith will have his past sins forgiven.", "")
    assert _support("Fasting Ramadan forgives past sins.", [p]) >= MIN_SUPPORT
    assert _support("Ramadan is the ninth month of the Islamic calendar.", [p]) < MIN_SUPPORT
    assert _cited_sentences("Allah is One [1]. Fasting is a shield [1,2].") == [("Allah is One.", [1]), ("Fasting is a shield.", [1, 2])]


def test_rulings_referred_and_fragments_dropped():
    import pytest
    from isharati.retrieval.engine import Referral, _cited_sentences, answer
    with pytest.raises(Referral):
        answer("Is music haram?", corpus=None, client=None)
    raw = "The Book is from Allah (Qur'an 40:2) the Wise [1]. ,. (and [2]. or [3]."
    assert _cited_sentences(raw) == [("The Book is from Allah the Wise.", [1])]


def test_bare_phrase_answer_is_retried():
    from isharati.retrieval.engine import Corpus, Passage, answer
    p = Passage("h1", "hadith", "HadeethEnc 1", "The five prayers are performed during the day and night.", "")
    corpus = Corpus.__new__(Corpus)
    corpus.search = lambda q, k, **kw: [p]
    replies = iter(["prayers", "The five (daily) prayers. [1]", "The five prayers are performed day and night. [1]"])

    class Client:
        def chat(self, messages):
            return next(replies)
    assert answer("How many prayers a day?", corpus, Client())["answer"] == "The five prayers are performed day and night."


def test_proportions_give_one_body_shape():
    import numpy as np
    from isharati.pose import proportions as P
    clip = np.zeros((5, 50, 3), np.float32)
    clip[:, 2, 0], clip[:, 5, 0] = -0.5, 0.5          # shoulders one unit apart, neck at the origin
    clip[:, 0, 1] = -0.28                              # a squashed head, as in Isharah's pixel data
    clip[:, 29, 1] = 0.5                               # a hand below the neck
    fixed = P.fix(clip)
    assert abs(P.ratio(fixed) - P.REF) < 1e-6
    assert abs(fixed[0, 29, 1] - 0.5 * P.REF / 0.28) < 1e-5  # everything scaled about the neck alike
    assert np.allclose(fixed[..., 0], clip[..., 0])            # widths untouched


def test_arabic_qa_normalisation():
    import pytest
    from pathlib import Path
    lex_path = Path("data/lexicon_v2/lexicon_qa.jsonl")
    if not lex_path.exists():
        pytest.skip("Arabic demo lexicon not built")
    from isharati.glossing import arabic as ar_glossing  # noqa: E402
    from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
    lex = ar_lexicon.qa_lexicon(lex_path)
    assert lex.match_token("خمسة").gloss == "5"                 # number words -> digit signs
    assert lex.match_token("إسلام") is None or lex.match_token("إسلام").gloss != "سلام"  # not «peace»
    g = ar_glossing.ArabicQAGlosser(lex, None)
    assert "محمد رسول الله" in g.phrases_in("لا إله إلا الله وأن محمدًا عبده ورسوله")
    assert "آلة الكمان" not in g.phrases_in("وتحج البيت إن استطعت إليه سبيلا")  # «ال» stems are not words
    assert lex.match_token("الصلوات").gloss == "الصلاة"         # a plural finds its singular sign
    tk = lex.match_token("تكفر")                                   # «expiates»: never the sign «كفر» (disbelief)
    assert tk is None or tk.gloss != "كفر"
    assert lex.match_token("وحج").gloss in ("الحج", "حج")                # a noun reading keeps its sign
    sh = lex.match_token("شرك")                                    # polytheism: never the sign «شر» (evil)
    assert sh is None or sh.gloss != "شر"
