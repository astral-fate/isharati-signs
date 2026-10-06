"""Turkish word -> TİD sign matching: zeyrek lemmas, reviewed synonyms, multi-word glosses, and the stem fallback
limited to words the analyser does not know."""
import json

import numpy as np
import pytest

pytest.importorskip("zeyrek")  # the analyser (or a lemma cache that already holds these words)


def _write(root, folder, file, glosses):
    d = root / folder
    (d / "signs").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(glosses):
        sid = f"{folder[:2]}{i}"
        np.save(d / "signs" / f"{sid}.npy", np.full((6, 50, 3), i, np.float32))
        rows.append({"gloss": g, "sign_id": sid, "dataset": folder, "keypoints_path": f"{folder}/signs/{sid}.npy",
                     "review_status": "pending", "is_religious": False, "is_letter": False})
    (d / file).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


@pytest.fixture
def lex(tmp_path):
    from isharati.lexicon.turkish import tid_lexicon
    _write(tmp_path, "dictionaries", "entries_tid_dictionary.jsonl",
           ["kitap", "gelmek", "namaz", "namaz kılmak", "peygamber", "renk", "ağız", "oğul", "iman", "insan", "kul",
            "hayvan", "inanmak", "mescid"])
    syn = tmp_path / "synonyms_tr.json"
    syn.write_text(json.dumps({"_comment": "test", "synonyms": {"resul": "peygamber", "mescit": "mescid",
                                                                "kitap": "iman", "zekât": "sadaka"}},
                              ensure_ascii=False), encoding="utf-8")
    lexicon = tid_lexicon(tmp_path)
    lexicon.load_synonyms(syn)
    return lexicon


def test_lemma_undoes_consonant_alternation_and_vowel_drop(lex):
    assert lex.match("kitabı") == (lex.lookup("kitap"), "lemma")
    assert lex.match_token("rengi").gloss == "renk"
    assert lex.match_token("ağzı").gloss == "ağız"
    assert lex.match_token("oğlu").gloss == "oğul"
    assert lex.match_token("namazı").gloss == "namaz"
    assert lex.match_token("namazlarını").gloss == "namaz"


def test_verb_matches_its_infinitive_gloss(lex):
    assert lex.match("geldi") == (lex.lookup("gelmek"), "lemma")
    assert lex.match_token("inandılar").gloss == "inanmak"


def test_meaning_changing_readings_never_match(lex):
    assert lex.match_token("inanmadı") is None   # negation: «did not believe» is not «believe»
    assert lex.match_token("imansız") is None    # «-sız»: without faith
    assert lex.match_token("kulluk") is None     # «-lık»: servitude, not «servant»


def test_reviewed_synonym_and_own_sign_wins(lex):
    assert lex.match("resulü") == (lex.lookup("peygamber"), "synonym")   # via its lemma «resul»
    assert lex.match_token("mescitte").gloss == "mescid"   # «mescit» (TDK spelling), inflected
    assert lex.lookup("mescit").gloss == "mescid"
    assert lex.match_token("kitap").gloss == "kitap"   # a word with its own sign keeps it over a listed synonym
    assert lex.match_token("zekât") is None            # a synonym whose target is not signed does nothing


def test_multi_word_gloss_preferred_over_its_words(lex):
    spans = lex.match_tokens("namazı kıldı ve kitabı".split())
    assert [(s, e, x.gloss) for s, e, x, _ in spans] == [(0, 2, "namaz kılmak"), (3, 4, "kitap")]
    assert lex.match_token("namaz kılmak").gloss == "namaz kılmak"
    assert lex.match_token("namazı kılmak").gloss == "namaz kılmak"


def test_false_five_letter_prefix_rejected(lex):
    # the old rule matched any word sharing a gloss's first five letters: «insanlık» (humanity) -> insan,
    # «hayvancık» -> hayvan; the analyser knows both words, so no stem guess is made
    assert lex.match_token("insanlık") is None
    assert lex.match_token("hayvancık") is None
    assert lex.match_token("insanlar").gloss == "insan"   # plain inflection still matches


def test_prefix_fallback_only_for_unknown_words(lex):
    from isharati.text import turkish_morph
    w = "peygamberimizinkiler"  # not in zeyrek's lexicon as one word? then the fallback may sign it
    if turkish_morph.readings(w):
        pytest.skip("the analyser knows this form")
    assert lex.match(w) == (lex.lookup("peygamber"), "prefix")
    lex.prefix_fallback = False
    assert lex.match_token(w) is None
