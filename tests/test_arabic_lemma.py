"""Arabic lemma matching (CAMeL analyses, meaning guard) and the stopword split."""
from pathlib import Path

import pytest

from isharati.text import arabic_morph as morph
from isharati.text.arabic import normalize_ar

LEX = Path("data/lexicon_v2/lexicon_qa.jsonl")


@pytest.fixture(scope="module")
def lex():
    if not LEX.exists() or not morph.NLP_PYTHON.exists():
        pytest.skip("Arabic lexicon or NLP environment not available")
    from isharati.lexicon.arabic import qa_lexicon
    return qa_lexicon(LEX)


def gloss(lex, w):
    e = lex.match_token(w)
    return normalize_ar(e.gloss) if e else None


@pytest.mark.parametrize("word,want", [
    ("يقول", None), ("قالت", "قال"), ("قالوا", "قال"), ("فقال", None), ("قلت", "قال"),
    ("كانوا", "كان"), ("يكون", "كان"), ("ربك", "رب"), ("ربنا", "رب"), ("يده", "يد"), ("بيده", "يد")])
def test_inflected_forms_find_the_lemma_sign(lex, word, want):
    # «يقول» and «فقال» have recorded signs of their own; the others reach the lemma's sign
    got = gloss(lex, word)
    if want is None:
        assert got is not None
    else:
        assert got == want


@pytest.mark.parametrize("word,never", [("وتكفر", "كفر"), ("إسلام", "سلام"), ("جنة", "جن"), ("ملك", "مل")])
def test_meaning_guards(lex, word, never):
    assert gloss(lex, word) != never


def test_stopwords_keep_questions_and_negation():
    for w in ("لماذا", "كيف", "متى", "أين", "ماذا", "هل", "لم", "لن", "لا"):
        assert normalize_ar(w) not in morph.STOPWORDS
    for w in ("عنه", "إلا", "به", "أنه", "منه", "عليهم", "إنما"):
        assert normalize_ar(w) in morph.STOPWORDS


def test_glosser_signs_questions_and_negation_drops_function_words():
    from isharati.glossing.arabic import ArabicQAGlosser
    drop = ArabicQAGlosser._dropped
    assert not drop("كيف", "كيف أصلي")
    assert not drop("لا", "لا تشرك بالله")
    assert not drop("هو", "قال")              # a pronoun sign placed by the model is kept
    assert not drop("له", "له الملك")          # preposition + pronoun -> هو
    assert drop("عن", "عن أبي هريرة")
    assert drop("من", "خرج من المسجد") and not drop("من", "من خلقك؟")
