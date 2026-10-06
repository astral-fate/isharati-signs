"""Approved sources and grounded answers in Turkish and Urdu, like isharati.retrieval.engine (English) and qa_ar (Arabic).

Corpus: the Qur'an in an approved translation from QuranEnc (Turkish: Rowwad Translation Center; Urdu: Muhammad
Junagarhi) and the same graded HadeethEnc hadith as the other languages, with their titles and explanations; cached
as data/<lang>/corpus.jsonl. Answering, citation and the support check are isharati.retrieval.engine.answer.

  python -m isharati.retrieval.turkish_urdu build tr|ur
  python -m isharati.retrieval.turkish_urdu ask tr "Ramazan nedir?"
"""
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from isharati.retrieval import engine as S

from isharati.config import DATA
from isharati.text.turkish import tr_norm  # noqa: F401  (also used by isharati.lexicon.turkish)
EN_CACHE = DATA / "asl" / "cache"   # the English build's hadith ids, so every corpus holds the same hadith

# ---- Turkish: Turkish casing, and the first five letters as the stem (a standard, robust choice for Turkish search)
TR_STOP = set("ve ile bir bu şu o da de mi mı mu mü ne için gibi ama fakat ki çok daha en her olan olarak olan "
              "ise diye kadar sonra önce ya veya hem ben sen biz siz onlar onun bunun şey nedir nasıl neden "
              # the demonstrative / third-person pronouns inflected (zeyrek lemmas o, bu, şu; ~12k corpus tokens),
              # discourse words, the vocative, and «b» (bin, "son of", in narrator chains). Words that carry meaning
              # stay signed: question words (kim, ne) and negatives (hiç, hiçbir).
              "onu ona onlara onların onları ondan onlardan onunla onlarla onda onlarda odur "
              "bunu bundan buna bunlar bunların bunda bununla bunlardan bunları budur bunlara bunlarla "
              "şunu şunları şudur şunlardır şunun şuna şunlar şundan "
              "şöyle şöyledir şöyleydi böyle böylece böyledir eğer ey b "
              # postpositions, and the light verb kılmak («namaz kılmak», «haram kılmak»: the noun carries the meaning
              # and is signed; kılmak is not), in its corpus forms
              "üzerine üzere kılmak kılmaktır kıldığı kıldı kılan kılar kılardı kılınan kılmıştır kılarken kıldık "
              "kıldığını kılınmıştır kıldım kılsın kılındı kılması kılmış kılmayı kılıp kılma kılmaya kılıyordu "
              "kıldıktan kılarak kılarsa kıldılar kılındığı kılarsın kılınması kılınır kılınmasından kıldığımız kıldın "
              "kılmaktan kılabilir kılmadan kılarlar kılmanın kıldıklarını kılmalarını kılınacak kılmasını kılanın "
              "kılıyor kılın kılınız".split())


def tr_forms(word):
    return {word, word[:5]} if len(word) > 5 else {word}


# ---- Urdu: fold Arabic-script letter variants; words as written (Urdu marks case with separate postpositions)
UR_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ہ", "ة": "ہ", "أ": "ا", "إ": "ا", "آ": "ا"})
UR_STOP = set("اور کے کی کا کو میں سے پر نے ہے ہیں تھا تھی تھے یہ وہ جو کہ بھی تو ایک اس ان کیا کیوں کیسے "
              "ہو ہوں گا گی گے لیے لئے ساتھ تک جب اگر یا نہ نہیں".split())


def ur_norm(text):
    t = re.sub(r"[ً-ٰٟ]", "", unicodedata.normalize("NFC", text or "")).translate(UR_FOLD)
    return re.sub(r"[^\w]+", " ", t).strip()


def ur_forms(word):
    forms = {word}
    for suffix in ("وں", "یں", "ات", "ے"):  # plurals and oblique forms: نمازیں, نمازوں -> نماز
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            forms.add(word[: -len(suffix)])
    return forms


def _lang(norm, forms, stop, expand, answer, ruling, framing, rewrite):
    def words(text):
        return [w for w in norm(text).split() if w not in stop and len(w) > 1]
    return S.Lang(terms=lambda t: [min(forms(w), key=lambda f: (len(f), f)) for w in words(t)],
                  forms=lambda t: [forms(w) for w in words(t)],
                  expand_system=expand, answer_system=answer, ruling=ruling, framing=frozenset(framing),
                  rewrite=rewrite, min_words=5)


def _answer_system(language):
    return (
        f"You answer questions about Islam for Deaf learners, in {language}, and your answer will be signed. "
        "FIRST: if the question asks whether something is allowed or forbidden, asks for a religious ruling (fatwa) "
        "for someone's situation, or is about a disputed matter, reply exactly: REFER. "
        "Otherwise use ONLY the numbered passages given; never add facts that are not in them, even well-known ones. "
        "For 'what is X?' build the answer on the passage that defines or describes X itself. "
        "Passage [1] answers the question most directly: build the answer mainly on it. "
        f"Answer in 2 to 4 complete, simple {language} sentences of at most 12 words each. Never paraphrase a Qur'an "
        "verse: quote it word for word or do not use it. After each sentence put the passage numbers it comes from, "
        "like [1] or [2,3]. If the passages do not answer the question, reply exactly: NO_ANSWER."
    )


def _expand_system(language):
    return (f"Rewrite a question about Islam as a search query for Qur'an translations and hadith in {language}. "
            f"Return one line of 6 to 12 {language} keywords that an authoritative answer would contain. No explanation.")


REWRITE = ("These sentences use words the cited passages do not contain, so they may add facts: {sentences}\n"
           "Rewrite the whole answer using only wording found in the cited passages, same format.")

LANGS = {
    "tr": {"name": "Turkish", "quran": "turkish_rwwad", "hadith": "tr",
           "lang": _lang(tr_norm, tr_forms, TR_STOP, _expand_system("Turkish"), _answer_system("Turkish"),
                         lambda q: bool(re.search(r"\b(helal|haram|caiz|fetva|günah mı|mekruh)\b", tr_norm(q))),
                         {"müslüman", "müslümanlar", "insan", "insanlar", "her"}, REWRITE)},
    "ur": {"name": "Urdu", "quran": "urdu_junagarhi", "hadith": "ur",
           "lang": _lang(ur_norm, ur_forms, UR_STOP, _expand_system("Urdu"), _answer_system("Urdu"),
                         lambda q: bool(re.search(r"(حلال|حرام|جائز|فتوی|فتویٰ|مکروہ)", ur_norm(q))),
                         {"مسلمان", "لوگ", "انسان", "ہر"}, REWRITE)},
}


def corpus_path(lang):
    from isharati.config import DATA
    return DATA / lang / "corpus.jsonl"


def build(lang, workers=8):
    cfg = LANGS[lang]
    cache = corpus_path(lang).parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    S.CACHE, en_cache = cache, S.CACHE  # _cached_get writes to S.CACHE: point it at this language's cache meanwhile
    try:
        with ThreadPoolExecutor(workers) as pool:
            suras = list(pool.map(lambda s: S._cached_get(f"quran_{s}", S.QURANENC.replace("english_saheeh", cfg["quran"]).format(s)),
                                  range(1, 115)))
            ids = S.hadith_ids(cfg["hadith"])  # every hadith HadeethEnc has in this language
            hadiths = list(pool.map(lambda hid: (hid, S._cached_get(f"hadith_{hid}", f"{S.HADEETH}/hadeeths/one/",
                                                                    {"language": cfg["hadith"], "id": hid})), ids))
    finally:
        S.CACHE = en_cache
    rows = []
    for sura, data in enumerate(suras, 1):
        for a in (data or {}).get("result", []):
            rows.append({"pid": f"q{sura}:{a['aya']}", "kind": "quran", "reference": f"Qur'an {sura}:{a['aya']}",
                         "text": re.sub(r"\[\d+\]", "", a["translation"]).strip(),
                         "url": f"https://quranenc.com/en/browse/{cfg['quran']}/{sura}#{a['aya']}"})
    for hid, h in hadiths:
        if h and h.get("hadeeth"):
            rows.append({"pid": f"h{hid}", "kind": "hadith", "reference": f"HadeethEnc {hid} ({h.get('grade', '')})",
                         "text": f"{h.get('title', '').strip()}. {h['hadeeth']} {h.get('explanation', '')}".strip(" ."),
                         "url": f"https://hadeethenc.com/{cfg['hadith']}/browse/hadith/{hid}"})
    corpus_path(lang).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"{lang}: {len(rows)} passages ({sum(r['kind'] == 'quran' for r in rows)} Qur'an, "
          f"{sum(r['kind'] == 'hadith' for r in rows)} hadith) -> {corpus_path(lang)}")


def corpus(lang):
    return S.Corpus(corpus_path(lang), LANGS[lang]["lang"])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "ask":
        from isharati.llm import default_client
        print(json.dumps(S.answer(" ".join(sys.argv[3:]), corpus(sys.argv[2]), default_client()), ensure_ascii=False, indent=1))
