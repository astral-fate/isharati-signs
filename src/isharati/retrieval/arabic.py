"""Arabic retrieval and grounding settings, and the Arabic corpus: the Qur'an (IslamicEval 2025 Uthmani text) and
the HadeethEnc hadith in Arabic with their explanations. Answering is isharati.retrieval.engine.answer.

  python -m isharati.retrieval.arabic build
  python -m isharati.retrieval.arabic ask "ما هو رمضان؟"
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from isharati.config import DATA
from isharati.retrieval import engine as S
from isharati.text.arabic import candidates, normalize_ar
from isharati.text.arabic_morph import STOPWORDS
from isharati.text.levels import classify_level

CORPUS = DATA / "ar" / "corpus.jsonl"
CACHE = CORPUS.parent / "cache"
QURAN = DATA / "corpora" / "quranic_verses.json"


def _terms(text):
    out = []
    for w in normalize_ar(text).split():
        if w not in STOPWORDS and len(w) > 1:
            c = candidates(w)
            out.append(c[-1] if c else w)  # the most stripped stem, so «الصلوات» and «صلاة» meet more often
    return out


def _forms(text):
    return [set(candidates(w)) for w in normalize_ar(text).split() if w not in STOPWORDS and len(w) > 1]


FRAMING = frozenset(f for w in "المسلم المسلمون المسلمين الناس الإنسان يعني أيضا يسمى تسمى".split()
                    for f in candidates(w))

RULING = re.compile(r"(حلال|حرام|يجوز|جائز|يحل|مكروه|فتوى|فتوي|مباح|محرم|حكم)")

EXPAND_SYSTEM = (
    "Rewrite a question about Islam as a search query for a collection of Qur'an verses and hadith in Arabic. "
    "Return one line of 6 to 12 Arabic keywords (Modern Standard Arabic, no diacritics) that an authoritative "
    "answer would contain. No explanation."
)

ANSWER_SYSTEM = (
    "You answer questions about Islam for Deaf learners, in Modern Standard Arabic, and your answer will be signed "
    "in Arabic Sign Language. "
    "FIRST: if the question asks whether something is allowed or forbidden (حلال، حرام، يجوز)، asks for a religious "
    "ruling (فتوى) for someone's situation, or is about a disputed matter, reply exactly: REFER. "
    "Otherwise use ONLY the numbered passages given; never add facts that are not in them, even well-known ones. "
    "For 'ما هو X؟' build the answer on the passage that defines or describes X itself, not one that only mentions it. "
    "Passage [1] answers the question most directly: build the answer mainly on it. "
    "Answer the question directly in 2 to 4 complete, simple Arabic sentences of at most 12 words each, without "
    "diacritics. You may rephrase the hadith explanations, but never paraphrase a Qur'an verse: quote it word for "
    "word or do not use it. After each sentence put the passage numbers it comes from, like [1] or [2,3]. "
    "If the passages do not answer the question, reply exactly: NO_ANSWER."
)

AR = S.Lang(terms=_terms, forms=_forms, expand_system=EXPAND_SYSTEM, answer_system=ANSWER_SYSTEM,
            ruling=lambda q: bool(RULING.search(normalize_ar(q))) or classify_level(q) in ("C", "D"),
            framing=FRAMING, min_words=4,
            rewrite="These sentences use words the cited passages do not contain, so they may add facts: {sentences}"
                    "\nRewrite the whole answer in Arabic using only wording found in the cited passages, "
                    "same format.")


def build(workers: int = 8):
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = [{"pid": f"q{v['surah_id']}:{v['ayah_id']}", "kind": "quran",
             "reference": f"القرآن {v['surah_name']} {v['surah_id']}:{v['ayah_id']}", "text": v["ayah_text"],
             "url": f"https://quran.com/{v['surah_id']}/{v['ayah_id']}"}
            for v in json.loads(QURAN.read_text(encoding="utf-8"))]
    S.CACHE, en_cache = CACHE, S.CACHE
    try:
        ids = S.hadith_ids("ar")  # every hadith HadeethEnc has in Arabic
    finally:
        S.CACHE = en_cache

    S.CACHE, en_cache = CACHE, S.CACHE  # _cached_get writes to S.CACHE; point it at the Arabic cache meanwhile
    try:
        with ThreadPoolExecutor(workers) as pool:
            hadiths = list(pool.map(lambda hid: (hid, S._cached_get(
                f"hadith_{hid}", f"{S.HADEETH}/hadeeths/one/", {"language": "ar", "id": hid})), ids))
    finally:
        S.CACHE = en_cache
    for hid, h in hadiths:
        if not h or not h.get("hadeeth"):
            continue
        rows.append({"pid": f"h{hid}", "kind": "hadith",
                     "reference": f"موسوعة الأحاديث {hid} ({h.get('grade', 'الدرجة غير مذكورة')})",
                     "text": f"{h.get('title', '').strip()}. {h['hadeeth']} الشرح: {h.get('explanation', '')}".strip(" ."),
                     "url": f"https://hadeethenc.com/ar/browse/hadith/{hid}"})
    CORPUS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"{len(rows)} passages ({sum(r['kind'] == 'quran' for r in rows)} Qur'an, "
          f"{sum(r['kind'] == 'hadith' for r in rows)} hadith) -> {CORPUS}")


def corpus() -> S.Corpus:
    return S.Corpus(CORPUS, AR)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["ask"]:
        from isharati.llm import default_client
        print(json.dumps(S.answer(" ".join(sys.argv[2:]), corpus(), default_client()), ensure_ascii=False, indent=1))

