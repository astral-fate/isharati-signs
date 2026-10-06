"""Approved English sources and grounded answers.

Corpus: the Qur'an in the Saheeh International translation (QuranEnc) and graded hadith with explanations
(HadeethEnc), cached as data/asl/corpus.jsonl. An answer is written ONLY from retrieved passages, in short sentences
for signing, with every passage cited; questions asking for a ruling or on disputed matters are referred.

  python -m isharati.retrieval.engine build     # download and cache the corpus (a few minutes)
  python -m isharati.retrieval.engine ask "Tell me about Islam"
"""
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from isharati.text.english import STOPWORDS, candidates, normalize_en

from isharati.config import DATA  # noqa: E402
CORPUS = DATA / "asl" / "corpus.jsonl"
QURANENC = "https://quranenc.com/api/v1/translation/sura/english_saheeh/{}"
HADEETH = "https://hadeethenc.com/api/v1"
# every root category: Qur'an and its sciences, hadith sciences, creed, jurisprudence, virtues and manners, da'wah,
# and seerah and history (with only creed, jurisprudence and virtues, eclipse prayers or the seerah had no passage)
HADITH_CATEGORIES = (1, 2, 3, 4, 5, 6, 7)


class Referral(Exception):
    """The question needs a qualified scholar; it is not answered or signed."""


@dataclass
class Passage:
    pid: str
    kind: str        # "quran" | "hadith"
    reference: str   # "Qur'an 112:1" | "HadeethEnc 5907 (Sahih)"
    text: str
    url: str


CACHE = CORPUS.parent / "cache"


def _cached_get(name, url, params=None):
    """GET with a file cache, so an interrupted build resumes where it stopped."""
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            data = requests.get(url, params=params, timeout=60).json()
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return None


def hadith_ids(lang: str) -> list:
    """Every HadeethEnc hadith available in a language, over all root categories (list pages are cached)."""
    ids = []
    for cat in HADITH_CATEGORIES:
        page = 1
        while True:
            name = f"list_{cat}_{page}" if lang == "en" else f"list_{lang}_{cat}_{page}"
            r = _cached_get(name, f"{HADEETH}/hadeeths/list/",
                            {"language": lang, "category_id": cat, "page": page, "per_page": 100})
            if not r or not r.get("data"):
                break
            ids += [h["id"] for h in r["data"]]
            if page >= int((r.get("meta") or {}).get("last_page") or page):
                break
            page += 1
    return list(dict.fromkeys(ids))


def build(workers: int = 8):
    from concurrent.futures import ThreadPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        suras = list(pool.map(lambda s: _cached_get(f"quran_{s}", QURANENC.format(s)), range(1, 115)))
    rows = []
    for sura, data in enumerate(suras, 1):
        for a in (data or {}).get("result", []):
            text = re.sub(r"\[\d+\]", "", a["translation"]).strip()
            rows.append({"pid": f"q{sura}:{a['aya']}", "kind": "quran", "reference": f"Qur'an {sura}:{a['aya']}",
                         "text": text, "url": f"https://quranenc.com/en/browse/english_saheeh/{sura}#{a['aya']}"})
    ids = hadith_ids("en")
    with ThreadPoolExecutor(workers) as pool:
        hadiths = list(pool.map(lambda hid: (hid, _cached_get(f"hadith_{hid}", f"{HADEETH}/hadeeths/one/",
                                                              {"language": "en", "id": hid})), dict.fromkeys(ids)))
    for hid, h in hadiths:
        if not h or not h.get("hadeeth"):
            continue
        # the title is HadeethEnc's one-line summary: the best words for retrieval
        text = f"{h.get('title', '').strip()}. {h['hadeeth']} Explanation: {h.get('explanation', '')}".strip(" .")
        rows.append({"pid": f"h{hid}", "kind": "hadith",
                     "reference": f"HadeethEnc {hid} ({h.get('grade', 'grade not stated')})",
                     "text": text, "url": f"https://hadeethenc.com/en/browse/hadith/{hid}"})
    CORPUS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"{len(rows)} passages ({sum(r['kind'] == 'quran' for r in rows)} Qur'an, "
          f"{sum(r['kind'] == 'hadith' for r in rows)} hadith) -> {CORPUS}")


def _terms(text):
    out = []
    for w in normalize_en(text).split():
        if w not in STOPWORDS and len(w) > 2:
            out.append(candidates(w)[-1])  # the shortest lemma guess, so "prayers" and "pray" meet
    return out


def _forms(text):
    """Each content word with all its lemma guesses ("fasting" -> fasting, fast, faste), for the support check."""
    return [set(candidates(w)) for w in normalize_en(text).split() if w not in STOPWORDS and len(w) > 2]


@dataclass
class Lang:
    """What differs between answer languages: tokenizing, prompts, the ruling pre-check, framing words."""
    terms: object          # text -> BM25 terms
    forms: object          # text -> [set of lemma guesses per content word]
    expand_system: str
    answer_system: str
    ruling: object         # question -> True if it asks for a ruling
    framing: frozenset
    rewrite: str           # asks for a rewrite; {sentences} is filled in
    min_words: int = 6     # a shorter single-sentence answer is a bare phrase, not an answer


EMBED_URL = "http://127.0.0.1:8766/embed"   # scripts/corpora/embeddings.py serve (multilingual-e5-large)
RRF_K = 60                                  # reciprocal-rank fusion constant


EMBED_MODEL = "intfloat/multilingual-e5-large"


def embed(texts, timeout=20):
    """Question vectors (unit length), or None when no embedder answers (retrieval then falls back to BM25).
    ISHARATI_EMBED=hf uses Hugging Face hosted inference of the same model the passages were embedded with (identical
    vectors, checked); otherwise the local embedding service."""
    import os
    try:
        if os.environ.get("ISHARATI_EMBED") == "hf":
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=os.environ.get("HF_TOKEN"), timeout=timeout)
            v = np.stack([np.asarray(client.feature_extraction("query: " + str(t)[:1500], model=EMBED_MODEL),
                                     np.float32).reshape(-1) for t in texts])
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        r = requests.post(EMBED_URL, json={"texts": list(texts)}, timeout=timeout)
        return np.asarray(r.json()["vectors"], np.float32)
    except Exception:
        return None


class Corpus:
    """Hybrid retrieval over the approved passages: BM25 (exact terms and names) fused with dense embeddings
    (meaning: «الخسوف» finds «كسوف», paraphrases match), by reciprocal rank. BM25 alone when no embeddings exist."""

    def __init__(self, path: Path = CORPUS, lang: "Lang | None" = None):
        self.lang = lang or EN
        path = Path(path)
        self.passages = [Passage(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines()]
        self.docs = [Counter(self.lang.terms(p.text)) for p in self.passages]
        self.avg = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.vectors = None
        ids = path.parent / "embeddings_ids.json"
        if (path.parent / "embeddings.npy").exists() and ids.exists():
            if json.loads(ids.read_text(encoding="utf-8")) == [p.pid for p in self.passages]:  # same corpus build
                self.vectors = np.load(path.parent / "embeddings.npy").astype(np.float32)

    HADITH_BOOST = 1.3  # hadith come with an explanation written as an answer; verses are often fragments

    def bm25(self, query: str, k: int) -> list[int]:
        q = self.lang.terms(query)
        scores = []
        for i, d in enumerate(self.docs):
            L = sum(d.values())
            s = sum(self.idf.get(t, 0) * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * L / self.avg)) for t in q if t in d)
            if s > 0:
                scores.append((s * (self.HADITH_BOOST if self.passages[i].kind == "hadith" else 1), i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]

    def dense(self, question: str, k: int) -> list[int]:
        if self.vectors is None:
            return []
        q = embed([question])
        if q is None or not len(q):
            return []
        sims = self.vectors @ q[0]
        return list(np.argsort(-sims)[:k])

    def search(self, query: str, k: int = 6, question: str | None = None) -> list[Passage]:
        """query: the expanded keywords (for BM25); question: the question as asked (for the embeddings)."""
        lists = [self.bm25(query, 100), self.dense(question or query, 100)]
        fused = Counter()
        for ranking in lists:
            for rank, i in enumerate(ranking):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        return [self.passages[i] for i, _ in fused.most_common(k)]


EXPAND_SYSTEM = (
    "Rewrite a question about Islam as a search query for a collection of Qur'an translations and hadith. "
    "Return one line of 6 to 12 English keywords and short phrases that an authoritative answer would contain "
    "(for example: five daily prayers salah obligatory). No explanation."
)


def expand(question: str, client, system: str = EXPAND_SYSTEM) -> str:
    """The question plus the words an answer would use, so keyword retrieval finds the defining passages."""
    try:
        reply = client.chat([{"role": "system", "content": system}, {"role": "user", "content": question}]).strip()
        line = reply.splitlines()[0] if reply else ""  # models sometimes add notes after the keywords
        if re.search(r"[؀-ۿ]", question):   # Arabic question: keep the Arabic keywords only
            line = " ".join(re.findall(r"[؀-ۿ]+", line))
        return question + " " + " ".join(line.split()[:14])
    except Exception:
        return question


ANSWER_SYSTEM = (
    "You answer questions about Islam for Deaf learners, and your answer will be signed in ASL. "
    "FIRST: if the question asks whether something is allowed or forbidden (halal, haram, permissible), asks for "
    "a religious ruling (fatwa) for someone's situation, or is about a disputed matter, reply exactly: REFER. "
    "Otherwise use ONLY the numbered passages given; never add facts that are not in them, even well-known ones "
    "(such as which month of the calendar something is). For 'what is X?', build the answer on the passage that "
    "defines or describes X itself, not one that only mentions it. "
    "Passage [1] answers the question most directly: build the answer mainly on it. "
    "Answer the question directly in 2 to 4 complete sentences, each with a subject and a verb and at most 12 words "
    "(for 'how many prayers a day?': 'Muslims pray five prayers every day and night.'). You may rephrase, but every "
    "fact must be stated in the passages. Never paraphrase a Qur'an verse: quote it word for word or use the hadith "
    "explanations. Never reply with a bare phrase or a list. After each sentence put the "
    "passage numbers it comes from, like [1] or [2,3]. If the passages do not answer the question, reply exactly: "
    "NO_ANSWER."
)


RULING = re.compile(r"\b(halal|haram|makruh|fatwa|permissible|permitted|forbidden|prohibited|allowed|sinful|"
                    r"is it (?:ok|okay|wrong|a sin)|can (?:i|muslims|a muslim|we)|may (?:i|muslims|a muslim))\b", re.I)


def answer(question: str, corpus: Corpus, client, k: int = 10) -> dict:
    lang = getattr(corpus, "lang", EN)
    if lang.ruling(question):  # checked before any model call, so a ruling is never generated
        raise Referral("questions about rulings go to a qualified scholar")
    query = expand(question, client, lang.expand_system)
    passages = corpus.search(query, k, question=question)
    if not passages:
        raise LookupError("no approved passage matches the question")
    passages = _best_first(question, passages, client)
    numbered = "\n".join(f"[{i + 1}] ({p.reference}) {p.text[:700]}" for i, p in enumerate(passages))
    msgs = [{"role": "system", "content": lang.answer_system},
            {"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"}]
    for attempt in range(2):
        raw = client.chat(msgs).strip()
        if raw.startswith("REFER"):
            raise Referral("this question needs a qualified scholar")
        if raw.startswith("NO_ANSWER"):
            raise LookupError("the approved passages do not answer this question")
        kept, dropped, used = [], [], set()
        for sentence, refs in _cited_sentences(raw):
            cited = [passages[i - 1] for i in refs if 1 <= i <= len(passages)]
            support = _support(sentence, cited, lang)
            if cited and all(p.kind == "quran" for p in cited) and not _quotes(sentence, cited, lang):
                # a paraphrased verse can change who or what it is about (41:7 is about idolaters, not every
                # Muslim who withholds zakah), so a sentence resting on verses alone must quote one exactly
                dropped.append({"sentence": sentence, "support": round(support, 2), "why": "paraphrases a verse"})
            elif cited and support >= MIN_SUPPORT:
                kept.append(sentence)
                used.update(p.pid for p in cited)
            else:
                dropped.append({"sentence": sentence, "support": round(support, 2)})
        words = sum(len(s.split()) for s in kept)
        if (not dropped and words >= lang.min_words) or len(kept) >= 2:
            break
        if not dropped:  # a bare phrase such as "The five (daily) prayers." is not an answer
            dropped = [{"sentence": s, "support": None, "why": "too short to answer"} for s in kept]
            kept, used = [], set()
        # one rewrite: name the unsupported sentences so the model restates them in the passages' own words
        msgs = msgs[:2] + [{"role": "assistant", "content": raw}, {"role": "user", "content":
                lang.rewrite.format(sentences=" | ".join(d["sentence"] for d in dropped))}]
    if not kept:
        raise LookupError("no sentence of the answer is supported by the passages it cites")
    sources = [p for p in passages if p.pid in used]
    return {"answer": " ".join(kept), "sources": [p.__dict__ for p in sources], "query": query,
            "dropped_unsupported": dropped, "retrieved": [p.reference for p in passages]}


PICK_SYSTEM = (
    "You pick sources. Given a question and numbered passages, reply with the number of the ONE passage that answers "
    "the question most directly: for 'what is X?' the passage that defines or describes X itself (for example the "
    "hadith of the five pillars for 'what is Islam?'), not one that only mentions X. Reply with the number only."
)


def _best_first(question, passages, client):
    """The passages with the one that answers most directly first; the answer is then built mainly on [1]."""
    listing = "\n".join(f"[{i + 1}] {p.text[:300]}" for i, p in enumerate(passages))
    try:
        reply = client.chat([{"role": "system", "content": PICK_SYSTEM},
                             {"role": "user", "content": f"Question: {question}\n\nPassages:\n{listing}"}])
        k = int(re.search(r"\d+", reply).group()) - 1
    except Exception:  # no usable pick: keep the retrieval order
        return passages
    return [passages[k]] + passages[:k] + passages[k + 1:] if 0 <= k < len(passages) else passages


# words that frame an answer without adding a fact ("Muslims pray five times"); they need not be in the passage
FRAMING = {"muslim", "muslims", "time", "times", "people", "person", "believer", "believers", "every", "each",
           "called", "call", "mean", "means", "also", "many", "way", "thing"}
MIN_SUPPORT = 0.8  # share of a sentence's content words that must occur in the passages it cites


def _cited_sentences(raw):
    """'Sentence one [1]. Sentence two [2,3].' -> [(sentence, [refs])]."""
    out = []
    text = re.sub(r"\((?:Qur.an|HadeethEnc)[^)]*\)", "", raw.replace("\n", " "))  # inline references, not content
    uncited = re.sub(r"(.+?)\s*\[([\d,\s]+)\]\s*\.?", "", text).strip(" .,;")
    if len(re.findall(r"[^\W\d_]{2,}", uncited)) >= 3:
        out.append((uncited + ".", []))  # no citation: kept visible so it is dropped as unsupported, not lost
    for m in re.finditer(r"(.+?)\s*\[([\d,\s]+)\]\s*\.?", text):
        sentence = re.sub(r"\s+", " ", m.group(1)).strip(" .,;")
        refs = [int(n) for n in re.split(r"[,\s]+", m.group(2)) if n]
        if len(re.findall(r"[^\W\d_]{2,}", sentence)) >= 3:  # a fragment like "(and" or "or" is not a sentence
            out.append((sentence[0].upper() + sentence[1:] + ".", refs))
    return out


def _key(s):
    return len(s), s  # the shortest lemma guess, ties broken the same way every time


def _quotes(sentence, cited, lang):
    """True if the sentence (after an introduction such as "Allah says:") is word for word part of a cited verse."""
    body = sentence.split(":", 1)[-1]
    words = [w for f in lang.forms(body) for w in [min(f, key=_key)]]
    if not words:
        return True
    for p in cited:
        verse = [min(f, key=_key) for f in lang.forms(p.text)]
        if any(verse[i:i + len(words)] == words for i in range(len(verse) - len(words) + 1)):
            return True
    return False


def _support(sentence, cited, lang=None):
    """Share of the sentence's content words (as lemma guesses) that appear in its cited passages. A number word
    such as "ninth" must appear too, so an added fact is caught."""
    lang = lang or EN
    words = [f for f in lang.forms(sentence) if not f & lang.framing]
    if not words:
        return 1.0
    vocab = set().union(*(f for p in cited for f in lang.forms(p.text)))
    return sum(1 for f in words if f & vocab) / len(words)


EN = Lang(terms=_terms, forms=_forms, expand_system=EXPAND_SYSTEM, answer_system=ANSWER_SYSTEM,
          ruling=lambda q: bool(RULING.search(q)), framing=frozenset(FRAMING),
          rewrite="These sentences use words the cited passages do not contain, so they may add facts: {sentences}"
                  "\nRewrite the whole answer using only wording found in the cited passages, same format.")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["ask"]:
        from isharati.llm import default_client
        print(json.dumps(answer(" ".join(sys.argv[2:]), Corpus(), default_client()), ensure_ascii=False, indent=1))
