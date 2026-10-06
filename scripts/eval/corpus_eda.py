"""Exploratory analysis of the four Qur'an/hadith corpora against the sign lexicons.

For each language: corpus size; the most frequent content words of the Qur'an and of the hadith; the lexicon's
coverage of word occurrences (Turkish: TİD dictionary; Urdu: the ISL lexicons, whose glosses are English words, so they
are measured on the English corpus of the same verses and hadith); for English and Arabic, how much of the covered text
each lexicon source contributes (the source of the sign that the matcher returns) and the signs used most often.

  .venv/Scripts/python scripts/eval/corpus_eda.py
Output: results/eda.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
OUT = ROOT / "results" / "eda.json"


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def counts(rows, words):
    c = {"quran": Counter(), "hadith": Counter()}
    for r in rows:
        c[r["kind"]].update(words(r["text"]))
    c["all"] = c["quran"] + c["hadith"]
    return c


def summary(c, stop, covered=None, top=25, covered_tokens=None):
    """covered(w): a word type is covered; covered_tokens[kind][w], if given: how many occurrences of w were covered in
    context (Turkish: a word inside a multi-word gloss is covered there even if it has no sign of its own)."""
    out = {}
    for k in ("quran", "hadith", "all"):
        content = {w: n for w, n in c[k].items() if w not in stop}
        s = {"tokens": sum(c[k].values()), "types": len(c[k]), "content_tokens": sum(content.values()),
             "content_types": len(content),
             "top_content": Counter(content).most_common(top)}
        if covered is not None:
            cov = (sum(min(n, covered_tokens[k][w]) for w, n in content.items()) if covered_tokens is not None
                   else sum(n for w, n in content.items() if covered(w)))
            s["content_token_coverage"] = round(cov / max(1, s["content_tokens"]), 4)
            s["content_type_coverage"] = round(sum(1 for w in content if covered(w)) / max(1, len(content)), 4)
            if covered_tokens is not None:  # occurrences left unsigned in context, most first
                left = Counter({w: n - min(n, covered_tokens[k][w]) for w, n in content.items()})
                s["top_uncovered"] = [(w, n) for w, n in left.most_common(top) if n > 0]
            else:
                s["top_uncovered"] = [(w, n) for w, n in Counter(content).most_common() if not covered(w)][:top]
        out[k] = s
    return out


def attribution(c, stop, match):
    """Content-word occurrences covered, by the source dataset of the matched sign; and the signs used most."""
    by_src, by_gloss, total, used = Counter(), Counter(), 0, defaultdict(set)
    for w, n in c["all"].items():
        if w in stop:
            continue
        total += n
        e = match(w)
        if e is not None:
            by_src[e.dataset] += n
            by_gloss[e.gloss] += n
            used[e.dataset].add(e.gloss)
    return {"content_tokens": total, "by_source": {k: round(v / total, 4) for k, v in by_src.most_common()},
            "glosses_used_by_source": {k: len(v) for k, v in used.items()},
            "top_signs": by_gloss.most_common(30)}


def main():
    res = {}
    # English / ASL
    from isharati.text.english import STOPWORDS, normalize_en, candidates
    from isharati.lexicon.asl import ASLLexicon
    en_rows = load(DATA / "asl" / "corpus.jsonl")
    en_words = lambda t: [w for w in normalize_en(t).split() if re.fullmatch(r"[a-z][a-z']*", w)]
    ce = counts(en_rows, en_words)
    lex = ASLLexicon.load(DATA / "asl" / "lexicon" / "lexicon_all.jsonl")
    memo = {}
    def en_match(w):
        if w not in memo:
            memo[w] = lex.match_token(w)
        return memo[w]
    res["en"] = {"passages": len(en_rows), **summary(ce, STOPWORDS, lambda w: en_match(w) is not None),
                 "attribution": attribution(ce, STOPWORDS, en_match)}
    print("en done", flush=True)

    # Urdu mode / ISL: English glosses, measured on the English corpus
    isl = set()
    # every word source the Urdu lexicon loads (islrtc: scraped dictionary; psl: Pakistan Sign Language, FESF dictionary)
    for f in ("wslp", "cislr", "islrtc", "psl"):
        path = DATA / "lexicon_isl" / f / "entries.jsonl"
        for e in (load(path) if path.exists() else []):
            if e.get("is_letter") or e.get("variant_of"):  # variants: other recordings
                continue
            g = e["gloss"].lower().strip()
            if " " not in g:
                isl.add(g)
    isl_cov = lambda w: any(x in isl for x in candidates(w))
    res["isl_on_en"] = {"glosses": len(isl), **{k: v for k, v in summary(ce, STOPWORDS, isl_cov).items()}}
    print("isl done", flush=True)

    # Arabic / ArSL
    from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
    from isharati.text import arabic_morph as ar_morph  # noqa: E402
    from isharati.text.arabic import normalize_ar
    ar_rows = load(DATA / "ar" / "corpus.jsonl")
    # «الشرح:» heads the commentary of 3,574 hadith passages: a section label, not a word of the text
    # words as spelled (hamza and ة kept, ar_morph.spelling): the analyser reads «المؤمنين» and «رأيت», not their
    # folded forms; stopwords and phrase signs are compared folded
    ar_words = lambda t: [w for w in ar_morph.spelling(re.sub(r"الشرح\s*:", " ", t)).split()
                          if re.fullmatch(r"[ء-ي]+", w) and normalize_ar(w)]
    ca = counts(ar_rows, ar_words)
    alex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    ar_morph.lemmatize(list(ca["all"]), analysed=True)  # the whole vocabulary in one analyser run
    amemo = {}
    def ar_match(w):
        if w not in amemo:
            amemo[w] = alex.match_token(w)
        return amemo[w]
    # what the app does with a word that has no sign: a function word the analyser reads only as a particle,
    # preposition, conjunction or relative («فإنه», «لأنه», «بذلك») is dropped, as ArSL grammar drops it (the glosser
    # is told so); such words are not content words. A function word with a sign («مع») and pronouns stay.
    def _pos(w):
        top = ar_morph.top_analyses(w)
        return top[0]["pos"] if top else None
    AR_FUNCTION = {w for w in ca["all"] if _pos(w) in ar_morph.FUNCTION_POS - {"pron"} and ar_match(w) is None}
    AR_STOP = {w for w in ca["all"] if normalize_ar(w) in ar_morph.STOPWORDS} | AR_FUNCTION
    # a proper name with no sign («عباس», «جبريل») is fingerspelled, as ArSL shows names: reported apart
    AR_NAMES = {w for w in ca["all"] if _pos(w) == "noun_prop" and ar_match(w) is None and w not in AR_STOP}
    # as for Turkish, a word is covered in context when it is signed by a multi-word sign the text contains word for
    # word («صلى الله عليه وسلم», «عز وجل», «أبو هريرة»): the glosser uses those phrase signs (phrases_in), and «وسلم»
    # has no sign of its own. A phrase word matches the text word with or without a leading «و/ف/ب/ل».
    phrases = {}
    for e in alex.sign_entries():
        pw = normalize_ar(e.gloss).split()
        if len(pw) >= 2:
            phrases.setdefault(pw[0], []).append(pw)
    CASES = {"ابي": "ابو", "ابا": "ابو"}  # «أبي هريرة», «أبا بكر»: the phrase sign «أبو ...» in another case
    def same(tok, w):
        """The text word is the phrase word, with or without a leading «و/ف/ب/ل», in any case of «أبو»."""
        t = normalize_ar(tok)
        forms = [t] + ([t[1:]] if t[:1] in "وفبل" else [])
        return any(x == w or CASES.get(x) == w for x in forms)
    acov = {"quran": Counter(), "hadith": Counter()}
    for r in ar_rows:
        ws = ar_words(r["text"])
        hit = [ar_match(w) is not None for w in ws]
        for i, w in enumerate(ws):
            n = normalize_ar(w)
            n = CASES.get(n, n)
            for pw in phrases.get(n, []) + (phrases.get(n[1:], []) if n[:1] in "وفبل" else []) +                     (phrases.get(CASES[n[1:]], []) if n[:1] in "وفبل" and n[1:] in CASES else []):
                if i + len(pw) <= len(ws) and all(same(ws[i + k], pw[k]) for k in range(len(pw))):
                    hit[i:i + len(pw)] = [True] * len(pw)
        acov[r["kind"]].update(w for w, h in zip(ws, hit) if h)
    acov["all"] = acov["quran"] + acov["hadith"]
    res["ar"] = {"passages": len(ar_rows), **summary(ca, AR_STOP, lambda w: ar_match(w) is not None,
                                                     covered_tokens=acov),
                 "attribution": attribution(ca, AR_STOP, ar_match),
                 "unsigned_function_words_dropped": sum(ca["all"][w] for w in AR_FUNCTION)}
    # signs, plus proper names with no sign shown by fingerspelling (in context, as above)
    for k in ("quran", "hadith", "all"):
        content = {w: n for w, n in ca[k].items() if w not in AR_STOP}
        shown = sum(min(n, acov[k][w]) if w not in AR_NAMES else n for w, n in content.items())
        res["ar"][k]["coverage_with_fingerspelled_names"] = round(shown / max(1, sum(content.values())), 4)
    # every content word still unsigned in context, most frequent first: the worklist for signs and reviewed synonyms
    left = Counter({w: n - min(n, acov["all"][w]) for w, n in ca["all"].items() if w not in AR_STOP})
    (OUT.parent / "ar_uncovered.json").write_text(json.dumps([[w, n] for w, n in left.most_common() if n > 0],
                                                             ensure_ascii=False), encoding="utf-8")
    print("ar done", flush=True)

    # Turkish / TİD: the app's own matcher (TIDLexicon.match_tokens: exact, lemma via zeyrek, reviewed synonyms, a
    # stem fallback only for words the analyser does not know; multi-word glosses first), on the lexicon the app loads
    # (tid_lexicon: dictionary, re-cut pairs, names, tid_youtube, AUTSL, cuneyt, extra sources). scripts/eval/tr_matcher_eval.py compares it
    # with the old five-letter rule and samples its precision.
    from isharati.retrieval import turkish_urdu as M
    from isharati.lexicon.turkish import tid_lexicon
    from isharati.text import turkish_morph
    tr_rows = load(DATA / "tr" / "corpus.jsonl")
    tr_words = lambda t: [w for w in M.tr_norm(t).split() if w.isalpha()]
    ct = counts(tr_rows, tr_words)
    tlex = tid_lexicon()
    turkish_morph.analyze(list(ct["all"]))  # fills the lemma cache once
    tcov = {"quran": Counter(), "hadith": Counter()}
    for r in tr_rows:
        ws = tr_words(r["text"])
        for s0, e0, _, _ in tlex.match_tokens(ws):
            tcov[r["kind"]].update(ws[s0:e0])
    tcov["all"] = tcov["quran"] + tcov["hadith"]
    tmemo = {}
    def tr_match(w):
        if w not in tmemo:
            tmemo[w] = tlex.match_token(w)
        return tmemo[w]
    res["tr"] = {"passages": len(tr_rows), "glosses": len(tlex.sign_entries()),
                 **summary(ct, M.TR_STOP, lambda w: tr_match(w) is not None, covered_tokens=tcov),
                 "attribution": attribution(ct, M.TR_STOP, tr_match)}
    print("tr done", flush=True)

    # Urdu corpus: size and frequent words (no Urdu-script sign lexicon)
    ur_rows = load(DATA / "ur" / "corpus.jsonl")
    cu = counts(ur_rows, lambda t: [w for w in M.ur_norm(t).split() if not w.isdigit()])
    res["ur"] = {"passages": len(ur_rows), **summary(cu, M.UR_STOP)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("en", "ar", "tr"):
        a = res[k]["all"]
        print(k, a["content_tokens"], a.get("content_token_coverage"))
    print("isl", res["isl_on_en"]["all"]["content_token_coverage"])


if __name__ == "__main__":
    main()
