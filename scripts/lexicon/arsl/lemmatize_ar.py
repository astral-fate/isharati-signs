"""Possible lemmas of Arabic words, with CAMeL Tools' morphological analyser (run with the NLP venv, which has it).

Reads a JSON list of words on stdin, prints {word: {"l": [lemma, ...], "v": only readable as a verb, "a": analyses}}.
  "l": undiacritised noun/adjective lemmas, most frequent reading first. Words that can only be read as verbs get no
       "l": undiacritised, a verb and a noun can share letters with different meanings («وتكفر», expiates, vs «كفر»,
       disbelief). Used by isharati.lexicon.arabic for nouns and adjectives back to the singular («الخطايا» -> خطيئة).
  "a": the distinct analyses, most probable first (pos_lex_logprob), as [pos, lex, logprob, enc0, per, num, gen]:
       the diacritised lemma («كَفَّر» vs «كَفَر» keeps the senses apart), the enclitic pronoun («2ms_poss» in «ربك»,
       «2ms_dobj» in «فتجيبك») and the verb's person/number/gender. isharati.text.arabic_morph.analyses reads them for
       lemma matching with a meaning guard (a verb form finds a sign only through the same verb lemma).
  D:\\islam\\models\\.venv-nlp\\Scripts\\python scripts/lexicon/arsl/lemmatize_ar.py < words.json
"""
import json
import os
import sys

os.environ.setdefault("CAMELTOOLS_DATA", r"D:\islam\models\camel_data")
from camel_tools.morphology.analyzer import Analyzer  # noqa: E402
from camel_tools.morphology.database import MorphologyDB  # noqa: E402
from camel_tools.utils.dediac import dediac_ar  # noqa: E402

NOMINAL = ("noun", "adj", "noun_quant", "adj_comp", "noun_num")
MAX_ANALYSES = 12


def lemmas_of(analyses):
    """The old "l"/"v" fields, unchanged."""
    scored = {}
    nominal = [a for a in analyses if a.get("pos") in NOMINAL]
    # a word that can only be read as a verb is left alone: «وتكفر» (expiates) must not become «كفر» (disbelief);
    # «وحج» can also be the noun «الحج», so it keeps its noun readings
    if any(a.get("pos") == "verb" for a in analyses) and not nominal:
        return {"l": [], "v": True}
    for a in analyses:
        if a.get("pos") not in ("noun", "adj", "noun_quant", "adj_comp"):
            continue  # singulars of nouns and adjectives only: those keep their meaning
        lex = dediac_ar(a.get("lex", "")).replace("_", " ").strip()
        lex = "".join(ch for ch in lex if not ch.isdigit())
        if lex:
            scored[lex] = max(scored.get(lex, -99.0), float(a.get("pos_logprob", -99.0)))
    return {"l": sorted(scored, key=scored.get, reverse=True)[:5], "v": False}


def compact(analyses):
    best = {}
    for a in analyses:
        key = (a.get("pos", ""), a.get("lex", ""), a.get("enc0", "0"), a.get("per", "na"), a.get("num", "na"),
               a.get("gen", "na"))
        lp = round(float(a.get("pos_lex_logprob", -99.0) or -99.0), 3)
        best[key] = max(best.get(key, -99.0), lp)
    rows = sorted(best.items(), key=lambda kv: -kv[1])[:MAX_ANALYSES]
    return [[k[0], k[1], lp, k[2], k[3], k[4], k[5]] for k, lp in rows]


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    words = json.loads(sys.stdin.read() or "[]")
    analyzer = Analyzer(MorphologyDB.builtin_db(), backoff="NOAN_PROP")
    out = {}
    for w in words:
        analyses = analyzer.analyze(w)
        out[w] = {**lemmas_of(analyses), "a": compact(analyses)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
