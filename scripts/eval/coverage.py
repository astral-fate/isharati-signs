"""How much of the Qur'an and hadith vocabulary the sign lexicon covers, per language.

Corpus: the approved passages the demo answers from (English: Saheeh International + HadeethEnc; Arabic: the Qur'an
text + HadeethEnc). A word counts as covered if the lexicon matches it the way the glosser does (forms, lemmas,
prefixes). Reported, for the Qur'an, the hadith and both:
  types     distinct words covered / distinct words
  tokens    word occurrences covered / all occurrences (frequent words weigh more: what a reader actually meets)
both over all words and over content words only (function words such as "the", «في» are not signed).
The most frequent uncovered content words are the list of signs to collect next.

  .venv/Scripts/python scripts/eval/coverage.py en|ar
Output: results/coverage_<lang>.{json,md}
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
OUT = ROOT / "results"


def setup(lang):
    if lang == "en":
        from isharati.text.english import STOPWORDS, normalize_en
        from isharati.lexicon.asl import ASLLexicon
        lex = ASLLexicon.load(DATA / "asl" / "lexicon" / "lexicon_all.jsonl")
        words = lambda text: [w for w in normalize_en(text).split() if re.fullmatch(r"[a-z][a-z']*", w)]
        return DATA / "asl" / "corpus.jsonl", lex, words, STOPWORDS
    from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
    from isharati.text import arabic_morph as ar_morph  # noqa: E402
    from isharati.text.arabic import normalize_ar
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    words = lambda text: [w for w in normalize_ar(text).split() if re.fullmatch(r"[ء-ي]+", w)]
    return DATA / "ar" / "corpus.jsonl", lex, words, ar_morph.STOPWORDS


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    corpus, lex, words, stop = setup(lang)
    counts = {"quran": Counter(), "hadith": Counter()}
    capital = Counter()  # how often a word is written with a capital letter: a name almost always, a word only at sentence starts
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            counts[r["kind"]].update(words(r["text"]))
            if lang == "en":
                for m in re.finditer(r"\b([A-Z][\w’‘ʿʾ'-]*)", r["text"]):
                    capital.update(words(m.group(1)))
    counts["all"] = counts["quran"] + counts["hadith"]
    vocab = counts["all"]
    if lang == "ar":
        from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
        from isharati.text import arabic_morph as ar_morph  # noqa: E402
        ar_morph.lemmatize(list(vocab))  # one batch through the morphological analyser, then matching is cached
    covered = {}
    for k, w in enumerate(vocab):
        covered[w] = lex.match_token(w) is not None
        if k % 2000 == 0:
            print(f"{k}/{len(vocab)} words checked", flush=True)

    def stats(c, content):
        items = [(w, n) for w, n in c.items() if not content or w not in stop]
        types = len(items)
        tok = sum(n for _, n in items)
        cov_types = sum(1 for w, _ in items if covered[w])
        cov_tok = sum(n for w, n in items if covered[w])
        return {"types": types, "types_covered": cov_types, "type_coverage": round(cov_types / max(1, types), 4),
                "tokens": tok, "tokens_covered": cov_tok, "token_coverage": round(cov_tok / max(1, tok), 4)}

    # proper names (Abu Hurayrah, Ibn Umar): written capitalised most of the time; in ASL they are fingerspelled or
    # given name signs, so vocabulary coverage is also reported without them
    # the key terms are vocabulary even though they are capitalised
    terms = {"allah", "god", "quran", "muslim", "muslims", "islam", "prophet", "ramadan", "messenger", "lord", "hajj",
             "zakah", "kaabah", "mecca", "makkah", "madinah", "jesus", "moses", "abraham", "noah", "satan", "paradise",
             "hell", "friday", "sunnah", "jinn", "angel", "angels", "gabriel", "day", "resurrection", "judgment",
             "almighty", "exalted", "hereafter", "glorified", "majestic", "merciful", "gracious"}
    names = {w for w, n in counts["all"].items()
             if lang == "en" and capital[w] >= 0.9 * n and n >= 2 and w not in terms}
    if lang == "en":  # name particles, written lowercase inside names («Abdullah ibn Umar», «Abu Bakr»)
        names |= {"ibn", "bin", "bint", "abu", "umm", "al"} & set(counts["all"])
    report = {k: {"all_words": stats(c, False), "content_words": stats(c, True),
                  "vocabulary_no_names": stats(Counter({w: n for w, n in c.items() if w not in names}), True)}
              for k, c in counts.items()}
    report["names"] = len(names)
    missing = [(w, n) for w, n in vocab.most_common() if w not in stop and not covered[w] and w not in names]
    report["top_missing_names"] = [(w, n) for w, n in vocab.most_common() if w in names and not covered[w]][:100]
    report["top_missing"] = missing[:500]
    report["lexicon_glosses"] = len({e.gloss for e in lex.sign_entries()})
    OUT.mkdir(exist_ok=True)
    (OUT / f"coverage_{lang}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Qur'an and hadith vocabulary covered by the {'ASL' if lang == 'en' else 'Arabic'} sign lexicon\n",
          f"Lexicon: {report['lexicon_glosses']} glosses. Corpus: {corpus.relative_to(ROOT)}.\n",
          "| source | words | distinct | distinct covered | occurrences | occurrences covered |", "|---|---|---|---|---|---|"]
    for k in ("quran", "hadith", "all"):
        for kind in ("vocabulary_no_names", "content_words", "all_words"):
            s = report[k][kind]
            md.append(f"| {k} | {kind.replace('_', ' ')} | {s['types']:,} | {s['types_covered']:,} ({s['type_coverage']:.1%}) | "
                      f"{s['tokens']:,} | {s['tokens_covered']:,} ({s['token_coverage']:.1%}) |")
    md += ["", "## Most frequent content words with no sign yet", "",
           ", ".join(f"{w} ({n})" for w, n in missing[:150])]
    (OUT / f"coverage_{lang}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:11]))


if __name__ == "__main__":
    main()
