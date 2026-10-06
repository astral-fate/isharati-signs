"""Arabic text->gloss against the ArabSign reference (Luqman, FG 2023): 50 Arabic sentences, each with the ArSL
sentence as signed (sign order, e.g. «لاتشرك بالله احد» -> «لا شرك الله احد»).

Both sides are mapped through the same sign lexicon, so the comparison is between signs, not spellings: a reference
word becomes its sign's gloss, or stays a word if no sign exists. Reported per sentence and overall:
  sign WER      word edit distance between the reference and the output sign sequences / reference length
                (phrase signs split into their words, spelling variants folded)
  sign recall   share of reference words the output contains
  coverage      share of reference words that have a sign in the lexicon (a ceiling for the glosser)
Output: results/arabsign_gloss_eval.{json,md}
  .venv/Scripts/python scripts/eval/arabsign_gloss.py [--rescore]
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.glossing import arabic as ar_glossing  # noqa: E402
from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402
from isharati.llm import default_client  # noqa: E402

ZIP = Path(r"D:\islam\ArabSign\drive-download-20260930T221317Z-1-001.zip")
OUT = ROOT / "results"


def reference():
    text = zipfile.ZipFile(ZIP).read("ArabSignGroundTruth.txt").decode("utf-16")
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1].strip():
            rows.append({"id": parts[0].strip(), "arabic": parts[1].strip(), "signs": parts[2].replace("-", " ").split()})
    return rows


def to_signs(words, lex):
    """Words -> sign glosses where the lexicon has one (phrase signs are matched first), else the word itself."""
    out, i = [], 0
    while i < len(words):
        for n in (3, 2):  # a phrase sign such as «الله اكبر» or «يوم القيامة»
            if i + n <= len(words) and (e := lex.lookup(" ".join(words[i:i + n]))) is not None:
                out.append(("sign", e.gloss))
                i += n
                break
        else:
            e = lex.match_token(words[i])
            out.append(("sign", e.gloss) if e else ("word", normalize_ar(words[i])))
            i += 1
    return out


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def words_of(glosses):
    """A gloss sequence as normalised words: phrase signs split, spelling variants folded (ة/ه, أ/ا)."""
    return [w for g in glosses for w in normalize_ar(g).split()]


def splits(results):
    """The glosser's order examples come from the odd-numbered sentences, so the even-numbered ones are the held-out
    test set; both halves and the whole are reported."""
    return {"test (even, held out)": score([x for x in results if int(x["id"]) % 2 == 0]),
            "dev (odd, examples)": score([x for x in results if int(x["id"]) % 2 == 1]),
            "all": score(results)}


def score(results):
    for x in results:
        ref, hyp = words_of(x["reference"]), words_of(x["output"])
        x["wer"] = edit(ref, hyp) / max(1, len(ref))
        x["recall"] = sum(1 for w in ref if w in hyp) / max(1, len(ref))
    n_ref = sum(len(words_of(x["reference"])) for x in results)
    return {"sentences": len(results),
            "sign_wer": round(sum(edit(words_of(x["reference"]), words_of(x["output"])) for x in results) / n_ref, 3),
            "sign_recall": round(sum(x["recall"] for x in results) / len(results), 3),
            "coverage": round(sum(x["coverage"] for x in results) / len(results), 3),
            "exact_match": sum(x["wer"] == 0 for x in results)}


def main():
    if "--rescore" in sys.argv:  # recompute the metrics from the saved outputs, without calling the model again
        data = json.loads((OUT / "arabsign_gloss_eval.json").read_text(encoding="utf-8"))
        results = data["sentences"]
        summary = splits(results)
        write(summary, results)
        print(json.dumps(summary, ensure_ascii=False))
        return
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    glosser = ar_glossing.ArabicQAGlosser(lex, default_client())
    rows, results = reference(), []
    for r in rows:
        ref = to_signs(r["signs"], lex)
        try:
            res = glosser.gloss(r["arabic"])
            hyp = [("sign", i.text) if not i.oov else ("word", normalize_ar(i.text)) for i in res.items]
        except Exception as e:  # a failed call counts as an empty output
            hyp, _ = [], print(r["id"], "failed:", e)
        ref_g, hyp_g = [g for _, g in ref], [g for _, g in hyp]
        ref_signs = [g for k, g in ref if k == "sign"]
        results.append({"id": r["id"], "arabic": r["arabic"], "reference": ref_g, "output": hyp_g,
                        "wer": edit(ref_g, hyp_g) / max(1, len(ref_g)),
                        "recall": sum(1 for g in ref_signs if g in hyp_g) / len(ref_signs) if ref_signs else None,
                        "coverage": len(ref_signs) / max(1, len(ref))})
        print(r["id"], f"WER {results[-1]['wer']:.2f}", "|", " ".join(ref_g), "->", " ".join(hyp_g), flush=True)
    summary = splits(results)
    write(summary, results)
    print(json.dumps(summary, ensure_ascii=False))


def write(summary, results):
    OUT.mkdir(exist_ok=True)
    (OUT / "arabsign_gloss_eval.json").write_text(json.dumps({"summary": summary, "sentences": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Arabic text->gloss vs ArabSign reference\n", "| split | sentences | sign WER | sign recall | coverage | exact |",
          "|---|---|---|---|---|---|"]
    md += [f"| {k} | {s['sentences']} | {s['sign_wer']:.2f} | {s['sign_recall']:.0%} | {s['coverage']:.0%} | {s['exact_match']} |"
           for k, s in summary.items()]
    md += ["", "| id | Arabic | reference signs | our signs | WER |", "|---|---|---|---|---|"]
    md += [f"| {x['id']} | {x['arabic']} | {' '.join(x['reference'])} | {' '.join(x['output'])} | {x['wer']:.2f} |" for x in results]
    (OUT / "arabsign_gloss_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
