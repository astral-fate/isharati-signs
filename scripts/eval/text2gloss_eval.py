"""Arabic text -> ArSL gloss, evaluated against human gloss annotations, with baselines and confidence intervals.

Gold sets (each sentence: Arabic text and the gloss sequence a Deaf signer / annotator produced):
  isharah     Isharah-2000 SI annotations (Saudi ArSL; 1,038 distinct sentences): a fixed random sample of N_ISHARAH,
              none of them used in the glosser's prompt
  religious   Isharah's religious sentences (53)
  arabsign    ArabSign (Luqman, FG 2023): the 25 even-numbered sentences (the odd ones are the glosser's order examples)

Systems:
  copy        the Arabic text itself, word for word (how far plain text already is from the gloss)
  rule        the app's lexicon matcher, no language model: each word -> its sign's gloss, function words dropped
  llm         the app's glosser (isharati.glossing.arabic.ArabicQAGlosser, lexicon-constrained LLM)

Scoring is on words after orthographic folding (alef forms, ة/ه, ى/ي, a leading «ال», diacritics, «_» in multi-word
glosses), so a spelling variant of the same sign is not an error:
  WER          word edit distance / reference length (order-sensitive)
  P / R / F1   order-free: multiset overlap of output and reference words (sign languages reorder)
  BLEU-4, chrF corpus scores (sacrebleu), as reported for text-to-gloss on PHOENIX-2014T
  ROUGE-L      F-measure of the longest common subsequence
with 95% bootstrap intervals over sentences (1,000 resamples) for WER and F1.

Outputs are cached per system and sentence (results/text2gloss/), so a rerun only scores.
  py scripts/eval/text2gloss_eval.py [--n 300] [--workers 6]
"""
import json
import random
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402

ISHARAH = Path(r"D:\islam\signlang\isharah")
ARABSIGN_ZIP = Path(r"D:\islam\ArabSign\drive-download-20260930T221317Z-1-001.zip")
OUT = ROOT / "results" / "text2gloss"
SEED = 2026


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ---------------------------------------------------------------- gold sets

def isharah(n):
    rows = [l.rstrip("\n").split("|") for l in (ISHARAH / "SI_train.txt").open(encoding="utf-8")][1:]
    first = {}
    for _, gloss, text in rows:  # a sentence signed by several signers: its first annotation
        first.setdefault(text.strip(), gloss.strip())
    texts = sorted(first)
    random.Random(SEED).shuffle(texts)
    return [{"id": f"ish{i:04d}", "text": t, "gold": first[t]} for i, t in enumerate(texts[:n])]


def religious():
    out = []
    for i, line in enumerate((ISHARAH / "religious_sentences.tsv").read_text(encoding="utf-8").splitlines()[1:]):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            out.append({"id": f"rel{i:03d}", "text": parts[1].strip(), "gold": parts[0].strip()})
    return out


def arabsign():
    text = zipfile.ZipFile(ARABSIGN_ZIP).read("ArabSignGroundTruth.txt").decode("utf-16")
    out = []
    for line in text.splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3 and p[1].strip() and int(p[0]) % 2 == 0:
            out.append({"id": f"as{p[0].strip()}", "text": p[1].strip(), "gold": p[2].replace("-", " ").strip()})
    return out


# ---------------------------------------------------------------- scoring

def fold(s: str) -> list[str]:
    words = []
    for w in normalize_ar(strip_diacritics(s.replace("_", " "))).split():
        w = re.sub(r"[^\w]", "", w)
        if len(w) > 3 and w.startswith("ال") and w != "الله":  # the article, but «الله» is one word
            w = w[2:]
        if w:
            words.append(w)
    return words


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def lcs(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            d[i + 1][j + 1] = d[i][j] + 1 if a[i] == b[j] else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]


def per_sentence(ref, hyp):
    r, h = Counter(ref), Counter(hyp)
    common = sum((r & h).values())
    l = lcs(ref, hyp)
    p_l, r_l = (l / len(hyp) if hyp else 0), (l / len(ref) if ref else 0)
    return {"errors": edit(ref, hyp), "ref_len": len(ref), "hyp_len": len(hyp), "common": common,
            "rougeL": 2 * p_l * r_l / (p_l + r_l) if p_l + r_l else 0.0}


def corpus(rows, idx=None):
    rows = rows if idx is None else [rows[i] for i in idx]
    e = sum(x["s"]["errors"] for x in rows)
    n = sum(x["s"]["ref_len"] for x in rows)
    c, hl = sum(x["s"]["common"] for x in rows), sum(x["s"]["hyp_len"] for x in rows)
    p, r = (c / hl if hl else 0), (c / n if n else 0)
    return {"wer": e / max(1, n), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score(rows):
    import sacrebleu
    for x in rows:
        x["s"] = per_sentence(fold(x["gold"]), fold(x["output"]))
    refs = [" ".join(fold(x["gold"])) for x in rows]
    hyps = [" ".join(fold(x["output"])) for x in rows]
    base = corpus(rows)
    rng = np.random.default_rng(SEED)
    boots = [corpus(rows, rng.integers(0, len(rows), len(rows))) for _ in range(1000)]
    ci = {k: [round(float(np.percentile([b[k] for b in boots], q)), 3) for q in (2.5, 97.5)] for k in ("wer", "f1")}
    return {"sentences": len(rows), **{k: round(v, 3) for k, v in base.items()}, "wer_ci": ci["wer"], "f1_ci": ci["f1"],
            "bleu4": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score, 1),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 1),
            "rougeL": round(float(np.mean([x["s"]["rougeL"] for x in rows])), 3),
            "exact": sum(x["s"]["errors"] == 0 for x in rows)}


# ---------------------------------------------------------------- systems

def systems():
    from isharati.glossing.arabic import ArabicQAGlosser
    from isharati.lexicon import arabic as ar_lexicon
    from isharati.llm import default_client
    from isharati.text.arabic_morph import STOPWORDS
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    glosser = ArabicQAGlosser(lex, default_client())

    def rule(text):
        out = []
        for w in normalize_ar(strip_diacritics(text)).split():
            if w in STOPWORDS:
                continue
            e = lex.match_token(w)
            out.append(e.gloss if e else w)
        return " ".join(out)

    def llm(text):
        return " ".join(i.text for i in glosser.gloss(text).items)

    return {"copy": lambda t: t, "rule": rule, "llm": llm}


def run(name, fn, items, workers):
    cache = OUT / f"{name}.jsonl"
    done = {}
    if cache.exists():
        done = {r["text"]: r["output"] for r in map(json.loads, cache.open(encoding="utf-8"))}
    todo = [x for x in items if x["text"] not in done]

    def one(x):
        try:
            return x["text"], fn(x["text"])
        except Exception as e:  # noqa: BLE001 - a failed call counts as an empty output, and is retried next run
            print(f"  {name} failed: {x['id']}: {e!r}"[:200], flush=True)
            return x["text"], None

    with ThreadPoolExecutor(workers if name == "llm" else 1) as pool, cache.open("a", encoding="utf-8") as f:
        for k, (text, out) in enumerate(pool.map(one, todo), 1):
            if out is not None:
                done[text] = out
                f.write(json.dumps({"text": text, "output": out}, ensure_ascii=False) + "\n")
                f.flush()
            if k % 25 == 0:
                print(f"  {name}: {k}/{len(todo)}", flush=True)
    return [{**x, "output": done.get(x["text"], "")} for x in items]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"isharah": isharah(arg("--n", 300)), "religious": religious(), "arabsign": arabsign()}
    allitems = [x for s in sets.values() for x in s]
    report = {}
    for name, fn in systems().items():
        print(f"{name}: {len(allitems)} sentences", flush=True)
        outs = {x["id"]: x for x in run(name, fn, allitems, arg("--workers", 6))}
        for sname, items in sets.items():
            report.setdefault(sname, {})[name] = score([dict(outs[x["id"]]) for x in items])
        examples = [outs[x["id"]] for x in sets["isharah"][:12]]
        report.setdefault("_examples", {})[name] = [{"text": x["text"], "gold": x["gold"], "output": x["output"]}
                                                     for x in examples]
    (ROOT / "results" / "text2gloss_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    md = ["# Arabic text -> ArSL gloss against human annotations", "",
          "Scored on folded words (spelling variants are not errors). WER is order-sensitive; P/R/F1 are order-free.", ""]
    for sname in sets:
        md += [f"## {sname} ({len(sets[sname])} sentences)", "",
               "| system | WER [95% CI] | precision | recall | F1 [95% CI] | BLEU-4 | chrF | ROUGE-L | exact |",
               "|---|---|---|---|---|---|---|---|---|"]
        for name, s in report[sname].items():
            md.append(f"| {name} | {s['wer']:.3f} [{s['wer_ci'][0]:.3f}-{s['wer_ci'][1]:.3f}] | {s['precision']:.3f} | "
                      f"{s['recall']:.3f} | {s['f1']:.3f} [{s['f1_ci'][0]:.3f}-{s['f1_ci'][1]:.3f}] | {s['bleu4']} | "
                      f"{s['chrf']} | {s['rougeL']:.3f} | {s['exact']} |")
        md.append("")
    md += ["## Examples (isharah)", "", "| text | gold | rule | llm |", "|---|---|---|---|"]
    for i, x in enumerate(report["_examples"]["llm"]):
        md.append(f"| {x['text']} | {x['gold']} | {report['_examples']['rule'][i]['output']} | {x['output']} |")
    (ROOT / "results" / "text2gloss_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
