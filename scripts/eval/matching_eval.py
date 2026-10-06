"""Evaluation of what Isharati does with Arabic text, against Deaf signers' annotations (Isharah, ArabSign).

The glosses a signer writes also hold choices no text-to-sign system makes from the text alone (an added «انا», the
question markers «سوال» / «استفهام», a sign for a word that is not in the sentence). These are left out: a signer's
sign is kept only when it comes from a word of the sentence, i.e. shares its root (CAMeL Tools): «يعملون» -> «عمل»,
«بالبرد» -> «برد», «اشعر» -> «شعور». What is measured is then what the system is built to do.

  Test 1, word -> sign. Each (sentence word, signer's sign) pair whose sign is in Isharati's lexicon: does the
          matcher send the word to that sign? Three levels of the same matcher:
            exact     the word as written (spelling folded)
            affixes   + prefixes and suffixes stripped (و، ب، ال، ـه، plurals ...), no morphological analyser
            full      + CAMeL lemmas and verb readings, reviewed synonyms (the app's matcher)
          Outcomes: right sign / no sign / wrong sign. A sign named differently from the signer's (يدخل vs دخل, أحب vs
          حب) is judged by a language model for the same meaning (cached in results/text2gloss/judge.json); a wrong
          sign shows another meaning (شركة -> شرك), the most serious error.
  Test 2, sentence. The glossed sentence against the signer's signs that come from its words: order-free precision,
          recall, F1 and WER, for the systems of text2gloss_eval.py (copy, rule, llm; their saved outputs).

  py scripts/eval/matching_eval.py
Output: results/matching_eval.{json,md}
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402
from isharati.text.arabic_morph import NLP_PYTHON, STOPWORDS  # noqa: E402
import text2gloss_eval as T  # noqa: E402

OUT = ROOT / "results"
ROOTS_CACHE = OUT / "text2gloss" / "roots.json"
ROOTS_CODE = r"""
import json, os, sys
os.environ.setdefault("CAMELTOOLS_DATA", r"D:\islam\models\camel_data")
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
a = Analyzer(MorphologyDB.builtin_db())
out = {}
for w in json.loads(sys.stdin.read()):
    out[w] = sorted({x.get("root", "") for x in a.analyze(w)
                     if x.get("root") and "#" not in x.get("root", "") and x.get("root") != "NTWS"})
print(json.dumps(out, ensure_ascii=False))
"""


def roots(words):
    """CAMeL roots of each word (all readings), cached on disk."""
    cache = json.loads(ROOTS_CACHE.read_text(encoding="utf-8")) if ROOTS_CACHE.exists() else {}
    todo = sorted({w for w in words if w not in cache})
    if todo:
        r = subprocess.run([str(NLP_PYTHON), "-c", ROOTS_CODE], input=json.dumps(todo, ensure_ascii=False),
                           capture_output=True, text=True, encoding="utf-8", timeout=3600)
        cache.update(json.loads(r.stdout.strip().splitlines()[-1]))
        ROOTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ROOTS_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {w: set(cache.get(w, [])) for w in words}


def text_words(text):
    return [w for w in normalize_ar(strip_diacritics(text)).split() if w not in STOPWORDS]


def gold_words(gloss):
    return normalize_ar(strip_diacritics(gloss.replace("_", " "))).split()


def aligned(item, R):
    """(sentence word, signer's sign) pairs that share a root, and the signer's signs in order that come from the
    sentence (the reference of test 2)."""
    tw, gw = text_words(item["text"]), gold_words(item["gold"])
    pairs, ref = [], []
    for g in gw:
        src = [w for w in tw if R[w] & R[g]] or [w for w in tw if T.fold(w) == T.fold(g)]
        if src:
            ref.append(g)
            pairs.append((src[0], g))
    return pairs, ref


def same(sign, gold):
    """The matched sign is the signer's: equal after folding, or the signer's word is one of the sign's words."""
    s, g = T.fold(sign), T.fold(gold)
    return s == g or (len(g) == 1 and g[0] in s)


JUDGE_CACHE = OUT / "text2gloss" / "judge.json"
JUDGE_PROMPT = (
    "You check Arabic Sign Language glossing. For each item: an Arabic word, the sign a Deaf signer used for it, and "
    "the sign our system chose (sign names are Arabic words). Answer whether our sign means the same as the signer's "
    "for that word, so a Deaf viewer gets the same meaning: tense, person, number and spelling do not matter (يدخل = "
    "دخل, أحب = حب, واحد = 1), a different meaning does (شرك, polytheism, is not شركة, company; درّس, taught, is not "
    "مدرسة, school; عمرة, the pilgrimage, is not عمر, age). Return JSON only: {\"same\": [true/false, ...]} in order.")


def judge(items):
    """Same meaning? for (word, signer's sign, our sign) triples whose labels differ; cached, 40 per call."""
    from isharati.llm import OpenRouterClient
    cache = json.loads(JUDGE_CACHE.read_text(encoding="utf-8")) if JUDGE_CACHE.exists() else {}
    key = lambda t: "|".join(t)
    todo = [t for t in dict.fromkeys(items) if key(t) not in cache]
    client = OpenRouterClient("qwen/qwen3.8-27b", timeout=120)
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        lines = "\n".join(f"{k + 1}. word: {w} | signer: {g} | ours: {o}" for k, (w, g, o) in enumerate(batch))
        reply = client.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": lines}])
        same = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])["same"]
        if len(same) == len(batch):
            cache.update({key(t): bool(v) for t, v in zip(batch, same)})
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return {t: cache.get(key(t)) for t in items}


def main():
    from isharati.lexicon import arabic as ar_lexicon
    from isharati.lexicon.arabic import Lexicon
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    folded_glosses = {tuple(T.fold(e.gloss)) for e in lex.sign_entries()}
    sets = {"isharah": T.isharah(300), "religious": T.religious(), "arabsign": T.arabsign()}
    words = {w for s in sets.values() for x in s for w in text_words(x["text"]) + gold_words(x["gold"])}
    print(f"roots for {len(words)} words", flush=True)
    R = roots(words)

    levels = {
        "exact": lambda w: lex.lookup(w),
        "affixes": lambda w: Lexicon.match_token(lex, w),  # the base matcher: string candidates only
        "full": lambda w: lex.match_token(w),
    }
    outs = {name: {r["text"]: r["output"] for r in map(json.loads, (OUT / "text2gloss" / f"{name}.jsonl").open(encoding="utf-8"))}
            for name in ("copy", "rule", "llm")}
    report, examples = {}, {"wrong": [], "missed": []}
    for sname, items in sets.items():
        # test 1
        pairs = [p for x in items for p in aligned(x, R)[0] if tuple(T.fold(p[1])) in folded_glosses]
        t1 = {}
        found = {lvl: [(w, g, fn(w)) for w, g in pairs] for lvl, fn in levels.items()}
        verdict = judge([(w, g, e.gloss) for lvl in found for w, g, e in found[lvl] if e is not None and not same(e.gloss, g)])
        for lvl, rows in found.items():
            c = Counter()
            for w, g, e in rows:
                if e is None:
                    k = "none"
                elif same(e.gloss, g) or verdict.get((w, g, e.gloss)):
                    k = "right"
                else:
                    k = "wrong"
                c[k] += 1
                if lvl == "full" and k != "right" and len(examples["wrong" if k == "wrong" else "missed"]) < 25:
                    examples["wrong" if k == "wrong" else "missed"].append(
                        {"set": sname, "word": w, "signer": g, "matched": e.gloss if e else None})
            n = max(1, sum(c.values()))
            t1[lvl] = {"pairs": n, "right": round(c["right"] / n, 3), "no_sign": round(c["none"] / n, 3),
                       "wrong_sign": round(c["wrong"] / n, 3)}
        # test 2
        t2 = {}
        for name, done in outs.items():
            e_sum = n_sum = common = hyp_n = 0
            for x in items:
                ref = T.fold(" ".join(aligned(x, R)[1]))
                hyp = T.fold(done.get(x["text"], ""))
                e_sum += T.edit(ref, hyp)
                n_sum += len(ref)
                common += sum((Counter(ref) & Counter(hyp)).values())
                hyp_n += len(hyp)
            p, r = common / max(1, hyp_n), common / max(1, n_sum)
            t2[name] = {"precision": round(p, 3), "recall": round(r, 3),
                        "f1": round(2 * p * r / (p + r) if p + r else 0, 3)}
        kept = sum(len(aligned(x, R)[1]) for x in items)
        total = sum(len(gold_words(x["gold"])) for x in items)
        report[sname] = {"sentences": len(items), "signer_signs_from_the_text": round(kept / max(1, total), 3),
                         "word_to_sign": t1, "sentence": t2}
    report["_examples"] = examples
    (OUT / "matching_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# What Isharati does with Arabic text, against Deaf signers' annotations", "",
          "Only the signer's signs that come from a word of the sentence (same root) are compared; the signer's added",
          "signs (an added «انا», question markers) are left out, since no text-to-sign system produces them.", ""]
    for sname in ("isharah", "religious", "arabsign"):
        s = report[sname]
        md += [f"## {sname} ({s['sentences']} sentences; {s['signer_signs_from_the_text']:.0%} of the signer's signs "
               "come from the text)", "", f"**Word -> sign** ({s['word_to_sign']['full']['pairs']} word pairs whose "
               "sign is in the lexicon)", "", "| matcher | right sign | no sign | wrong sign |", "|---|---|---|---|"]
        md += [f"| {k} | {v['right']:.1%} | {v['no_sign']:.1%} | {v['wrong_sign']:.1%} |" for k, v in s["word_to_sign"].items()]
        md += ["", "**Sentence** (the signer's signs that come from the text)", "",
               "| system | precision | recall | F1 |", "|---|---|---|---|"]
        md += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |" for k, v in s["sentence"].items()]
        md.append("")
    md += ["## Errors of the full matcher (first 25 each)", "", "| set | word | signer's sign | matched | kind |",
           "|---|---|---|---|---|"]
    md += [f"| {e['set']} | {e['word']} | {e['signer']} | {e['matched']} | wrong sign |" for e in examples["wrong"]]
    md += [f"| {e['set']} | {e['word']} | {e['signer']} | — | no sign |" for e in examples["missed"]]
    (OUT / "matching_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
