"""Sentence-level back-translation for Arabic -> ArSL: BLEU, chrF and ROUGE on 10 signed sentences.

Each sentence goes through the app's Studio path (the LLM glosser restricted to the lexicon, then the assembled pose),
and is translated back to Arabic in two ways:
  gloss -> text   the glosses the system meant to sign, turned back into an Arabic sentence by an LLM that never sees
                  the source. Measures the meaning lost by glossing alone (an upper bound for back-translation).
  pose -> text    the glosses a recogniser reads off the produced pose (DTW 1-nearest-neighbour against a signer whose
                  recordings are not used to build the output: data/lexicon/templates.jsonl), turned into Arabic by
                  the same LLM. This is back-translation proper: what the signing itself carries.
Both are scored against the source sentence, after normalize_ar (no diacritics, folded alef/yaa/taa marbuta):
  BLEU  sacrebleu, sentence and corpus level (effective order for short sentences)
  chrF  sacrebleu character n-gram F-score, kinder to Arabic morphology (clitics, prefixes)
  ROUGE-1 / ROUGE-2 / ROUGE-L F1 on whitespace tokens (rouge_score's tokenizer drops Arabic letters, so it is
        computed here)
Output: results/bt_sentence_eval.{json,md}
  python scripts/eval/bt_sentence.py [--runs 3]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.backtranslate import DTWRecognizer, score  # noqa: E402
from isharati.glossing.arabic import ArabicQAGlosser  # noqa: E402
from isharati.llm import default_client  # noqa: E402
from isharati.pipeline import AR_TEMPLATES, load_lexicon  # noqa: E402
from isharati.pose.poser import MissingAwarePoser  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402

OUT = ROOT / "results"

# Ten sentences of the kind the app signs: Qur'an and hadith wording and short statements about the pillars.
SENTENCES = [
    ("إنما الأعمال بالنيات", "Bukhari 1"),
    ("الدين النصيحة", "Muslim 55"),
    ("المسلم أخو المسلم", "Bukhari 2442"),
    ("الطهور شطر الإيمان", "Muslim 223"),
    ("الصدق يهدي إلى البر", "Bukhari 6094"),
    ("الحمد لله رب العالمين", "al-Fatihah 1:2"),
    ("أقيموا الصلاة وآتوا الزكاة", "al-Baqarah 2:43"),
    ("صوم رمضان فرض على كل مسلم", "statement"),
    ("الحج ركن من أركان الإسلام", "statement"),
    ("الله رحيم بعباده", "statement"),
]

BACK_SYSTEM = ("You translate Arabic Sign Language back into Arabic. You get the signs of one sentence as glosses, in "
               "signing order (sign languages drop function words and reorder). Write the single Arabic sentence they "
               "express, in Modern Standard Arabic, without adding facts. Reply with the sentence only.")


def back_translate(client, glosses):
    if not glosses:
        return ""
    reply = client.chat([{"role": "system", "content": BACK_SYSTEM},
                         {"role": "user", "content": " - ".join(glosses)}])
    return reply.strip().splitlines()[0].strip() if reply.strip() else ""


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def f1(ref, hyp):
    if not ref or not hyp:
        return 0.0
    common = sum(min(ref.count(g), hyp.count(g)) for g in set(hyp))
    if not common:
        return 0.0
    p, r = common / len(hyp), common / len(ref)
    return 2 * p * r / (p + r)


def lcs(a, b):
    d = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], prev + 1 if x == y else max(d[j], d[j - 1])
    return d[-1]


def rouge(ref, hyp):
    r, h = normalize_ar(ref).split(), normalize_ar(hyp).split()
    l = lcs(r, h)
    rl = 0.0 if not l else 2 * (l / len(h)) * (l / len(r)) / ((l / len(h)) + (l / len(r)))
    return {"rouge1": f1(ngrams(r, 1), ngrams(h, 1)), "rouge2": f1(ngrams(r, 2), ngrams(h, 2)), "rougeL": rl}


def metrics(refs, hyps):
    """Corpus BLEU / chrF and mean sentence ROUGE over aligned lists (all text already normalised)."""
    refs_n, hyps_n = [normalize_ar(x) for x in refs], [normalize_ar(x) for x in hyps]
    rs = [rouge(r, h) for r, h in zip(refs, hyps)]
    return {"bleu": sacrebleu.corpus_bleu(hyps_n, [refs_n]).score,
            "chrf": sacrebleu.corpus_chrf(hyps_n, [refs_n]).score,
            **{k: 100 * statistics.mean(x[k] for x in rs) for k in ("rouge1", "rouge2", "rougeL")}}


def sentence_scores(ref, hyp):
    r, h = normalize_ar(ref), normalize_ar(hyp)
    return {"bleu": sacrebleu.sentence_bleu(h, [r], use_effective_order=True).score,
            "chrf": sacrebleu.sentence_chrf(h, [r]).score, **{k: 100 * v for k, v in rouge(ref, hyp).items()}}


def run_once(lexicon, glosser, recognizer, client):
    poser, rows = MissingAwarePoser(lexicon), []
    template_glosses = {g for g, _ in recognizer.templates}
    for text, source in SENTENCES:
        res = glosser.gloss(text)
        pose, segments = poser.pose(res)
        bt = score(pose, segments, res.glosses, recognizer)
        rows.append({"text": text, "source": source, "glosses": res.glosses, "missing": res.oov,
                     "recognized": bt.recognized, "gloss_acc": bt.gloss_acc, "gloss_wer": bt.wer,
                     "in_templates": sum(g in template_glosses for g in res.glosses),
                     "gloss_to_text": back_translate(client, res.glosses),
                     "pose_to_text": back_translate(client, bt.recognized)})
    refs = [r["text"] for r in rows]
    for r in rows:
        r["scores"] = {"gloss_to_text": sentence_scores(r["text"], r["gloss_to_text"]),
                       "pose_to_text": sentence_scores(r["text"], r["pose_to_text"])}
    return {"rows": rows,
            "gloss_to_text": metrics(refs, [r["gloss_to_text"] for r in rows]),
            "pose_to_text": metrics(refs, [r["pose_to_text"] for r in rows]),
            "gloss_acc": statistics.mean(r["gloss_acc"] for r in rows),
            "gloss_wer": statistics.mean(r["gloss_wer"] for r in rows)}


def write_md(result):
    runs, keys = result["runs"], ("bleu", "chrf", "rouge1", "rouge2", "rougeL")
    def cell(path, k):
        vals = [r[path][k] for r in runs]
        return f"{statistics.mean(vals):.1f}" + (f" ± {statistics.stdev(vals):.1f}" if len(vals) > 1 else "")
    lines = ["# Sentence-level back-translation, Arabic -> ArSL (10 sentences)", "",
             f"{len(runs)} run(s); mean ± sd over runs. Recogniser: {result['recognizer']}. "
             f"Back-translation LLM: {result['llm']}.", "",
             "| path | BLEU | chrF | ROUGE-1 | ROUGE-2 | ROUGE-L |", "|---|---|---|---|---|---|"]
    for path, label in (("gloss_to_text", "gloss -> text (glossing loss only)"),
                        ("pose_to_text", "pose -> text (back-translation)")):
        lines.append(f"| {label} | " + " | ".join(cell(path, k) for k in keys) + " |")
    accs = [r["gloss_acc"] for r in runs]
    lines += ["", f"Sign-level back-translation on the same sentences: gloss accuracy "
              f"{100 * statistics.mean(accs):.0f}%, gloss WER {statistics.mean(r['gloss_wer'] for r in runs):.2f}. "
              f"Only {sum(r['in_templates'] for r in runs[0]['rows'])} of the "
              f"{sum(len(r['glosses']) for r in runs[0]['rows'])} signed glosses (run 1) have a held-out-signer "
              "template, so the recogniser cannot name the others: the pose -> text row is bounded by the "
              "recogniser, not only by the signing.", "",
              "Run 1, per sentence:", "",
              "| source | sentence | glosses | gloss -> text | recognised | pose -> text | BLEU g/p | chrF g/p |",
              "|---|---|---|---|---|---|---|---|"]
    for r in runs[0]["rows"]:
        g, p = r["scores"]["gloss_to_text"], r["scores"]["pose_to_text"]
        lines.append(f"| {r['source']} | {r['text']} | {' '.join(r['glosses'])} | {r['gloss_to_text']} | "
                     f"{' '.join(r['recognized'])} | {r['pose_to_text']} | {g['bleu']:.0f} / {p['bleu']:.0f} | "
                     f"{g['chrf']:.0f} / {p['chrf']:.0f} |")
    (OUT / "bt_sentence_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    client = default_client()
    lexicon = load_lexicon("ar")
    glosser = ArabicQAGlosser(lexicon, client)
    recognizer = DTWRecognizer.from_templates(AR_TEMPLATES)
    runs = []
    for i in range(args.runs):
        runs.append(run_once(lexicon, glosser, recognizer, client))
        print(f"run {i + 1}: gloss->text {runs[-1]['gloss_to_text']}  pose->text {runs[-1]['pose_to_text']}", flush=True)
    result = {"recognizer": recognizer.name, "llm": getattr(client, "last", None), "sentences": SENTENCES, "runs": runs}
    OUT.mkdir(exist_ok=True)
    (OUT / "bt_sentence_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(result)
    print((OUT / "bt_sentence_eval.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
