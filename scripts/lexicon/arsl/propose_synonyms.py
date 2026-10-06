"""Proposals for reviewed Arabic synonyms: the most frequent words the ArSL lexicon does not sign (results/ar_uncovered.json,
from scripts/eval/corpus_eda.py), grouped by their lemma (CAMeL), each with an existing sign of the same meaning chosen by
a language model, or none. Nothing here enters the app: every proposal is reviewed, and the accepted ones are written to
src/isharati/lexicon/synonyms_ar.json (verb_lemmas / noun_lemmas, so every form of the lemma finds the sign).

  py scripts/lexicon/arsl/propose_synonyms.py [--top 1500]
Output: results/ar_synonym_proposals.json  [{kind, lemma, example, count, proposal}]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
from isharati.llm import OpenRouterClient  # noqa: E402
from isharati.text import arabic_morph as morph  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402

OUT = ROOT / "results" / "ar_synonym_proposals.json"
PROMPT = (
    "You map Arabic words from the Qur'an and hadith to Arabic Sign Language signs. For each item (a lemma, its part of "
    "speech, and an example form from the texts) choose ONE sign from the list whose meaning is the same, so a Deaf "
    "viewer understands exactly that meaning: a synonym, or the same concept as a noun/verb (آمن -> إيمان, أخبر -> "
    "خبر). Never a related, broader or opposite concept, never a sign that only shares letters (شرك is not شركة). If no "
    "sign has the same meaning, answer null. Return JSON only: {\"signs\": [sign or null, ...]} in the order given.")


def main():
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 1500
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    uncovered = json.loads((ROOT / "results" / "ar_uncovered.json").read_text(encoding="utf-8"))[:top]
    groups = defaultdict(lambda: {"count": 0, "forms": []})
    for w, n in uncovered:
        a = morph.top_analyses(w)
        if not a or a[0]["pos"] == "noun_prop":
            continue
        kind = "verb" if a[0]["pos"] == "verb" else "nom" if a[0]["pos"] in morph.NOMINAL else None
        if kind is None:
            continue
        g = groups[(kind, strip_diacritics(a[0]["lex"]).split("_")[0])]
        g["count"] += n
        g["forms"].append(w)
    items = sorted(groups.items(), key=lambda kv: -kv[1]["count"])
    signs = sorted({e.gloss for e in lex.sign_entries() if len(e.gloss.split()) <= 2})
    client = OpenRouterClient("qwen/qwen3.8-27b", timeout=240)
    done = {(p["kind"], p["lemma"]): p for p in json.loads(OUT.read_text(encoding="utf-8"))} if OUT.exists() else {}
    todo = [(k, v) for k, v in items if k not in done]
    for i in range(0, len(todo), 60):
        batch = todo[i:i + 60]
        lines = "\n".join(f"{j + 1}. {lem} ({'verb' if kind == 'verb' else 'noun/adjective'}), e.g. «{v['forms'][0]}»"
                          for j, ((kind, lem), v) in enumerate(batch))
        reply = client.chat([{"role": "system", "content": PROMPT},
                             {"role": "user", "content": "Signs: " + "، ".join(signs) + "\n\nItems:\n" + lines}])
        try:
            got = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])["signs"]
        except (ValueError, KeyError):
            print("unreadable reply, batch skipped", flush=True)
            continue
        if len(got) != len(batch):
            print("length mismatch, batch skipped", flush=True)
            continue
        for ((kind, lem), v), s in zip(batch, got):
            ok = s if s and normalize_ar(s) in lex._signs else None  # only signs that exist
            done[(kind, lem)] = {"kind": kind, "lemma": lem, "example": v["forms"][0], "forms": v["forms"][:6],
                                 "count": v["count"], "proposal": ok}
        OUT.write_text(json.dumps(sorted(done.values(), key=lambda p: -p["count"]), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"{min(i + 60, len(todo))}/{len(todo)} lemmas", flush=True)
    props = [p for p in done.values() if p["proposal"]]
    print(f"{len(done)} lemmas, {len(props)} with a proposed sign, covering {sum(p['count'] for p in props):,} tokens")


if __name__ == "__main__":
    main()
