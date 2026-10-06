"""Which uncovered Qur'an/hadith words an ASL signer would express with a sign the lexicon already has.

ASL signs concepts, not English words: «deeds» is signed WORK/DO, «indeed» and «nor» are not signed at all (the
sentence carries them). For the most frequent uncovered words (results/coverage_en.json top_missing) the LLM picks
  an existing gloss that an ASL signer would use for that word in Islamic texts,
  DROP when ASL does not sign the word (function words, emphasis), or
  NEED when the word needs a sign of its own.
The map is a proposal, marked review_status pending, for a Deaf reviewer to confirm.

  .venv/Scripts/python scripts/eval/concept_map.py
Output: data/asl/concept_map.json; results/coverage_en.json gains "concept" figures
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
OUT = DATA / "asl" / "concept_map.json"
BATCH = 40

SYSTEM = """You are an ASL linguist preparing Islamic texts (Qur'an translation, hadith) for ASL signing.
For each English word, answer how a fluent ASL signer would sign it, using ONLY the gloss list given:
- an existing gloss with the same meaning in these texts (e.g. deeds -> WORK, punishment -> PUNISH if listed),
- DROP if ASL does not sign the word (articles, emphasis such as "indeed", connectives the sign order carries),
- NEED if no listed gloss expresses it and it must be signed (fingerspelling does not count).
Never pick a gloss that only sounds similar. Reply with JSON only: {"word": "GLOSS" | "DROP" | "NEED", ...}"""


def main():
    from isharati.lexicon.asl import ASLLexicon
    from isharati.llm import default_client
    lex = ASLLexicon.load(DATA / "asl" / "lexicon" / "lexicon_all.jsonl")
    glosses = sorted({e.gloss.rstrip("0123456789") for e in lex.sign_entries() if not e.is_letter})
    known = set(glosses)
    report = json.loads((ROOT / "results" / "coverage_en.json").read_text(encoding="utf-8"))
    missing = report["top_missing"]
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    client = default_client()
    todo = [w for w, _ in missing if w not in done]
    for k in range(0, len(todo), BATCH):
        words = todo[k:k + BATCH]
        msg = f"Gloss list: {' '.join(glosses)}\n\nWords: {', '.join(words)}"
        try:
            reply = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}])
            got = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
        except Exception as e:
            print("batch failed:", e, flush=True)
            continue
        for w in words:
            v = str(got.get(w, "")).strip().upper()
            if v in ("DROP", "NEED") or v in known:  # a gloss outside the list is an invention: not kept
                done[w] = {"sign": v, "review_status": "pending"}
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{min(k + BATCH, len(todo))}/{len(todo)}", flush=True)

    tot = report["all"]["vocabulary_no_names"]
    n = dict(missing)
    by = {"gloss": 0, "DROP": 0, "NEED": 0}
    for w, v in done.items():
        if w in n:
            by["gloss" if v["sign"] not in ("DROP", "NEED") else v["sign"]] += n[w]
    report["concept"] = {
        "words_mapped": len(done), "tokens": tot["tokens"],
        "direct": tot["tokens_covered"], "via_existing_gloss": by["gloss"], "not_signed_in_asl": by["DROP"],
        "need_new_sign_top500": by["NEED"],
        "signable_coverage": round((tot["tokens_covered"] + by["gloss"]) / tot["tokens"], 4),
        "signable_coverage_excl_dropped": round((tot["tokens_covered"] + by["gloss"]) / (tot["tokens"] - by["DROP"]), 4)}
    (ROOT / "results" / "coverage_en.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["concept"], indent=1))
    need = [w for w, _ in missing if done.get(w, {}).get("sign") == "NEED"]
    print("NEED:", ", ".join(need[:80]))


if __name__ == "__main__":
    main()
