"""Second pass over the uncovered words that concept_map.py marked NEED: the closest existing ASL sign that conveys the
word's meaning in Qur'an and hadith texts (a related form such as guidance -> GUIDE, or a close concept), or NONE.

  .venv/Scripts/python scripts/lexicon/asl/close_signs.py
Output: data/asl/close_signs.json  {word: {"sign": GLOSS | "NONE", "relation": "form" | "close"}}; merge into
src/isharati/lexicon/synonyms.json only after review (scripts/lexicon/asl/merge.py reads that file).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
OUT = DATA / "asl" / "close_signs.json"
BATCH = 40

SYSTEM = """You are an ASL linguist preparing Qur'an translations and hadith for ASL signing. For each English word, pick the
existing sign from the gloss list that an ASL signer would use for it in these religious texts:
- "form": a gloss of the same word family (guidance -> GUIDE, prostration -> BOW if that is how it is signed),
- "close": a different gloss whose meaning is close enough that a Deaf reader understands the same thing,
- NONE if every listed gloss would change the meaning (do not stretch: a wrong sign in religious text is worse than none).
Use only glosses from the list. Reply with JSON only: {"word": {"sign": "GLOSS" | "NONE", "relation": "form" | "close"}, ...}"""


def main():
    from isharati.lexicon.asl import ASLLexicon
    from isharati.llm import default_client
    lex = ASLLexicon.load(DATA / "asl" / "lexicon" / "lexicon_all.jsonl")
    glosses = sorted({e.gloss.rstrip("0123456789") for e in lex.sign_entries() if not e.is_letter})
    cmap = json.loads((DATA / "asl" / "concept_map.json").read_text(encoding="utf-8"))
    need = [w for w, v in cmap.items() if v["sign"] == "NEED"]
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [w for w in need if w not in done]
    client = default_client()
    for k in range(0, len(todo), BATCH):
        words = todo[k:k + BATCH]
        try:
            reply = client.chat([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"Gloss list: {' '.join(glosses)}\n\nWords: {', '.join(words)}"}])
            got = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
        except Exception as e:
            print("batch failed:", e, flush=True)
            continue
        for w in words:
            v = got.get(w) or {}
            sign = str(v.get("sign", "")).upper()
            if sign == "NONE" or lex.lookup(sign):
                done[w] = {"sign": sign, "relation": v.get("relation", "")}
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{min(k + BATCH, len(todo))}/{len(todo)}", flush=True)
    found = {w: v for w, v in done.items() if v["sign"] != "NONE"}
    print(len(found), "of", len(done), "have a close sign")


if __name__ == "__main__":
    main()
