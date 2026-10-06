"""Merge every ASL sign source into data/asl/lexicon/lexicon_all.jsonl (loaded by src/isharati/pipeline.py).

One entry per gloss; the first source listed wins, so religious terms come from Deaf signers of Islamic vocabulary
before general dictionaries. Paths stay relative to data/asl/lexicon/.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEX = DATA / "asl" / "lexicon"
SOURCES = [  # (entries file, path prefix relative to LEX)
    (LEX / "islamic" / "entries.jsonl", ""),        # GDM's 40 Islamic signs (Deaf signer), paths already prefixed
    (LEX / "noor" / "entries.jsonl", ""),           # Noor For Sign, Ramadan vocabulary
    (LEX / "single_clips" / "entries.jsonl", ""),
    (LEX / "obscure_asl" / "entries.jsonl", ""),
    (LEX / "lexicon.jsonl", ""),                     # ASL Citizen (signs/..., templates in templates.jsonl)
    (LEX / "wlasl" / "entries.jsonl", ""),          # WLASL: words ASL Citizen lacks (scripts/lexicon/asl/wlasl.py)
    (LEX / "msasl" / "entries.jsonl", ""),          # MS-ASL: words neither has (scripts/lexicon/asl/msasl.py)
]


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
    from isharati.lexicon.asl_citizen import write_index
    from isharati.lexicon.asl import gloss_key
    write_index()  # ASL Citizen signs built so far (the build may still be running)
    chosen, seen = [], set()
    for path, prefix in SOURCES:
        if not path.exists():
            print(f"missing {path.relative_to(ROOT)}, skipped")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            key = gloss_key(e["gloss"])
            if key and key not in seen and (LEX / (prefix + e["keypoints_path"])).exists():
                seen.add(key)
                e["keypoints_path"] = prefix + e["keypoints_path"]
                chosen.append(e)
    # synonyms: a word with no sign of its own is signed with an existing sign of the same meaning (CRESCENT -> MOON)
    synonyms = json.loads((Path(__file__).parent / "synonyms.json").read_text(encoding="utf-8"))
    by_key = {gloss_key(e["gloss"]): e for e in chosen}
    for word, target in synonyms.items():
        if word.startswith("_") or gloss_key(word) in seen or gloss_key(target) not in by_key:
            continue
        t = by_key[gloss_key(target)]
        chosen.append({**t, "gloss": word.upper(), "dataset": "synonym", "review_status": "pending"})
        seen.add(gloss_key(word))
    (LEX / "lexicon_all.jsonl").write_text("\n".join(json.dumps(e) for e in chosen), encoding="utf-8")
    print(f"lexicon_all: {len(chosen)} glosses {dict(Counter(e['dataset'] for e in chosen))}")


if __name__ == "__main__":
    main()
