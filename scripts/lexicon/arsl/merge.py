"""The Arabic lexicon of the question-answering demo: data/lexicon_v2/lexicon_qa.jsonl.

The public lexicon (KArSL + religious signs) first; then Tawasol Center signs (scripts/lexicon/arsl/tawasol_terms.py)
for glosses it lacks, such as رمضان and the phrase signs of the pillars of Islam; then the explained Saudi and unified
dictionaries; then Isharah sentence-final signs
(scripts/lexicon/arsl/isharah/last_sign.py) that passed their validation, for non-commercial research; then Levantine
Shorts (scripts/lexicon/arsl/jordan_shorts.py), tagged by dialect.
  .venv/Scripts/python scripts/lexicon/arsl/merge.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402

BASE = DATA / "lexicon_v2"
FIELDS = {f.name for f in __import__("dataclasses").fields(__import__("isharati.types", fromlist=["SignEntry"]).SignEntry)}
SOURCES = [  # first source wins per gloss: Saudi signers first, other dialects only to fill gaps
    BASE / "reviewed" / "entries.jsonl",              # native-signer decisions (fingerspelled.py): replace any other sign
    BASE / "reviewed" / "recordings.jsonl",           # recordings the team chose over a weak sign (reviewed_recordings.py)
    BASE / "lexicon.jsonl",                           # KArSL + religious signs from the curriculum
    BASE / "tawasol" / "entries.jsonl",               # Tawasol Center vocabulary lessons (Saudi)
    BASE / "saudi_dictionary" / "entries.jsonl",      # Saudi / unified Arabic dictionary, explained entry by entry
    BASE / "arabic_dictionaries" / "entries_unified_dictionary.jsonl",   # the unified Arabic sign dictionary
    BASE / "arabic_dictionaries" / "entries_kuwaiti_dictionary.jsonl",   # Gulf: the Kuwaiti sign dictionary
    BASE / "arabic_dictionaries" / "entries_children_dictionary.jsonl",
    BASE / "arabic_dictionaries" / "entries_scouts_dictionary.jsonl",
    BASE / "ayman_abbas" / "entries.jsonl",           # «إشارتي هي لغتي» (Ayman Abbas) captioned vocabulary videos
    BASE / "timed_signs" / "entries.jsonl",           # single signs cut at Whisper-timed words (timed_signs.py)
    BASE / "isharah_final" / "accepted.jsonl",        # Isharah sentence-final signs that passed validation
    BASE / "new_arabic_sources" / "entries.jsonl",    # Newly extracted Islamic & educational signs from drive E:
    BASE / "jordan_shorts" / "entries.jsonl",         # Levantine one-word Shorts, last
    BASE / "arabsign" / "entries_vetted.jsonl",       # ArabSign sentence clips, glosses vetted against the signed sentence
]


def main():
    rows, seen, counts, notes = [], set(), {}, {}
    for path in SOURCES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("variant_of"):  # another recording of a sign already listed: never a replacement
                continue
            key = normalize_ar(e["gloss"])
            if key and key not in seen and (BASE / e["keypoints_path"]).exists():
                seen.add(key)
                extra = {k: e.pop(k) for k in list(e) if k not in FIELDS}  # notes, dialect: kept beside the entry
                if extra:
                    notes[e["sign_id"]] = extra
                rows.append(e)
                counts[e["dataset"]] = counts.get(e["dataset"], 0) + 1
    (BASE / "lexicon_qa.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (BASE / "lexicon_qa_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"lexicon_qa: {len(rows)} glosses {counts}")


if __name__ == "__main__":
    main()
