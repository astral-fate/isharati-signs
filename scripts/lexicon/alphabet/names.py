"""Common Sahaba names, fingerspelled from each language's one-signer alphabet (scripts/lexicon/alphabet/build.py),
for the languages whose sign lexicon has no sign for them: ASL (English transliteration) and TİD (Turkish).

Each name is spelled letter by letter with isharati.pose.fingerspell.build (static letters as holds, motion letters as
their whole movement, all on the alphabet signer's hand), with the face track. Names a lexicon already signs are
skipped, so a real sign is never displaced. All entries are "pending" a Deaf reviewer, like the Arabic names in
scripts/lexicon/arsl/fingerspelled.json. Output, read by asl_lexicon() / tid_lexicon() (Lexicon.add_words):
  data/asl/lexicon/names/{signs/*.npy, entries.jsonl}   data/lexicon_tr/names/{signs/*.npy, entries.jsonl}
  data/face/asl/<sign_id>.npz                            data/face/tr/<sign_id>.npz
  PYTHONPATH=src python scripts/lexicon/alphabet/names.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon.asl import asl_lexicon  # noqa: E402
from isharati.lexicon.turkish import tid_lexicon  # noqa: E402
from isharati.pose import fingerspell  # noqa: E402

# English and Turkish forms of the same companions (and of Hamza, Muadh, Musab ...), in the spelling each language uses
NAMES = {
    "en": ["ABU BAKR", "UMAR", "UTHMAN", "ALI", "AISHA", "KHADIJA", "FATIMA", "ABU HURAYRAH", "BILAL", "HAMZA",
           "HASAN", "HUSAYN", "ANAS", "MUADH", "SALMAN", "TALHA", "ZUBAYR", "HAFSA", "ZAYNAB", "JABIR", "SAAD",
           "ABU DHARR", "MUSAB", "SUHAYB"],
    "tr": ["EBU BEKİR", "ÖMER", "OSMAN", "ALİ", "AYŞE", "HATİCE", "FATMA", "EBU HÜREYRE", "BİLAL", "HAMZA", "HASAN",
           "HÜSEYİN", "ENES", "MUAZ", "SELMAN", "TALHA", "ZÜBEYR", "HAFSA", "ZEYNEP", "CABİR", "SAD", "EBU ZER",
           "MUSAB", "SUHEYB"],
}
LANGS = {"en": (asl_lexicon, DATA / "asl" / "lexicon", DATA / "face" / "asl", "aslname"),
         "tr": (tid_lexicon, DATA / "lexicon_tr", DATA / "face" / "tr", "tidname")}


def main():
    for lang, (load, lex_dir, faces, prefix) in LANGS.items():
        lex = load()
        src = json.loads((lex_dir / "alphabet" / "sources.json").read_text(encoding="utf-8"))
        motion = {r["letter"] for r in src["letters"] if r["kind"] == "motion"}
        out = lex_dir / "names"
        (out / "signs").mkdir(parents=True, exist_ok=True)
        rows = []
        for name in NAMES[lang]:
            if lex.lookup(name) is not None:
                print(f"{lang} {name}: already signed ({lex.lookup(name).sign_id}), skipped")
                continue
            entries = lex.fingerspell(name)
            clip, face, hand, _, lengths = fingerspell.build(entries, lex.keypoints, faces, motion=motion,
                                                              hand=src["signer_hand"])
            sid = f"{prefix}_" + hashlib.sha1(name.encode()).hexdigest()[:8]
            np.save(out / "signs" / f"{sid}.npy", clip)
            if face is not None:
                fingerspell.save_face(faces / f"{sid}.npz", face, len(clip), f"fingerspelled from {src['url']}")
            rows.append({"gloss": name, "sign_id": sid, "dataset": f"{prefix}_fingerspelled",
                         "keypoints_path": str(Path(out.name) / "signs" / f"{sid}.npy").replace("\\", "/") if lang == "tr"
                         else f"names/signs/{sid}.npy",
                         "review_status": "pending", "is_religious": True, "is_letter": False,
                         "fingerspelled": " ".join(e.gloss for e in entries), "note": "Sahabi name, pending Deaf review"})
            print(f"{lang} {name}: {' '.join(e.gloss for e in entries)} -> {sid}, {len(clip)} frames, {hand} hand, "
                  f"face {'yes' if face is not None else 'no'}")
        (out / "entries.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                           encoding="utf-8")
        print(f"{lang}: {len(rows)} names -> {out / 'entries.jsonl'}")


if __name__ == "__main__":
    main()
