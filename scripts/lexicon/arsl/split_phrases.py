"""Single signs cut out of phrase signs, where no source has the word on its own: محمد from «محمد رسول الله», الإسلام
from «أركان الإسلام», «أركان الإسلام الخمسة» and «وزارة الشؤون الإسلامية».

A phrase sign is its word signs in a row, so it is cut into n parts at the n-1 deepest pauses in hand motion
(gloss_clips.cut). A cut is accepted only if it is checked:
  - by known parts: a part whose word has its own sign (رسول, الله) must be closer to that sign than to the other
    parts' signs — the cut is then in the right place, and the remaining part is the missing word;
  - by agreement: the same word cut from different phrases (الإسلام from three) must be closer to each other than to
    the other parts of those phrases.
Accepted signs go to data/lexicon_v2/split/{signs/*.npy, entries.jsonl, report.md} (review_status "pending").
  .venv/Scripts/python scripts/lexicon/arsl/split_phrases.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent / "isharah"))
from gloss_clips import cut  # noqa: E402

from isharati.pose.dtw import dtw_distance  # noqa: E402
from isharati.lexicon.arabic import Lexicon  # noqa: E402
from isharati.pose.proportions import fix  # noqa: E402

BASE = DATA / "lexicon_v2"
OUT = BASE / "split"
# phrase gloss -> its words, in order; None = a word we want (the target), a string = a word with its own sign
PHRASES = {
    "محمد رسول الله": ["محمد", "رسول", "الله"],
    "أركان الإسلام": ["أركان", "الإسلام"],
    "أركان الإسلام الخمسة": ["أركان", "الإسلام", "5"],
    "وزارة الشؤون الإسلامية": ["وزارة", "الشؤون", "الإسلام"],
}
TARGETS = {"محمد", "الإسلام"}


def main():
    lex = Lexicon.load(BASE / "lexicon_qa.jsonl")
    clip = lambda g: fix(lex.keypoints(lex.lookup(g)))
    parts = {}  # (phrase, word) -> clip
    for phrase, words in PHRASES.items():
        e = lex.lookup(phrase)
        if e is None:
            print("no phrase sign:", phrase)
            continue
        p = fix(lex.keypoints(e))
        spans = cut(p, len(words))
        if not spans:
            print("too short to cut:", phrase)
            continue
        for w, (a, b) in zip(words, spans):
            parts[(phrase, w)] = p[a:b]
    report = ["# Signs cut out of phrase signs\n", "Lower DTW = more similar.\n"]
    ok = {}

    # check 1: known parts land on their own signs
    report += ["## Known parts vs their own signs\n", "| phrase | part | DTW to own sign | DTW to the phrase's other parts' signs | ok |",
               "|---|---|---|---|---|"]
    for phrase, words in PHRASES.items():
        known = [w for w in words if w not in TARGETS and lex.lookup(w) is not None]
        good = True
        for w in known:
            if (phrase, w) not in parts:
                good = False
                continue
            own = dtw_distance(parts[(phrase, w)], clip(w))
            others = [dtw_distance(parts[(phrase, w)], clip(o)) for o in known if o != w]
            fine = not others or own < min(others)
            good &= fine
            report.append(f"| {phrase} | {w} | {own:.3f} | {', '.join(f'{x:.3f}' for x in others) or '-'} | {'yes' if fine else 'no'} |")
        for w in words:
            if w in TARGETS and (phrase, w) in parts:
                ok.setdefault(w, []).append((phrase, good and bool(known)))

    # check 2: the same target cut from different phrases agrees
    report += ["\n## The same word cut from different phrases\n", "| word | pair | DTW between the two cuts | mean DTW to the other parts |",
               "|---|---|---|---|"]
    agree = {}
    for w in TARGETS:
        cuts = [(ph, parts[(ph, w)]) for ph, words in PHRASES.items() if (ph, w) in parts]
        for (p1, c1), (p2, c2) in combinations(cuts, 2):
            same = dtw_distance(c1, c2)
            rest = [dtw_distance(c1, parts[(p2, o)]) for o in PHRASES[p2] if o != w and (p2, o) in parts] + \
                   [dtw_distance(c2, parts[(p1, o)]) for o in PHRASES[p1] if o != w and (p1, o) in parts]
            fine = same < np.mean(rest)
            agree.setdefault(w, []).append(fine)
            report.append(f"| {w} | {p1} / {p2} | {same:.3f} | {np.mean(rest):.3f} {'(agree)' if fine else '(disagree)'} |")

    # accept: a target checked by known parts, or agreeing across phrases; the medoid of its accepted cuts is kept
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, report_acc = [], ["\n## Accepted\n"]
    for w in sorted(TARGETS):
        by_known = [ph for ph, good in ok.get(w, []) if good]
        by_agree = sum(agree.get(w, [])) >= max(1, len(agree.get(w, [])) - 1) and agree.get(w)
        chosen = by_known or ([ph for ph, _ in ok.get(w, [])] if by_agree else [])
        if not chosen:
            report_acc.append(f"- {w}: not accepted (cuts did not check out)")
            continue
        cands = [parts[(ph, w)] for ph in chosen]
        k = int(np.argmin([sum(dtw_distance(x, y) for y in cands) for x in cands]))
        sid = f"split_{len(entries):02d}"
        np.save(OUT / "signs" / f"{sid}.npy", cands[k])
        entries.append({"gloss": w, "sign_id": sid, "dataset": "phrase_split", "keypoints_path": f"split/signs/{sid}.npy",
                        "review_status": "pending", "is_religious": True, "is_letter": False})
        report_acc.append(f"- {w}: accepted, from «{chosen[k]}» ({'known parts' if by_known else 'agreement across phrases'})")
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(report + report_acc) + "\n", encoding="utf-8")
    print("\n".join(report + report_acc))


if __name__ == "__main__":
    main()
