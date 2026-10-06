"""Annotation kit for sign boundaries in Isharah sentences (the labels automatic cutting could not replace).

Picks ~200 sentences spread over every signer and sentence length (religious sentences first), renders each
as a skeleton video with the frame number burned in, and writes the task list the annotation page reads.

  python scripts/lexicon/arsl/isharah/prepare_annotation.py [--per-signer 12]

Output: data/lexicon_v2/isharah/annotation/
  videos/<sid>.mp4   skeleton video, 25 fps, frame number top-left
  tasks.js           the task list (loaded by annotate.html; a .js file so it opens from disk)
  annotate.html      copied here, open it in a browser to label
"""
import argparse
import json
import pickle
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from gloss_clips import ISHARAH, PKL, to_contract  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402
from isharati.pose.render import SkeletonRenderer  # noqa: E402

OUT = DATA / "lexicon_v2" / "isharah" / "annotation"
RELIGIOUS = Path(r"D:\islam\signlang\isharah\religious_sentences.tsv")


class NumberedRenderer(SkeletonRenderer):
    def _frame(self, f):
        img = super()._frame(f)
        cv2.putText(img, str(self.i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        self.i += 1
        return img

    def render(self, pose, out_path):
        self.i = 0
        return super().render(pose, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-signer", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(7)
    sentences = {}
    for split in ("train.csv", "dev.csv"):
        for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
            if "|" in line:
                sid, g = line.split("|", 1)
                sentences[sid] = g.split()
    religious = {normalize_ar(l.split("\t")[0].replace("_", " "))
                 for l in RELIGIOUS.read_text(encoding="utf-8").splitlines()[1:] if l.strip()}

    by_signer = defaultdict(list)
    for sid, g in sentences.items():
        if 2 <= len(g) <= 6:
            by_signer[sid.split("_")[0]].append(sid)
    chosen = []
    for signer, sids in sorted(by_signer.items()):
        rel = [s for s in sids if normalize_ar(" ".join(sentences[s]).replace("_", " ")) in religious]
        rest = [s for s in sids if s not in rel]
        rng.shuffle(rel)
        rng.shuffle(rest)
        # a mix of lengths: round-robin over 2..6 glosses
        by_len = defaultdict(list)
        for s in rest:
            by_len[len(sentences[s])].append(s)
        pick = rel[:2]
        while len(pick) < args.per_signer and any(by_len.values()):
            for n in range(2, 7):
                if by_len[n] and len(pick) < args.per_signer:
                    pick.append(by_len[n].pop())
        chosen += pick

    print(f"{len(chosen)} sentences from {len(by_signer)} signers; loading poses", flush=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    renderer = NumberedRenderer(size=480)
    tasks = []
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    for sid in chosen:
        if sid not in data:
            continue
        pose = to_contract(data[sid]["keypoints"])
        renderer.render(pose, OUT / "videos" / f"{sid}.mp4")
        tasks.append({"sid": sid, "signer": sid.split("_")[0], "glosses": sentences[sid],
                      "frames": int(len(pose)), "video": f"videos/{sid}.mp4"})
    (OUT / "tasks.js").write_text("const TASKS = " + json.dumps(tasks, ensure_ascii=False, indent=1) + ";\n",
                                  encoding="utf-8")
    shutil.copy(Path(__file__).parent / "annotate.html", OUT / "annotate.html")
    print(f"{len(tasks)} videos -> {OUT}; open {OUT / 'annotate.html'}")


if __name__ == "__main__":
    main()
