"""Fetch the AUTSL colour videos of the signs the TİD lexicon lacks, a few signers each, without the 13 GB of shards.

AUTSL (Sincan & Keles 2020; 226 signs, 43 signers, 36,302 Kinect clips; academic and research use only, no commercial
use) is mirrored openly on Hugging Face as aipieces/AUTSL: {train,val,test}/shard_NNN_MMM.zip, each ~300 MB with the
_color.mp4 and _depth.mp4 of ~500 samples. A zip can be read from the middle: its central directory sits at the end, so
HTTP range requests list every shard (~80 kB each) and then pull just the wanted _color.mp4 members (~180 kB each).
A shard already on disk (F:/tid_datasets/autsl/raw/<split>_<shard>.zip) is read locally.

Which signs: AUTSL classes whose Turkish gloss (restored from the ASCII-folded class name, gloss_restore.json of
gap_overlap.py) is not yet a TİD lexicon sign; up to --per samples per sign, each from a different signer, spread over
the train/val/test signers.

--missing fetches the AUTSL classes the manifest does not have yet (the signs whose gloss the lexicon already had,
added later as variants, autsl_build.py), merges their rows into the manifest and writes fetch.done when finished, so
a following autsl_track.py --follow / autsl_build.py --follow can stream behind it.

  .venv/Scripts/python scripts/lexicon/tid_datasets/autsl_fetch.py [--per 10] [--all | --missing --per 5]
Output: F:/tid_datasets/autsl/videos/<class_id>/<sample>_color.mp4, F:/tid_datasets/autsl/fetch_manifest.csv
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

BASE = "https://huggingface.co/datasets/aipieces/AUTSL/resolve/main/"
API = "https://huggingface.co/api/datasets/aipieces/AUTSL/tree/main/"
AUTSL = Path("F:/tid_datasets/autsl")
SHARED = Path("F:/tid_datasets/_shared")
SPLITS = {"train": "train_labels.csv", "val": "validation_labels.csv", "test": "test_labels.csv"}
# restorations gap_overlap.py got wrong (checked against the English class names)
FIX = {"sali": "salı", "masallah": "maşallah"}


class HTTPFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests, with a block cache."""
    BLOCK = 1 << 18

    def __init__(self, url, session):
        self.s = session
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url  # the signed CDN address, so later ranges skip the redirect
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo = b * self.BLOCK
            hi = min(self.size, lo + self.BLOCK) - 1
            for _ in range(4):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    r.raise_for_status()
                    self.cache[b] = r.content
                    break
                except requests.RequestException:
                    continue
            else:
                raise IOError(f"range {lo}-{hi} failed")
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(b)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def targets(all_signs):
    """class_id -> (folded name, Turkish gloss, English) for the AUTSL signs to fetch."""
    from isharati.lexicon.turkish import tid_lexicon
    from isharati.retrieval import turkish_urdu as M
    restore = json.loads((SHARED / "gloss_restore.json").read_text(encoding="utf-8"))["AUTSL"]
    lex = tid_lexicon()
    forms = set()
    for k in lex._signs:
        if k and " " not in k:
            forms |= M.tr_forms(k)
    out = {}
    for r in csv.DictReader(open(AUTSL / "SignList_ClassId_TR_EN.csv", encoding="utf-8")):
        g = M.tr_norm(FIX.get(r["TR"], restore[r["TR"]][0])).replace("â", "a")
        have = g in lex._signs if " " in g else bool(M.tr_forms(g) & forms)
        if all_signs or not have:
            out[int(r["ClassId"])] = (r["TR"], g, r["EN"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10, help="samples per sign, one per signer")
    ap.add_argument("--all", action="store_true", help="every AUTSL sign, not just those the lexicon lacks")
    ap.add_argument("--missing", action="store_true", help="every AUTSL sign not yet in fetch_manifest.csv")
    a = ap.parse_args()
    manifest = AUTSL / "fetch_manifest.csv"
    old = list(csv.DictReader(open(manifest, encoding="utf-8"))) if a.missing and manifest.exists() else []
    (AUTSL / "fetch.done").unlink(missing_ok=True)
    want = targets(a.all or a.missing)
    if a.missing:
        want = {c: v for c, v in want.items() if c not in {int(r["class_id"]) for r in old}}
    print(f"{len(want)} AUTSL signs to fetch", flush=True)

    # sample -> (split, class); then choose per class up to --per samples from distinct signers, round-robin over splits
    by_class = defaultdict(lambda: defaultdict(list))
    for split, f in SPLITS.items():
        for sample, cid in csv.reader(open(AUTSL / f, encoding="utf-8")):
            if int(cid) in want:
                signer = sample.split("_")[0]
                by_class[int(cid)][signer].append((split, sample))
    chosen = {}
    for cid, signers in by_class.items():
        order = sorted(signers, key=lambda s: int(re.sub(r"\D", "", s)))
        for s in order[:a.per]:
            split, sample = sorted(signers[s])[0]
            chosen[f"{sample}_color.mp4"] = (cid, split, sample)
    print(f"{len(chosen)} clips chosen", flush=True)

    session = requests.Session()
    shards = []
    for split in SPLITS:
        files = [x["path"] for x in session.get(API + split, timeout=60).json() if x["path"].endswith(".zip")]
        shards += [(split, p) for p in sorted(files)]

    def one(job):
        """The chosen clips in one shard; shards are read in parallel (each range request waits on the CDN)."""
        split, path = job
        out_rows = []
        local = AUTSL / "raw" / f"{split}_{Path(path).name}"
        src = open(local, "rb") if local.exists() else HTTPFile(BASE + path, requests.Session())
        with zipfile.ZipFile(src) as z:
            for n in (n for n in z.namelist() if Path(n).name in chosen):
                cid, sp, sample = chosen[Path(n).name]
                out = AUTSL / "videos" / f"{cid:03d}" / Path(n).name
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".part")
                    tmp.write_bytes(z.read(n))
                    tmp.replace(out)
                out_rows.append({"class_id": cid, "tr": want[cid][0], "gloss": want[cid][1], "en": want[cid][2],
                                 "split": sp, "sample": sample, "signer": sample.split("_")[0],
                                 "video": str(out.relative_to(AUTSL)).replace("\\", "/")})
        print(f"{path}: {len(out_rows)} clips", flush=True)
        return out_rows

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as ex:
        rows = [r for part in ex.map(one, shards) for r in part]
    done = len(rows)
    rows = [{**r, "class_id": int(r["class_id"])} for r in old] + rows
    tmp = manifest.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["class_id"], r["sample"])))
    tmp.replace(manifest)
    (AUTSL / "fetch.done").write_text(str(done))
    print(f"{done} clips -> {AUTSL / 'videos'}")


if __name__ == "__main__":
    main()
