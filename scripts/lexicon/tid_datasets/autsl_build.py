"""AUTSL -> a TİD lexicon source: one recording per sign, the medoid of its signers, with the signer's face.

Input: the Holistic tracks of the fetched clips (autsl_fetch.py, autsl_track.py; ~10 signers per sign).
Per clip:
  - segmentation: AUTSL signers start and end with the hands resting in view, so "signing" is a wrist raised above its
    resting height (the lowest it gets in the clip) by more than RAISE shoulder widths; the clip is cut from the first
    to the last raised frame, PAD frames either side. A clip where neither wrist ever rises is kept whole.
  - quality: the signing hand detected in >= 70 % of the cut, 0.4-4 s long, shoulders seen.
  - the pose clip is normalize(interpolate_missing(raw)) of the cut on the 25 fps grid, like the TİD dictionary clips,
    mirrored (AUTSL's colour video is a mirror image, see MIRRORED).
Per sign, the medoid across signers on handshape and motion: every clip is put in the right-handed frame (a left-
handed signer mirrored), resampled to N frames, and described by the signing hand's path about the neck (motion) and
its 21 joints about the wrist over palm length (handshape). The clip with the least summed distance to the others is
kept, among those signed with the majority hand, so the lexicon never shows a mirrored signer.
The face (scripts/face/attach.py's save_face format, data/face/tr/<sign_id>.npz) is cut from the same track over the
same frames, so it needs no search (match 0).

Variants (rows with "variant_of", which are not glosses: TIDLexicon.add_words and corpus_eda skip them; they are kept
for recognition / ML, and publish.py packs their poses and faces with the rest):
  - beside each sign's medoid, the next NVAR clips of its pool by summed distance, <sign_id>_v2, <sign_id>_v3;
  - an AUTSL sign whose gloss the TİD dictionary already signs (exact key, or ALIAS) is itself a variant of that
    dictionary sign: autsl_NNN gets "variant_of": <dictionary sign_id>, and so do its _v2/_v3. The dictionary is
    the reference because it is stable (tid_youtube/ is rebuilt by its own scraper; a gloss only it has stays a main
    AUTSL row, which loads after it and is shadowed while tid_youtube has it).
Rows already in entries.jsonl are kept as they are (sign ids and files stable); only missing rows are written, each
clip and face atomically, then the row appended by an atomic rewrite of entries.jsonl, so a reader never sees a half
file. --follow streams behind autsl_fetch.py --missing and autsl_track.py --follow: a class is built once all its
fetched clips are tracked, and the run ends when every AUTSL class is done and fetch.done exists.

  PYTHONPATH=src .venv/Scripts/python scripts/lexicon/tid_datasets/autsl_build.py [--follow]
Output: data/lexicon_tr/autsl/{signs/*.npy, entries.jsonl, report.json}, data/face/tr/autsl_*.npz (+ manifest rows),
        F:/tid_datasets/autsl/qa/*.png (contact sheets of 10 random signs first built in the run)
"""
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
from isharati.config import DATA  # noqa: E402
from isharati.pose import proportions  # noqa: E402
from isharati.pose.keypoints import L_SH, L_WR, LH, NECK, R_SH, R_WR, RH, interpolate_missing, normalize  # noqa: E402
from attach import fifty, window_frame  # noqa: E402


def mirror_face(f, names):
    """scripts/lexicon/arsl/fingerspelled.py's: x -> -x, and every ...Left blendshape swapped with its ...Right."""
    blend, face = np.array(f["blend"], copy=True), np.array(f["face"], np.float32, copy=True)
    idx = {n: k for k, n in enumerate(names)}
    for n, k in idx.items():
        if n.endswith("Left") and n[:-4] + "Right" in idx:
            j = idx[n[:-4] + "Right"]
            blend[:, [k, j]] = blend[:, [j, k]]
    face[..., 0] *= -1
    return blend, face

AUTSL = Path("F:/tid_datasets/autsl")
OUT = DATA / "lexicon_tr" / "autsl"
FACE = DATA / "face" / "tr"
DATASET = "autsl"
RAISE, PAD, N = 0.25, 3, 24
MIN_HAND, MIN_T, MAX_T = 0.70, 10, 100  # hands seen; 0.4-4 s at 25 fps
# signs whose AUTSL meaning is not the corpus's sense of the word: left out of the lexicon
EXCLUDE = {"söz": "AUTSL's sign means 'promise'; in the Qur'an/hadith corpus söz is mostly 'word, saying'",
           "ilgilenmemek": "a negated verb: by its five-letter stem it would sign ilgili, ilgilenmek as 'not interested'"}
ISLAMIC = {"bayram", "helal", "melek", "oruç", "şeytan", "maşallah"}
NVAR = 2  # variants kept beside the medoid
ALIAS = {"teşekkür": "teşekkürler"}  # AUTSL's "thanks" is the dictionary's teşekkürler (not a key by itself)
SWAP = [(2, 5), (3, 6), (4, 7)]  # R_SH/L_SH, R_EL/L_EL, R_WR/L_WR
# MIRRORED: the Kinect v2 colour stream is a mirror image. MediaPipe then reads a right-handed signer as left-handed:
# 84 % of AUTSL clips sign with the "left" hand where the TİD dictionary's (ordinary video) sign 87 % with the right.
# Every AUTSL clip, and its face, is mirrored back.


def mirror(clip):
    out = clip.copy()
    out[..., 0] *= -1
    for a, b in SWAP:
        out[:, [a, b]] = out[:, [b, a]]
    out[:, LH], out[:, RH] = out[:, RH].copy(), out[:, LH].copy()
    return out


def signing_hand(clip):
    """The hand whose wrist rises highest (image y points down)."""
    return "right" if clip[:, R_WR, 1].min() < clip[:, L_WR, 1].min() else "left"


def cut(raw):
    """(start, end) of the signing in a raw [T,50,3] track, by raised wrists."""
    width = np.nanmedian(np.linalg.norm(raw[:, R_SH, :2] - raw[:, L_SH, :2], axis=-1))
    raised = np.zeros(len(raw), bool)
    for wr in (R_WR, L_WR):
        y = raw[:, wr, 1]
        if np.isfinite(y).any():
            raised |= np.nan_to_num(y, nan=np.nanmax(y)) < np.nanmax(y) - RAISE * width
    idx = np.where(raised)[0]
    if not len(idx):
        return 0, len(raw)
    return max(0, idx[0] - PAD), min(len(raw), idx[-1] + 1 + PAD)


def clip_from_track(path):
    z = np.load(path)
    track = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    meta = json.loads(str(z["meta"]))
    raw = fifty(track)
    if np.isnan(raw[:, R_SH, 0]).all():
        return None
    s, e = cut(raw)
    filled, _ = interpolate_missing(raw[s:e])
    clip = mirror(normalize(filled))  # AUTSL's Kinect colour video is mirror-imaged (see MIRRORED)
    hand = LH if signing_hand(clip) == "right" else RH  # its slot in the unmirrored track
    seen = float(np.mean(~np.isnan(raw[s:e, hand, 0]).all(axis=1)))
    return {"clip": clip, "raw": raw, "track": track, "meta": meta, "s": s, "e": e, "hand_seen": seen,
            "hand": signing_hand(clip)}


def features(clip):
    """Right-handed, N frames: signing-hand path about the neck, and its handshape about the wrist."""
    c = mirror(clip) if signing_hand(clip) == "left" else clip
    idx = np.linspace(0, len(c) - 1, N).round().astype(int)
    c = c[idx]
    h = c[:, RH, :2]
    motion = h[:, 0] - c[:, NECK, :2]
    palm = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1, keepdims=True) + 1e-6
    shape = (h - h[:, :1]) / palm[..., None]
    return motion, shape


def medoid(clips):
    feats = [features(c) for c in clips]
    n = len(feats)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = np.linalg.norm(feats[i][0] - feats[j][0], axis=-1).mean()
            s = np.linalg.norm(feats[i][1] - feats[j][1], axis=-1).mean()
            d[i, j] = d[j, i] = m + 0.5 * s  # shoulder widths of path + palm lengths of shape (~same scale)
    return int(np.argmin(d.sum(1))), d


def save_face(sid, c, clip):
    s, e = c["s"], c["e"]
    _, neck, scale = window_frame(c["raw"][s:e])
    face = (c["track"]["face"][s:e].astype(np.float32) - neck) / scale
    names = list(c["meta"]["blend_names"])
    blend, face = mirror_face({"blend": c["track"]["blend"][s:e], "face": face}, names)  # as the clip
    t = c["track"]["time"]
    row = {"sign_id": sid, "dataset": DATASET, "video": Path(c["meta"]["video"]).name,
           "start": round(float(t[s]), 3), "end": round(float(t[e - 1]), 3), "frames": int(e - s), "match": 0.0,
           "found": True, "face_detected": round(float(np.mean(~np.isnan(blend[:, 0]))), 3)}
    tmp = FACE / f"{sid}.npz.part"  # not *.npz: publish.py globs those
    with open(tmp, "wb") as f:
        np.savez_compressed(f, blend=blend, face=face.astype(np.float16),
                            blend_names=np.array(c["meta"]["blend_names"]), aspect=c["meta"]["aspect"],
                            ratio=proportions.ratio(clip), match=0.0, start=row["start"], end=row["end"],
                            video=row["video"])
    os.replace(tmp, FACE / f"{sid}.npz")
    return row


def save_clip(sid, clip):
    tmp = OUT / "signs" / f"{sid}.npy.part"
    with open(tmp, "wb") as f:
        np.save(f, clip.astype(np.float32))
    os.replace(tmp, OUT / "signs" / f"{sid}.npy")


def write_lines(path, rows):
    """Atomic rewrite of a jsonl file: a reader sees the old file or the new one, never half of it."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def read_lines(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.exists() else []


def contact_sheet(sign, cands, best, path, variants=()):
    """Top: a frame of each candidate video at mid-signing (chosen one starred in red, variants v2/v3 in blue).
    Bottom: the chosen clip's skeleton."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(cands)
    fig, ax = plt.subplots(2, max(k, 6), figsize=(2.2 * max(k, 6), 4.6))
    for a in ax.ravel():
        a.axis("off")
    for i, c in enumerate(cands):
        cap = cv2.VideoCapture(str(c["meta"]["video"]))
        mid = (c["track"]["time"][c["s"]] + c["track"]["time"][c["e"] - 1]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
        ok, img = cap.read()
        cap.release()
        if ok:
            ax[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tag = "* " if i == best else f"v{list(variants).index(i) + 2} " if i in variants else ""
        ax[0, i].set_title(tag + f"{c['signer']} {c['e'] - c['s']}f {c['hand_seen']:.0%}", fontsize=7,
                           color="red" if i == best else "blue" if i in variants else "black")
    clip = cands[best]["clip"]
    for j, t in enumerate(np.linspace(0, len(clip) - 1, max(k, 6)).round().astype(int)):
        f = clip[t]
        a = ax[1, j]
        a.scatter(f[:8, 0], -f[:8, 1], s=8, c="k")
        a.scatter(f[LH, 0], -f[LH, 1], s=3, c="b")
        a.scatter(f[RH, 0], -f[RH, 1], s=3, c="r")
        a.set_xlim(-2, 2)
        a.set_ylim(-2.5, 1.5)
        a.set_title(f"t={t}", fontsize=7)
    fig.suptitle(f"{sign['gloss']} ({sign['en']}), AUTSL class {sign['class_id']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def corpus_freq():
    """gloss -> how many Turkish corpus tokens start with it."""
    from collections import Counter
    from isharati.retrieval import turkish_urdu as M
    ct = Counter()
    for line in open(DATA / "tr" / "corpus.jsonl", encoding="utf-8"):
        ct.update(w for w in M.tr_norm(json.loads(line)["text"]).split() if w.isalpha())
    return lambda g: sum(n for w, n in ct.items() if w.startswith(M.tr_norm(g)))


def dictionary_signs():
    """Turkish key -> (sign_id, gloss) of the TİD dictionary signs (what an AUTSL variant points to)."""
    from isharati.lexicon.turkish import TID_FILE, TID_ROOT
    from isharati.text.turkish import tr_norm
    out = {}
    for r in read_lines(TID_ROOT / TID_FILE):
        if not r.get("is_letter"):
            out.setdefault(tr_norm(r["gloss"]), (r["sign_id"], r["gloss"]))
    return out


def candidates(rs):
    """The clips of one sign that pass QA, and the rejected ones with the reason."""
    cands, rejected = [], []
    for r in rs:
        p = AUTSL / "tracks" / f"{r['sample']}.npz"
        if (AUTSL / "tracks" / f"{r['sample']}.failed").exists():
            rejected.append((r["sample"], "tracking failed"))
            continue
        c = clip_from_track(p) if p.exists() else None
        if c is None:
            rejected.append((r["sample"], "no track/body"))
            continue
        T = c["e"] - c["s"]
        if c["hand_seen"] < MIN_HAND:
            rejected.append((r["sample"], f"hand seen {c['hand_seen']:.2f}"))
        elif not MIN_T <= T <= MAX_T:
            rejected.append((r["sample"], f"{T} frames"))
        else:
            c["signer"], c["sample"] = r["signer"], r["sample"]
            cands.append(c)
    return cands, rejected


class Build:
    def __init__(self):
        (OUT / "signs").mkdir(parents=True, exist_ok=True)
        FACE.mkdir(parents=True, exist_ok=True)
        self.entries = read_lines(OUT / "entries.jsonl")
        self.have = {e["sign_id"]: e for e in self.entries}
        rp = OUT / "report.json"
        self.report = {int(k): v for k, v in json.loads(rp.read_text(encoding="utf-8")).items()} if rp.exists() else {}
        self.faces, self.sheets, self.new_classes = [], {}, []
        self.dict = dictionary_signs()

    def sign(self, cid, rs):
        from isharati.text.turkish import tr_norm
        sign = {"class_id": cid, "gloss": rs[0]["gloss"], "en": rs[0]["en"], "tr": rs[0]["tr"]}
        if sign["gloss"] in EXCLUDE:
            self.report[cid] = {**sign, "kept": None, "excluded": EXCLUDE[sign["gloss"]]}
            return
        cands, rejected = candidates(rs)
        info = {**self.report.get(cid, {}), **sign, "candidates": len(cands), "rejected": rejected}
        if not cands:
            self.report[cid] = {**info, "kept": None}
            print(f"{cid:3d} {sign['gloss']:16s} no usable clip ({len(rejected)} rejected)", flush=True)
            return
        hands = [c["hand"] for c in cands]
        main_hand = max(set(hands), key=hands.count)
        pool = [c for c in cands if c["hand"] == main_hand]
        if len(pool) >= 2:
            b, d = medoid([c["clip"] for c in pool])
            order = [int(i) for i in np.argsort(d.sum(1), kind="stable")]
            spread = float(np.median(d[b][np.arange(len(pool)) != b]))
        else:
            b, order, spread = 0, [0], None
        sid = f"autsl_{cid:03d}"
        if sid in self.have:  # built before: that clip stays the main one
            kept = self.have[sid]["source"].rsplit(", ", 1)[-1]
            at = [i for i, c in enumerate(pool) if c["sample"] == kept]
            if at and at[0] != b:
                print(f"{cid:3d} medoid moved ({pool[b]['sample']}), keeping {kept}", flush=True)
            b = at[0] if at else b
        variants = [i for i in order if i != b][:NVAR]
        key = ALIAS.get(tr_norm(sign["gloss"]), tr_norm(sign["gloss"]))
        target = self.dict.get(key)
        ids = [(sid, b, None)] + [(f"{sid}_v{k + 2}", i, k + 2) for k, i in enumerate(variants)]
        wrote = []
        for vid, i, rank in ids:
            if vid in self.have:
                continue
            c = pool[i]
            row = {"gloss": sign["gloss"], "sign_id": vid, "dataset": DATASET,
                   "keypoints_path": f"autsl/signs/{vid}.npy", "review_status": "pending",
                   "is_religious": sign["gloss"] in ISLAMIC, "is_letter": False,
                   "source": f"AUTSL class {cid} ({sign['tr']}, {sign['en']}), {c['sample']}",
                   "signer": c["signer"], "medoid_of": len(pool)}
            if rank or target:
                # not a gloss: the sign the lexicon uses for this word (the dictionary's, or this AUTSL main clip)
                row["variant_of"] = target[0] if target else sid
                row["variant_rank"] = rank or 1
                if target:
                    row["variant_gloss"] = target[1]
            save_clip(vid, c["clip"])
            f = save_face(vid, c, c["clip"])
            f["gloss"] = sign["gloss"]
            self.faces.append(f)
            self.entries.append(row)
            self.have[vid] = row
            write_lines(OUT / "entries.jsonl", self.entries)  # streamed: the lexicon can be read mid-run
            wrote.append(vid)
        if sid in wrote:
            self.new_classes.append(cid)
        self.report[cid] = {**info, "kept": pool[b]["sample"], "medoid_of": len(pool), "hand": main_hand,
                            "frames": int(pool[b]["e"] - pool[b]["s"]), "hand_seen": round(pool[b]["hand_seen"], 3),
                            "median_distance": None if spread is None else round(spread, 3),
                            "variants": [pool[i]["sample"] for i in variants],
                            "variant_of": target[0] if target else None}
        self.sheets[cid] = (sign, pool, b, variants)
        print(f"{cid:3d} {sign['gloss']:16s} {len(pool)}/{len(rs)} clips, main {pool[b]['sample']}"
              f"{' (variant of ' + target[1] + ' ' + target[0] + ')' if target else ''}, +{len(variants)} variants, "
              f"wrote {wrote}", flush=True)

    def save(self, final=False):
        if final:
            # most frequent in the corpus first: with stems, the first gloss of a five-letter prefix takes the words that
            # only match by prefix (bayramı: bayram, not bayrak). Variant rows (not glosses) after, by sign id.
            freq = corpus_freq()
            mains = sorted((e for e in self.entries if not e.get("variant_of")), key=lambda e: -freq(e["gloss"]))
            var = sorted((e for e in self.entries if e.get("variant_of")), key=lambda e: e["sign_id"])
            self.entries = mains + var
            write_lines(OUT / "entries.jsonl", self.entries)
        tmp = OUT / "report.json.part"
        tmp.write_text(json.dumps({str(k): v for k, v in sorted(self.report.items())}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, OUT / "report.json")
        if self.faces:
            # the face manifest is shared with other TİD sources: re-read it now and replace only these sign ids
            path = FACE / "manifest.jsonl"
            ids = {f["sign_id"] for f in self.faces}
            write_lines(path, [m for m in read_lines(path) if m.get("sign_id") not in ids] + self.faces)


def tracked(rs):
    t = AUTSL / "tracks"
    return all((t / f"{r['sample']}.npz").exists() or (t / f"{r['sample']}.failed").exists() for r in rs)


def main():
    follow = "--follow" in sys.argv
    b = Build()
    pending = {int(r["ClassId"]) for r in csv.DictReader(open(AUTSL / "SignList_ClassId_TR_EN.csv", encoding="utf-8"))}
    while pending:
        fetched = (AUTSL / "fetch.done").exists()  # read before the manifest: once set, the manifest is complete
        by_class = defaultdict(list)
        for r in csv.DictReader(open(AUTSL / "fetch_manifest.csv", encoding="utf-8")):
            by_class[int(r["class_id"])].append(r)
        n = len(pending)
        for cid in sorted(pending):
            rs = by_class.get(cid)
            if not rs:
                if fetched or not follow:
                    b.report.setdefault(cid, {"class_id": cid, "kept": None, "rejected": [], "note": "no clip fetched"})
                    pending.discard(cid)
                continue
            if follow and not tracked(rs):
                continue
            b.sign(cid, rs)
            pending.discard(cid)
        if len(pending) != n:
            b.save()
        if pending:
            print(f"waiting: {len(pending)} classes", flush=True)
            time.sleep(30)
    b.save(final=True)
    qa = AUTSL / "qa"
    qa.mkdir(exist_ok=True)
    random.seed(0)
    picks = random.sample(sorted(b.new_classes), min(10, len(b.new_classes)))
    for cid in picks:
        sign, pool, best, variants = b.sheets[cid]
        contact_sheet(sign, pool, best, qa / f"{cid:03d}_{sign['tr']}.png", variants)
    mains = sum(not e.get("variant_of") for e in b.entries)
    print(f"{len(b.entries)} rows ({mains} glosses, {len(b.entries) - mains} variants) -> {OUT}; "
          f"{len(b.faces)} new faces -> {FACE}; QA sheets {picks} -> {qa}")


if __name__ == "__main__":
    main()
