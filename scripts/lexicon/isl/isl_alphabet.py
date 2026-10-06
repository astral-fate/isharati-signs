"""The ISL manual alphabet (A-Z) from ISLRTC's «Alphabets A Z in ISL» video -> letter signs of the ISL lexicon.

Source: https://youtu.be/VdwKSyza5oI (Indian Sign Language Research and Training Centre, Govt of India, 2023-09-19;
143 s, one signer; no open licence stated, standard YouTube terms: research use, ask ISLRTC before redistributing).
Every letter is shown under a large caption of the letter (top left), A to Z in order; C, E, I, J, T, U, V and Z show
two or three variants («(Sign 1)», «(Sign 2)») under the same letter. Segmentation:
  1. caption intervals: the white caption pixels in the top-left box; a new interval starts when the caption disappears
     or changes (the caption mask's IoU with the interval's first mask under 0.5); intervals under 0.4 s are dropped.
     The video gives exactly 26, which are A..Z in order (checked: the build stops otherwise);
  2. inside each interval, the first signing run (islamic_terms.signing_runs: a hand above waist level) is the letter,
     i.e. «(Sign 1)» for letters with variants; the whole run is kept (the hold of a static letter, the whole movement of
     J, Z and the other moving letters), with 2 frames of padding;
  3. the same checks as the ISLRTC words (0.4-4 s, dominant hand in 70% of frames, one signing run).
ISL letters are mostly two-handed: both hands are kept as tracked, nothing is mirrored.

The video goes once through the MediaPipe Holistic task (scripts/face/track.py); each letter's face, for the same
frames as its clip, is saved as data/face/ur/islalpha_<L>.npz (scripts/face/attach.py's format).

  "<gathring data>/.venv/Scripts/python" scripts/face/track.py F:/isl_new/alphabet/videos/VdwKSyza5oI.mp4 F:/isl_new/alphabet/tracks/VdwKSyza5oI.npz
  PYTHONPATH=src py scripts/lexicon/isl/isl_alphabet.py build      # letters, entries, faces
  PYTHONPATH=src py scripts/lexicon/isl/isl_alphabet.py qa         # contact sheets, skeleton videos, spelled test words
Output: data/lexicon_isl/alphabet/{signs/<L>.npy, entries.jsonl}, data/face/ur/islalpha_<L>.npz; QA in
F:/isl_new/alphabet/qa/.
"""
import json
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
from isharati.config import DATA  # noqa: E402

VID = "VdwKSyza5oI"
URL = f"https://youtu.be/{VID}"
WORK = Path(r"F:\isl_new\alphabet")
VIDEO, TRACK = WORK / "videos" / f"{VID}.mp4", WORK / "tracks" / f"{VID}.npz"
OUT = DATA / "lexicon_isl" / "alphabet"
FACE = DATA / "face" / "ur"
LICENCE = "ISLRTC (Govt of India) YouTube channel; no open licence stated, standard YouTube terms; research use"
BOX = (slice(90, 380), slice(40, 430))  # the caption's box in the 1280x720 frame (rows, columns)
HOLD, TRANSITION = 8, 5                 # spelled test words: frames per letter hold and per transition


def caption_intervals():
    """[(start_s, end_s)] of the letter captions, as described in the module docstring."""
    import cv2
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    segs, cur, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m = (f[BOX] > 190).all(axis=2)
        if m.mean() < 0.004:
            if cur:
                segs.append(cur)
            cur = None
        else:
            if cur is not None and (m & cur[2]).sum() / max(1, (m | cur[2]).sum()) < 0.5:
                segs.append(cur)
                cur = None
            cur = [i, i, m] if cur is None else [cur[0], i, cur[2]]
        i += 1
    if cur:
        segs.append(cur)
    cap.release()
    return [(s / fps, e / fps) for s, e, _ in segs if (e - s) / fps >= 0.4]


def build():
    from attach import fifty, save_face
    from islrtc import MAX_S, MIN_HAND, MIN_S, PAD, signing_runs
    from isharati.pose.keypoints import LH, RH, interpolate_missing, normalize
    caps = caption_intervals()
    if len(caps) != 26:
        sys.exit(f"{len(caps)} caption intervals, expected 26: check the caption box")
    z = np.load(TRACK)
    meta = json.loads(str(z["meta"]))
    tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    raw, t = fifty(tr), tr["time"]
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    rows, report = [], []
    for letter, (c0, c1) in zip(string.ascii_uppercase, caps):
        i0, i1 = int(np.searchsorted(t, c0)), int(np.searchsorted(t, c1))
        runs = [(a + i0, b + i0) for a, b in signing_runs(raw[i0:i1]) if b - a >= 6]
        rec = {"letter": letter, "caption": [round(c0, 2), round(c1, 2)], "runs": len(runs)}
        if not runs:
            rec["reason"] = "no signing run under the caption"
            report.append(rec)
            continue
        a, b = runs[0]
        a, b = max(0, a - PAD), min(len(raw), b + PAD)
        seg = raw[a:b]
        lh, rh = ~np.isnan(seg[:, LH, 0]).all(1), ~np.isnan(seg[:, RH, 0]).all(1)
        dom = max(lh.mean(), rh.mean())
        dur = (b - a) / 25
        inner = len([r for r in signing_runs(seg) if r[1] - r[0] >= 6])
        filled, missing = interpolate_missing(seg)
        rec.update(start=round(float(t[a]), 3), end=round(float(t[b - 1]), 3), frames=b - a,
                   left=round(float(lh.mean()), 2), right=round(float(rh.mean()), 2),
                   hands="both" if min(lh.mean(), rh.mean()) >= 0.5 else ("right" if rh.mean() > lh.mean() else "left"))
        reason = (f"duration {dur:.2f} s" if not MIN_S <= dur <= MAX_S else
                  f"dominant hand {dom:.0%}" if dom < MIN_HAND else
                  f"{inner} signing runs" if inner != 1 else None)
        if reason:
            rec["reason"] = reason
            report.append(rec)
            continue
        clip = normalize(filled).astype(np.float32)
        np.save(OUT / "signs" / f"{letter}.npy", clip)
        sid = f"islalpha_{letter}"
        row = {"start": rec["start"], "end": rec["end"], "video": VIDEO.name}
        save_face(FACE, sid, tr, raw, meta, a, b - a, 0.0, row, clip)
        rec["face_detected"] = row["face_detected"]
        rows.append({"gloss": letter, "sign_id": sid, "dataset": "islrtc_isl_alphabet",
                     "keypoints_path": f"alphabet/signs/{letter}.npy", "review_status": "pending",
                     "is_religious": False, "is_letter": True})
        report.append(rec)
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (WORK / "letters.json").write_text(json.dumps({"source": URL, "licence": LICENCE, "letters": report}, indent=1),
                                       encoding="utf-8")
    for r in report:
        print(r)
    print(f"{len(rows)}/26 letters -> {OUT / 'entries.jsonl'}")


def spell(word, lex):
    """A fingerspelled word from the letter clips, as scripts/lexicon/arsl/fingerspelled.py builds one (each letter's
    stillest HOLD frames in its middle 80%, smoothstep transitions), but with both hands as signed: ISL letters are
    two-handed, so no letter is mirrored onto one hand."""
    from isharati.pose.stitch import _transition
    parts, used = [], []
    info = json.loads((WORK / "letters.json").read_text(encoding="utf-8"))
    for ch in word:
        e = lex._letters[ch.lower()]
        c = lex.keypoints(e)
        hands = np.concatenate([c[:, 8:29, :2], c[:, 29:50, :2]], axis=1)
        sp = np.r_[0, np.linalg.norm(np.diff(hands, axis=0), axis=-1).mean(-1)]
        lo, hi = int(0.1 * len(c)), max(int(0.9 * len(c)) - HOLD, int(0.1 * len(c)))
        s = min(range(lo, hi + 1), key=lambda k: sp[k:k + HOLD].sum()) if len(c) > HOLD else 0
        seg = c[s:s + HOLD].astype(np.float32)
        if parts:
            parts.append(_transition(parts[-1][-1], seg[0], TRANSITION))
        parts.append(seg)
        used.append(next(r["hands"] for r in info["letters"] if r["letter"] == ch.upper()))
    return np.concatenate(parts), used


def qa():
    import cv2
    sys.path.insert(0, str(Path(__file__).parent))
    from islrtc import contact_sheet
    from isharati.lexicon.isl import isl_lexicon
    from isharati.pose.render import SkeletonRenderer
    q = WORK / "qa"
    q.mkdir(parents=True, exist_ok=True)
    info = json.loads((WORK / "letters.json").read_text(encoding="utf-8"))["letters"]
    ok = [r for r in info if "reason" not in r]
    # all letters in order: the middle frame of each kept span, labelled
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for r in ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, (r["start"] + r["end"]) / 2 * 1000)
        _, img = cap.read()
        img = cv2.resize(img[:, 280:1000], (240, 240))
        cv2.rectangle(img, (0, 0), (240, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{r['letter']}  {r['hands']}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    while len(tiles) % 7:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[k:k + 7], axis=1) for k in range(0, len(tiles), 7)], axis=0)
    cv2.imwrite(str(q / "alphabet_A-Z.jpg"), sheet)
    for r in ok:  # per letter: before / start / middle / end / after the kept span
        s, e = r["start"], r["end"]
        contact_sheet(VIDEO, [s - 0.24, s, (s + e) / 2, e, e + 0.24], ["before", "start", "middle", "end", "after"],
                      q / f"letter_{r['letter']}.jpg", f"{r['letter']} | caption {r['caption']} | kept {s:.2f}-{e:.2f}s "
                      f"| hands {r['hands']} (L {r['left']:.0%}, R {r['right']:.0%})")
    lex = isl_lexicon()
    for L in "AJZHY":
        SkeletonRenderer().render(lex.keypoints(lex._letters[L.lower()]), q / f"skeleton_letter_{L}.mp4")
    for word in ("ALI", "AISHA"):
        clip, used = spell(word, lex)
        SkeletonRenderer().render(clip, q / f"spelled_{word}.mp4")
        print(word, f"{len(clip)} frames;", ", ".join(f"{ch}: {h}" for ch, h in zip(word, used)))
    print("->", q)


if __name__ == "__main__":
    {"build": build, "qa": qa}[sys.argv[1]]()
