"""One signer's complete fingerspelling alphabet as letter signs (is_letter) for the ASL and TİD lexicons.

Sources (videos, tracks and QA sheets kept on F:\\alphabets\\<lang>\\):
  asl  «The ASL Alphabet | ASL - American Sign Language - ABCs», ASL THAT, https://www.youtube.com/watch?v=tkMg8g8vVUo
       one signer, frontal, A-Z with the uploader's English closed captions (one caption per letter); the signer fingerspells
       with the right hand and keeps it up between letters. Cropped to the signer (the left third shows two close-up insets).
  tid  «Türk İşaret Dili Alfabesi», Nilüfer Özel Eğitim Meslek Lisesi, https://www.youtube.com/watch?v=9X2Px4afMBk
       one signer, frontal, the 29 Turkish letters then W X Q, each under its own coloured caption; she raises her hands
       for each letter and lowers them after. Right hand dominant; most letters are two-handed.
Both are under the standard YouTube licence (no Creative Commons): research use, attribution kept here and in
alphabet/sources.json; not for redistribution of the video.

Per letter:
  1. its caption interval: the VTT cue (asl) or the run of one caption image in the corner (tid, scripts in
     F:\\alphabets\\tools\\tid_captions2.py; every interval checked on F:\\alphabets\\tid\\qa\\caption_intervals.jpg);
  2. inside it, the hold: the longest stretch where the hand(s) move less than STILL shoulder widths per frame
     (a 3-frame median), at most MAX_HOLD frames around its middle;
     a motion letter keeps the whole movement instead: ASL J and Z by windows checked frame by frame
     (qa/detail_J.jpg, qa/detail_Z.jpg); TİD Ç Ğ İ J Ö Ş Ü from the moment the raised right hand settles to the moment
     it starts down (the left hand moves meanwhile: the diacritic, or J's trace);
  3. the clip is normalize(interpolate_missing(raw)) of the 25 fps Holistic track (scripts/face/track.py), like every
     other lexicon clip, and the face of the same frames goes to data/face/<asl|tr>/<sign_id>.npz (attach.save_face).
Checks: the signing hand(s) detected in >= 70% of the clip's frames, a plausible length, every letter exactly once and
in order, no two clips overlapping. QA sheet: qa/alphabet_sheet.jpg, one labelled tile per letter in alphabet order.

  "<gathring data>/.venv/Scripts/python" scripts/lexicon/alphabet/build.py asl|tid
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
from attach import fifty, save_face  # noqa: E402
from isharati.config import DATA  # noqa: E402
from isharati.pose.keypoints import L_SH, LH, R_SH, RH, interpolate_missing, normalize  # noqa: E402

STILL, MAX_HOLD, MIN_HOLD, MAX_MOTION, MIN_DETECT = 0.0125, 25, 8, 50, 0.70
F = Path(r"F:\alphabets")

TID_BOUNDS = [0.48, 3.76, 7.20, 11.04, 14.32, 18.20, 27.32, 30.72, 38.40, 46.00, 50.08, 53.68, 56.80, 60.36, 64.56,
              67.80, 73.32, 77.48, 80.72, 85.24, 91.68, 97.88, 103.20, 109.44, 115.72, 119.60, 123.24, 127.00, 130.36,
              134.48, 139.16, 142.64, 147.08]  # caption changes (N and Ö each had one flicker inside, merged)
TID_LETTERS = "A B C Ç D E F G Ğ H I İ J K L M N O Ö P R S Ş T U Ü V Y Z W X Q".split()
TID_ASCII = {"Ç": "c_cedilla", "Ğ": "g_breve", "I": "i_dotless", "İ": "i_dot", "Ö": "o_umlaut", "Ş": "s_cedilla",
             "Ü": "u_umlaut"}

LANGS = {
    "asl": dict(
        video="tkMg8g8vVUo", url="https://www.youtube.com/watch?v=tkMg8g8vVUo", channel="ASL THAT",
        title="The ASL Alphabet | ASL - American Sign Language - ABCs", uploaded="2013-11-09",
        licence="Standard YouTube Licence (no CC); used for non-commercial research with attribution",
        lex=DATA / "asl" / "lexicon", faces=DATA / "face" / "asl", prefix="aslabc_", dataset="aslthat_alphabet",
        crop=(460, 0, 1280, 720), hands=("right",), letters=None,  # from the VTT captions
        motion={"J": (36.36, 37.50), "Z": (87.90, 88.90)},  # I-hand dropping and turning; index tracing the Z
        windows={}, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "tid": dict(
        video="9X2Px4afMBk", url="https://www.youtube.com/watch?v=9X2Px4afMBk",
        channel="Nilüfer Özel Eğitim Meslek Lisesi", title="Türk İşaret Dili Alfabesi", uploaded="2019-11-11",
        licence="Standard YouTube Licence (no CC); used for non-commercial research with attribution",
        lex=DATA / "lexicon_tr", faces=DATA / "face" / "tr", prefix="tidabc_", dataset="nilufer_tid_alphabet",
        crop=None, hands=("right", "left"),
        letters=list(zip(TID_LETTERS, TID_BOUNDS, TID_BOUNDS[1:])),
        motion={k: None for k in "Ç Ğ İ J Ö Ş Ü".split()},  # window found from the track (settle -> start down)
        # E, G and Ğ are each shown twice in a different form; the first form is kept (E: flat hand crossed by the
        # index, as Hayata Ses Ver and Saide Akbulut also sign it; G: stacked fists; Ğ: thumb up on the fist)
        windows={"E": (18.20, 22.60), "G": (30.72, 34.20), "Ğ": (38.40, 41.40)},
        variants={"E": (23.0, 27.2), "G": (34.4, 38.2), "Ğ": (41.6, 45.9)},
        alphabet="ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"),
}


def vtt_letters(path):
    cues = re.findall(r"(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)\s*\n(.+)", path.read_text(encoding="utf-8"))
    sec = lambda h, m, s: int(h) * 3600 + int(m) * 60 + float(s)
    return [(txt.strip(), sec(*c[:3]), sec(*c[3:6])) for *c, txt in cues]


def runs(mask):
    """[start, end) of each run of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def hand_speed(filled, sw, hand):
    pts = filled[:, RH if hand == "right" else LH, :2]
    sp = np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=-1).mean(-1)] / sw
    pad = np.r_[sp[:1], sp, sp[-1:]]
    return np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)  # 3-frame median


def seen_speed(filled, sw, hand, det):
    """hand_speed, infinite where the hand was not seen in this frame and the one before (an undetected stretch is
    interpolated flat, which would read as perfectly still)."""
    sp = hand_speed(filled, sw, hand)
    seen = det & np.r_[False, det[:-1]]
    return np.where(seen, sp, np.inf)


def segment(cfg, letter, i0, i1, raw, filled, sw, det):
    """[s, e) frames of the letter inside frames [i0, i1) of the track, and its kind."""
    R, L = det["right"], det["left"]
    if letter in cfg["motion"] and cfg["motion"][letter] is not None:
        return cfg["_frame"](cfg["motion"][letter][0]), cfg["_frame"](cfg["motion"][letter][1]), "motion"
    sp_r = hand_speed(filled, sw, "right")
    if cfg["crop"] is None:  # TİD: the hands are raised for the letter and lowered after: the right hand's active run
        act = max(runs(R[i0:i1]), key=lambda r: r[1] - r[0])
        a, b = i0 + act[0], i0 + act[1]
        if letter in cfg["motion"]:  # from the raised hands settling to their going down: the right hand still,
            # the left one moving at signing speed, not raising or lowering
            ok = seen_speed(filled, sw, "right", R)[a:b] < 2 * STILL
            if L[a:b].mean() > 0.15:
                ok &= seen_speed(filled, sw, "left", L)[a:b] < 4 * STILL
            settled = np.where(ok)[0]
            s, e = a + settled[0], a + settled[-1] + 1
            return s, min(e, s + MAX_MOTION), "motion"  # Ş repeats its shake for 4 s: two seconds keep the movement
        both = L[a:b].mean() > 0.15  # M and N: the left hand is two-handed but often missed
        sp = np.maximum(sp_r, hand_speed(filled, sw, "left")) if both else sp_r
    else:
        a, b, sp = i0, i1, sp_r
    calm = sp[a:b] < STILL
    if cfg["crop"] is None and both:  # a two-handed letter: only frames where both hands were seen
        calm &= L[a:b]
    still = [(a + s, a + e) for s, e in runs(calm)]
    s, e = max(still, key=lambda r: r[1] - r[0]) if still else (a, a)
    if e - s < MIN_HOLD:  # no long still run: the stillest 12 frames
        n = min(12, b - a)
        missed = (~L[a:b] if cfg["crop"] is None and both else np.zeros(b - a, bool)).astype(float)
        s = min(range(a, b - n + 1), key=lambda k: sp[k:k + n].sum() + missed[k - a:k - a + n].sum())  # seen hands first
        e = s + n
    if e - s > MAX_HOLD:
        m = (s + e) // 2
        s, e = m - MAX_HOLD // 2, m - MAX_HOLD // 2 + MAX_HOLD
    return s, e, "static"


def tile(cap, fps, t_sec, crop, label, kind, w=180, h=200):
    def grab(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        x0, y0, x1, y1 = crop
        return f[y0:y1, x0:x1]
    frames = [grab(t) for t in t_sec]
    if len(frames) == 1:
        img = cv2.resize(frames[0], (w, h))
    else:  # motion letter: start, middle and end overlaid, so the path of the hand shows
        img = np.mean([cv2.resize(f, (w, h)).astype(np.float32) for f in frames], axis=0).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if kind == "motion":
        cv2.putText(img, "motion", (w - 58, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return img


def main():
    lang = sys.argv[1]
    cfg = LANGS[lang]
    home = F / lang
    z = np.load(home / "tracks" / f"{cfg['video']}.npz")
    meta = json.loads(str(z["meta"]))
    track = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    raw, t = fifty(track), track["time"]
    filled, _ = interpolate_missing(raw)
    sw = float(np.nanmedian(np.linalg.norm(raw[:, R_SH, :2] - raw[:, L_SH, :2], axis=-1)))
    det = {"right": ~np.isnan(raw[:, RH, 0]).all(1), "left": ~np.isnan(raw[:, LH, 0]).all(1)}
    cfg["_frame"] = lambda sec: int(np.searchsorted(t, sec))
    letters = cfg["letters"] or vtt_letters(home / "videos" / f"{cfg['video']}.en.vtt")
    out = cfg["lex"] / "alphabet"
    (out / "signs").mkdir(parents=True, exist_ok=True)
    cfg["faces"].mkdir(parents=True, exist_ok=True)
    rows, report, problems = [], [], []
    for letter, c0, c1 in letters:
        c0, c1 = cfg["windows"].get(letter, (c0, c1))
        i0, i1 = cfg["_frame"](c0), cfg["_frame"](c1)
        s, e, kind = segment(cfg, letter, i0, i1, raw, filled, sw, det)
        clip = normalize(interpolate_missing(raw[s:e])[0])
        used = [h for h in cfg["hands"] if h == "right" or det[h][s:e].mean() > 0.3]
        rates = {h: round(float(det[h][s:e].mean()), 2) for h in used}
        sid = cfg["prefix"] + (TID_ASCII.get(letter, letter.lower()) if lang == "tid" else letter.lower())
        np.save(out / "signs" / f"{sid}.npy", clip.astype(np.float32))
        row = {"start": round(float(t[s]), 3), "end": round(float(t[e - 1]), 3), "video": Path(meta["video"]).name}
        save_face(cfg["faces"], sid, track, raw, meta, s, e - s, 0.0, row, clip)
        rows.append({"gloss": letter, "sign_id": sid, "dataset": cfg["dataset"],
                     "keypoints_path": f"alphabet/signs/{sid}.npy", "review_status": "pending",
                     "is_religious": False, "is_letter": True})
        ok_len = (MIN_HOLD <= e - s <= MAX_HOLD) if kind == "static" else (10 <= e - s <= MAX_MOTION)
        ok_det = all(r >= MIN_DETECT for r in rates.values())
        rec = {"letter": letter, "sign_id": sid, "kind": kind, "caption": [round(c0, 2), round(c1, 2)],
               "start": row["start"], "end": row["end"], "frames": int(e - s), "hands": used, "detected": rates,
               "face_detected": row["face_detected"], "length_ok": ok_len, "detection_ok": ok_det, "_se": (s, e)}
        report.append(rec)
        if not (ok_len and ok_det):
            problems.append(f"{letter}: frames {e - s} ({'ok' if ok_len else 'BAD'}), detected {rates}")
        print(f"{letter:2s} {kind:6s} {row['start']:7.2f}-{row['end']:7.2f}s {e - s:3d} frames hands {rates} "
              f"face {row['face_detected']:.0%}")
    # every letter of the alphabet exactly once, in order, clips not overlapping
    glosses = [r["letter"] for r in report]
    missing = [ch for ch in cfg["alphabet"] if ch not in glosses]
    dupes = sorted({g for g in glosses if glosses.count(g) > 1})
    overlaps = [f"{a['letter']}/{b['letter']}" for a, b in zip(report, report[1:]) if b["_se"][0] < a["_se"][1]]
    print(f"missing {missing} duplicates {dupes} overlaps {overlaps} problems {problems}")
    (out / "entries.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                       encoding="utf-8")
    # QA sheet: one tile per letter, alphabet order
    cap = cv2.VideoCapture(str(home / "videos" / f"{cfg['video']}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    crop = cfg["crop"] or (330, 160, 930, 720)
    tiles = []
    for r in report:
        ts = [r["start"], (r["start"] + r["end"]) / 2, r["end"]] if r["kind"] == "motion" else [(r["start"] + r["end"]) / 2]
        tiles.append(tile(cap, fps, ts, crop, f"{r['letter']}  {r['frames']}f  {r['start']:.1f}s", r["kind"]))
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(home / "qa" / "alphabet_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for r in report:
        del r["_se"]
    src = {k: cfg[k] for k in ("video", "url", "channel", "title", "uploaded", "licence")}
    src.update(signer_hand="right", crop=cfg["crop"], variants_not_kept=cfg.get("variants", {}),
               missing=missing, duplicates=dupes, overlaps=overlaps, problems=problems, letters=report,
               qa_sheet=str(home / "qa" / "alphabet_sheet.jpg"), video_file=str(home / "videos" / f"{cfg['video']}.mp4"),
               track_file=str(home / "tracks" / f"{cfg['video']}.npz"))
    (out / "sources.json").write_text(json.dumps(src, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{len(rows)} letters -> {out / 'entries.jsonl'}; sheet {home / 'qa' / 'alphabet_sheet.jpg'}")


if __name__ == "__main__":
    main()
