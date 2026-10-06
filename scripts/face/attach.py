"""Stage 2 of the face pass: find every stored sign clip inside its video's Holistic track and save the face that
goes with it, frame for frame.

A lexicon clip is normalize(interpolate_missing(raw[s:e])) of a 25 fps track of its video, and most builders did not
save s and e. The clip itself is the key: every window of the new track (scripts/face/track.py) is turned into the
same 50-point form and compared with the clip (mean distance over shoulders, elbows, wrists and the 42 hand points,
in shoulder widths); the closest window is the clip's place in the video. A hint time (terms.csv) narrows the search
when the builder kept one. A distance well above what re-running MediaPipe gives on the same frames (0.1-0.3: this is
the Holistic task model, the builders used the older solution API) means the clip was not found, and nothing is
written for it; when the builder recorded a start time, landing within 0.2 s of it also counts.

Per sign, data/face/<lexicon>/<sign_id>.npz:
  blend (T,52) float16    face blendshape scores, same frames as the clip; names in blend_names
  face  (T,478,3) float16 face landmarks in the clip's own frame (neck-centred, shoulder widths, crop-normalised
                          axes like the clip; multiply x by aspect for square units)
  aspect, match (distance), start / end (seconds in the source video), video
and one row per sign in data/face/<lexicon>/manifest.jsonl.

  PYTHONPATH=src python scripts/face/attach.py gdm|noor|...
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.pose.keypoints import L_EL, L_SH, L_WR, NECK, NOSE, R_EL, R_SH, R_WR, interpolate_missing  # noqa: E402

TRACKS = DATA / "face" / "tracks"
BODY = [(NOSE, 0), (R_SH, 12), (R_EL, 14), (R_WR, 16), (L_SH, 11), (L_EL, 13), (L_WR, 15)]
COMPARE = [R_SH, R_EL, R_WR, L_SH, L_EL, L_WR] + list(range(8, 50))
FOUND = 0.25  # windows farther than this (shoulder widths) are not the clip

# source -> lexicon file (relative to DATA), dataset name, [(track file, hint table or None)]
SOURCES = {
    "gdm": ("asl/lexicon/lexicon_all.jsonl", "gdm_islamic_signs", "asl",
            [("gdm_L-lWEt7OeJk.npz", "asl/lexicon/islamic/terms.csv")]),
    "noor": ("asl/lexicon/lexicon_all.jsonl", "noor_for_sign", "asl",
             [("noor_VWzAH0j56f4.npz", "asl/lexicon/noor/terms.csv")]),
    "curriculum": ("lexicon_v2/lexicon_qa.jsonl", "quran_curriculum", "ar", "curriculum"),  # tracks qc_<surah>.npz
    # Tawasol lessons: sign taw<video>_<k> comes from lesson <video>; no times kept for most, so a full search
    "tawasol": ("lexicon_v2/lexicon_qa.jsonl", "tawasol", "ar",
                [(f"taw_{n}.npz", None, f"taw{n}_") for n in
                 "002 003 004 005 006 007 009 076 083 086 087 088 089 091 101 102 104 108 109 111 113 117 118 120 "
                 "121 123 125 131 137 139 147 152 153".split()]),
}


def fifty(track):
    """The track in the lexicon's [T,50,3] layout (NaN where not detected), as keypoints.frame_from_holistic builds it."""
    T = len(track["time"])
    out = np.full((T, 50, 3), np.nan, np.float32)
    for j, mp_id in BODY:
        out[:, j] = track["pose"][:, mp_id, :3]
    out[:, NECK] = (out[:, R_SH] + out[:, L_SH]) / 2
    out[:, 8:29], out[:, 29:50] = track["left_hand"], track["right_hand"]
    return out


def window_frame(raw):
    """normalize() of a window, also returning the neck track and scale so the face can be put in the same frame."""
    filled, _ = interpolate_missing(raw)
    widths = np.linalg.norm(filled[:, R_SH] - filled[:, L_SH], axis=-1)
    widths = widths[np.isfinite(widths) & (widths > 1e-6)]
    if widths.size == 0:
        return None, None, None
    scale = float(np.median(widths))
    neck = filled[:, NECK:NECK + 1, :]
    return (filled - neck) / scale, neck, scale


def distance(a, b, seen=None):
    """Mean 2D distance over the compared joints; seen (T,50) masks joints the new track did not detect (an undetected
    hand is zero-filled, which would swamp a clip that matches everywhere else)."""
    d = np.linalg.norm(a[:, COMPARE, :2] - b[:, COMPARE, :2], axis=-1)
    if seen is not None:
        d = np.where(seen[:, COMPARE], d, np.nan)
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.inf


def locate(clip, raw50, hint=None, slack=75):
    """Start index of the window of raw50 that best matches clip (2D distance), and that distance."""
    n, L = len(raw50), len(clip)
    lo, hi = (0, n - L + 1) if hint is None else (max(0, hint - slack), min(n - L + 1, hint + slack + 1))
    # coarse pass on a global normalisation (fast), then the exact per-window normalisation on the best 20
    seen = ~np.isnan(raw50[..., 0])
    filled, _ = interpolate_missing(raw50)
    w = np.linalg.norm(filled[:, R_SH] - filled[:, L_SH], axis=-1)
    g = (filled - filled[:, NECK:NECK + 1]) / float(np.median(w[w > 1e-6]))
    coarse = [(distance(clip, g[s:s + L], seen[s:s + L]), s) for s in range(lo, hi)]
    best = None
    for _, s in sorted(coarse)[:20]:
        win, _, _ = window_frame(raw50[s:s + L])
        if win is not None:
            d = distance(clip, win, seen[s:s + L])
            best = (d, s) if best is None or d < best[0] else best
    return best[1], best[0]


def curriculum_candidates():
    """The Qur'an curriculum (scripts/lexicon/arsl/quran_curriculum.py): each sign is one occurrence of its lemma,
    frames [i0, i1) of a surah video, the DTW medoid of up to 12. Returns normalize_ar(lemma) -> [(surah, i0, i1)]."""
    from collections import defaultdict
    from isharati.text.arabic import normalize_ar, strip_diacritics
    occ = defaultdict(list)
    for line in open(r"D:\islam\gathring data\gloss_analysis\word_glosses.jsonl", encoding="utf-8"):
        row = json.loads(line)
        for w in row["words"]:
            if w.get("frames") and w["frames"][1] - w["frames"][0] >= 3:
                occ[normalize_ar(strip_diacritics(w["lemma"]))].append((row["surah"], *w["frames"]))
    return occ


def attach_curriculum(signs, lex_dir, out_dir, dataset):
    """Every candidate occurrence of a sign's lemma is tried; the same model on the same frames matches almost exactly."""
    from isharati.text.arabic import normalize_ar
    occ, tracks, manifest = curriculum_candidates(), {}, []
    for sid, r in signs.items():
        clip = np.load(lex_dir / r["keypoints_path"])
        best = None
        for surah, i0, i1 in occ.get(normalize_ar(r["gloss"]), [])[:12]:
            if i1 - i0 != len(clip):
                continue
            if surah not in tracks:
                path = TRACKS / f"qc_{surah:03d}.npz"
                if not path.exists():
                    continue
                z = np.load(path)
                tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
                tracks[surah] = (tr, fifty(tr), json.loads(str(z["meta"])))
            tr, raw50, meta = tracks[surah]
            win, _, _ = window_frame(raw50[i0:i1])
            if win is None:
                continue
            d = distance(clip, win, ~np.isnan(raw50[i0:i1, :, 0]))
            if best is None or d < best[0]:
                best = (d, surah, i0, i1)
        row = {"sign_id": sid, "gloss": r["gloss"], "dataset": dataset}
        if best is None:
            row.update(found=False, match=None)
        else:
            d, surah, i0, i1 = best
            tr, raw50, meta = tracks[surah]
            row.update(video=Path(meta["video"]).name, surah=surah, start=round(float(tr["time"][i0]), 3),
                       end=round(float(tr["time"][i1 - 1]), 3), frames=len(clip), match=round(d, 3), found=d <= FOUND)
            if row["found"]:
                save_face(out_dir, sid, tr, raw50, meta, i0, len(clip), d, row, clip)
        manifest.append(row)
        print(f"{sid:10s} {r['gloss'][:16]:16s} match {row['match']} {'ok' if row['found'] else 'NOT FOUND'} "
              f"face {row.get('face_detected', 0):.0%}", flush=True)
    return manifest


def save_face(out_dir, sid, track, raw50, meta, s, L, d, row, clip):
    _, neck, scale = window_frame(raw50[s:s + L])
    face = (track["face"][s:s + L].astype(np.float32) - neck) / scale
    blend = track["blend"][s:s + L]
    row["face_detected"] = round(float(np.mean(~np.isnan(blend[:, 0]))), 3)
    # ratio: the clip's head height over shoulder width, which proportions.fix rescales at load; the face follows
    from isharati.pose import proportions
    np.savez_compressed(out_dir / f"{sid}.npz", blend=blend, face=face.astype(np.float16),
                        blend_names=np.array(meta["blend_names"]), aspect=meta["aspect"],
                        ratio=proportions.ratio(clip),
                        match=d, start=row["start"], end=row["end"], video=row["video"])


def main():
    name = sys.argv[1]
    lex_file, dataset, lang, tracks = SOURCES[name]
    rows = [json.loads(l) for l in open(DATA / lex_file, encoding="utf-8")]
    signs = {}
    for r in rows:  # aliases share a clip: one face file per sign_id
        if r["dataset"] == dataset:
            signs.setdefault(r["sign_id"], r)
    out_dir = DATA / "face" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    lex_dir = (DATA / lex_file).parent
    if tracks == "curriculum":
        manifest = attach_curriculum(signs, lex_dir, out_dir, dataset)
        tracks = []
    for track_file, hints_file, *only in tracks:  # only: the sign_id prefix of the signs cut from this video
        if not (TRACKS / track_file).exists():
            print("no track yet:", track_file)
            continue
        z = np.load(TRACKS / track_file)
        meta = json.loads(str(z["meta"]))
        track = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
        raw50, t = fifty(track), track["time"]
        hints = {}
        if hints_file:
            import csv
            # terms.csv has one row per term in the builder's order, and sign <prefix>_<n> is row n
            hints = {n: float(r["start_s"]) for n, r in enumerate(csv.DictReader(open(DATA / hints_file, encoding="utf-8")))
                     if r["start_s"]}
        for sid, r in signs.items():
            if only and not sid.startswith(only[0]):
                continue
            clip = np.load(lex_dir / r["keypoints_path"])
            h = hints.get(int(sid.rsplit("_", 1)[1])) if hints else None
            # GDM and Noor recorded the start on this same 25 fps grid: take it as is, and only measure the match
            s, d = locate(clip, raw50, None if h is None else int(np.argmin(np.abs(t - h))), slack=0 if h is not None else 75)
            row = {"sign_id": sid, "gloss": r["gloss"], "dataset": dataset, "video": Path(meta["video"]).name,
                   "start": round(float(t[s]), 3), "end": round(float(t[s + len(clip) - 1]), 3),
                   "frames": len(clip), "match": round(d, 3)}
            # found: the window starts where the builder recorded the sign (when it did), or matches closely
            row["hint_offset"] = None if h is None else round(float(t[s]) - h, 2)
            row["found"] = d <= FOUND or (h is not None and abs(row["hint_offset"]) <= 0.2 and d <= 2 * FOUND)
            if row["found"]:
                save_face(out_dir, sid, track, raw50, meta, s, len(clip), d, row, clip)
            manifest.append(row)
            print(f"{sid:10s} {r['gloss'][:22]:22s} match {d:.3f} offset {row['hint_offset']} {'ok' if row['found'] else 'NOT FOUND'} "
                  f"face {row.get('face_detected', 0):.0%}", flush=True)
    path = out_dir / "manifest.jsonl"
    keep = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    keep = [m for m in keep if m["dataset"] != dataset] + manifest
    path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in keep) + "\n", encoding="utf-8")
    ok = sum(m["found"] for m in manifest)
    print(f"{dataset}: {ok}/{len(manifest)} signs found in their video -> {out_dir}")


if __name__ == "__main__":
    main()
