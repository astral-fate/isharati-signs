"""Publish the Qur'an sign-language ground truth as three standalone private Hugging Face datasets, one per source:

  quran-sign-kfc          King Fahd Complex, translation of the meanings of the Qur'an in sign language (6 releases)
  tafsir-mukhtasar-sign   al-Mukhtasar fi Tafsir al-Qur'an al-Karim in sign language (playlist PLofjBfpgjcQxcM8eMrsiNUFjip13cfCin)
  quran-sign-curriculum   Al-Kharj school curriculum, 22 short surahs

Each repository holds a manifest (one row per segment: surah, ayahs, Uthmani text, tafsir text, word timings, the
source video's YouTube id and timestamp), per-surah counts, the full MediaPipe Holistic keypoints and the same
segments as Isharati poses [T,50,3] at 25 fps, both packed per surah. No video is uploaded. The videos belong to their
publishers and state no licence, so the repositories are private ("permission to be requested").

  py scripts/hub/publish_quran_sign.py kfc|mukhtasar|curriculum|tebyan|diyanet [--dry-run] [--update] [--out DIR]

Run with `env -u HF_TOKEN` (the hub login token is used).
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pose.keypoints import interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

GATHERED = Path(r"D:\islam\gathring data")
BUILD = Path(r"D:\islam\hub_build")
COLLECTION = "Qur'an in sign language"

SOURCES = {
    "kfc": {
        "folder": "kfc_videos", "repo": "quran-sign-kfc",
        "title": "Qur'an in sign language: King Fahd Complex",
        "source": "King Fahd Glorious Qur'an Printing Complex, translation of the meanings of the Qur'an in sign "
                  "language, releases 1–6 (YouTube)",
        "owner": "King Fahd Glorious Qur'an Printing Complex",
    },
    "mukhtasar": {
        "folder": "tafsir_videos", "repo": "tafsir-mukhtasar-sign",
        "title": "Qur'an in sign language: al-Mukhtasar fi Tafsir",
        "source": "al-Mukhtasar fi Tafsir al-Qur'an al-Karim (المختصر في تفسير القرآن الكريم) in sign language, "
                  "YouTube playlist PLofjBfpgjcQxcM8eMrsiNUFjip13cfCin",
        "owner": "the channel publishing al-Mukhtasar fi Tafsir in sign language",
    },
    "curriculum": {
        "folder": "quran_sign_language_videos", "repo": "quran-sign-curriculum",
        "title": "Qur'an in sign language: Al-Kharj curriculum",
        "source": "Al-Kharj school Qur'an curriculum in sign language, 22 short surahs (YouTube)",
        "owner": "the curriculum's publisher",
    },
    "tebyan": {
        "folder": "tebyan_videos", "repo": "quran-sign-tebyan",
        "title": "Qur'an in sign language: Tebyan",
        "source": "Tebyan Qur'an (tebyanquran.com), per-ayah sign-language videos from api.tebyanquran.com "
                  "(signer 1: Juz' Amma; signer 2: from al-Fatiha)",
        "owner": "Tebyan Qur'an (tebyanquran.com)",
        "method": "Each video is the publisher's own clip for one ayah (a few span two ayat), so the label is the "
                  "publisher's ayah number and the fixed Qur'an text (Uthmani script); no transcription or "
                  "segmentation is involved. The videos have no audio. Each row links to the original clip "
                  "(`video_url`).",
    },
    "diyanet": {
        "folder": "diyanet_tid_quran", "repo": "quran-sign-diyanet-tid", "videos": Path(r"F:\tid_quran_videos\diyanet_tid_quran"),
        "title": "Qur'an in sign language: Diyanet, Turkish Sign Language",
        "source": "Diyanet İşleri Başkanlığı, «İşaret Dili ile Kur'an» (from the book «İşitme Engellilere Özel Elif-Bâ»), "
                  "YouTube playlist PLtApkPE49w-tQ8-zFm1QVrwrmYKauonS9, episodes 28 (al-Fatiha) and 29 (al-Baqarah 1-5)",
        "owner": "Diyanet İşleri Başkanlığı (Presidency of Religious Affairs, Türkiye)",
        "languages": ["tr", "sgn"], "sl_tag": "turkish-sign-language",
        "method": "The publisher shows each phrase of the ayah on screen while it is signed. The text-panel changes "
                  "(detected at 5 fps, mapped to ayat by hand) give each segment's span; the label is the fixed Qur'an "
                  "text (Uthmani script). Word times are spread evenly inside each on-screen phrase, so phrase spans "
                  "are exact and word spans approximate. (Whisper was tried and caught the intro's spoken basmala, so "
                  "it is not used for these labels.)",
    },
}

# MediaPipe pose ids -> Isharati joints 0..7 (nose, neck = mid-shoulders, R sh/el/wr, L sh/el/wr); 8..28 left hand,
# 29..49 right hand, as isharati.pose.keypoints.frame_from_holistic
BODY = [(0, 0), (2, 12), (3, 14), (4, 16), (5, 11), (6, 13), (7, 15)]


def to_isharati(z) -> np.ndarray | None:
    """One segment's Holistic arrays -> Isharati pose [T,50,3] float16 at 25 fps; None when no shoulders are seen."""
    pose, lh, rh, t = (np.asarray(z[k], np.float32) for k in ("pose", "left_hand", "right_hand", "time"))
    T = len(t)
    if T == 0:
        return None
    a = np.full((T, 50, 3), np.nan, np.float32)
    for j, mp in BODY:
        a[:, j] = pose[:, mp, :3]
    a[:, 1] = (a[:, 2] + a[:, 5]) / 2
    a[:, 8:29], a[:, 29:50] = lh[:, :, :3], rh[:, :, :3]
    zero = ~np.isfinite(a).all(axis=-1) | (np.abs(a).sum(axis=-1) == 0)   # undetected joints as NaN, not 0,0,0
    a[zero] = np.nan
    try:
        filled, _ = interpolate_missing(a)
        p = normalize(filled)
    except ValueError:  # no shoulders in the clip
        return None
    if T == 1:
        return p.astype(np.float16)
    grid = np.arange(t[0], t[-1] + 1e-6, 1.0 / FPS)          # resample to 25 fps from the real timestamps
    flat = p.reshape(T, -1)
    out = np.stack([np.interp(grid, t, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape(len(grid), 50, 3).astype(np.float16)


def frame_aspects(src_dir: Path) -> dict[str, float]:
    """video file name -> width / height of the downloaded video, from the yt-dlp .info.json next to it."""
    out = {}
    for info in src_dir.glob("*/*.info.json"):
        try:
            j = json.loads(info.read_text(encoding="utf-8"))
            if j.get("width") and j.get("height"):
                out[info.name[: -len(".info.json")] + ".mp4"] = j["width"] / j["height"]
        except (OSError, json.JSONDecodeError):
            continue
    return out


def square_units(z, aspect: float | None) -> dict:
    """MediaPipe x is a share of the frame width and y of its height: on a 16:9 video one unit of x is 1.78 of y.
    Scaling x by width/height puts both in the same unit, or the body comes out stretched tall (upper arm ~1.3-1.4
    shoulder widths instead of ~0.8) and the avatar's arms no longer fit it."""
    d = {k: np.array(z[k], np.float32) for k in ("pose", "left_hand", "right_hand", "time")}
    if aspect:
        for k in ("pose", "left_hand", "right_hand"):
            d[k][..., 0] *= aspect
    return d


MAX_TILT = 35  # degrees: a shoulder line steeper than this is not the signer facing the camera


def signer_frames(pose: np.ndarray) -> np.ndarray:
    """bool [T]: frames that show the signer as in the rest of the clip. The shoulders must be at the clip's usual
    width (an intro zoom, a tiny figure on a title card or a one-frame misdetection 5x narrower would collapse the
    avatar), face the camera the same way (left shoulder on the same side) and lie within MAX_TILT of level: a false
    body between an intro and the signing (Nawawi 42: shoulders swapped at the frame's bottom edge) otherwise turns
    the avatar round and lays it on its side. Use on square units (square_units) so the angle is a true angle."""
    pose = np.asarray(pose, np.float32)
    dx, dy = pose[:, 11, 0] - pose[:, 12, 0], pose[:, 11, 1] - pose[:, 12, 1]
    width = np.hypot(dx, dy)
    if not np.isfinite(width).any():
        return np.zeros(len(pose), bool)
    med = np.nanmedian(width)
    with np.errstate(invalid="ignore"):
        good = np.isfinite(width) & (width > 0.6 * med) & (width < 1.6 * med)
        if good.any():
            good &= np.sign(dx) == np.sign(np.nanmedian(dx[good]))
        good &= np.abs(np.degrees(np.arctan2(dy, np.abs(dx)))) < MAX_TILT
    return good


def blank_non_signer(d: dict) -> dict:
    """NaN the frames signer_frames rejects, so interpolation bridges them from the real signing on either side."""
    bad = ~signer_frames(d["pose"])
    for k in ("pose", "left_hand", "right_hand"):
        d[k][bad] = np.nan
    return d


def despike(d: dict, window: int = 7, limit: float = 0.5) -> dict:
    """Blank MediaPipe's one- or two-frame glitches so interpolation fills them: a body joint (or a whole hand, by its
    root) that sits more than `limit` shoulder widths from the median of its neighbouring frames. Real signing never
    moves a wrist 0.5 shoulder widths away and back within a few frames (that is ~15 shoulder widths a second)."""
    pose = d["pose"]
    sw = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)
    med = np.nanmedian(sw) if np.isfinite(sw).any() else np.nan
    T = len(pose)
    if not np.isfinite(med) or T < window:
        return d
    half = window // 2

    def outliers(xy):  # xy [T,2] -> bool [T]
        pad = np.pad(xy, ((half, half), (0, 0)), mode="edge")
        stack = np.stack([pad[i:i + T] for i in range(window)])
        with np.errstate(all="ignore"):
            ref = np.nanmedian(stack, axis=0)
        return np.linalg.norm(xy - ref, axis=-1) > limit * med

    for j in (0, 11, 12, 13, 14, 15, 16):
        bad = outliers(pose[:, j, :2])
        pose[bad, j] = np.nan
    for k in ("left_hand", "right_hand"):
        bad = outliers(d[k][:, 0, :2])
        d[k][bad] = np.nan
    return d


def youtube_ids(src_dir: Path) -> dict[str, str | None]:
    """video file name -> YouTube id, from the yt-dlp .info.json next to each video."""
    ids = {}
    for info in src_dir.glob("*/*.info.json"):
        try:
            ids[info.name[: -len(".info.json")] + ".mp4"] = json.loads(info.read_text(encoding="utf-8")).get("id")
        except (OSError, json.JSONDecodeError):
            continue
    return ids


def manifest_row(r: dict, yt: dict) -> dict:
    """The published manifest row for one segment (the source layouts normalised into one)."""
    if r.get("label_source") == "publisher":            # Tebyan: one whole clip per ayah (or two), plain ayah numbers
        r = {**r, "ayahs": [[int(r["surah"]), int(a)] for a in r["ayahs"]], "start": 0.0, "end": r["duration"],
             "video": Path(r["video"]).name}
    if r.get("ayahs"):
        ayahs = [list(map(int, x)) for x in r["ayahs"]]
        uthmani = " ".join(a["text_uthmani"] for a in r.get("ayah_texts") or [])
    elif r.get("ayah"):
        ayahs, uthmani = [[int(r["surah"]), int(r["ayah"])]], r.get("text_uthmani", "")
    else:
        ayahs, uthmani = [], ""
    vid = yt.get(r["video"])
    start = float(r["start"])
    return {
        "id": r["id"], "type": r["type"], "surah": int(r["surah"]), "surah_name": r.get("surah_name", ""),
        "ayahs": ayahs, "text_uthmani": uthmani,
        "tafsir_text": r.get("text") if r["type"] in ("tafsir", "intro") else None,
        "words": [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end"),
                   "frames": w.get("clip_index") or w.get("frames")} for w in r.get("words") or []],
        "start": start, "end": float(r["end"]), "fps_source": float(r["fps"]), "frames": int(r.get("clip_frames", 0)),
        "detection_rate": r.get("detection_rate", {}),
        "youtube_id": vid, "youtube_url": f"https://www.youtube.com/watch?v={vid}&t={int(start)}s" if vid else None,
        "video_title": r["video"][:-4],
        **({"video_url": r["video_url"], "signer": r.get("signer")} if r.get("video_url") else {}),
    }


def build(key: str, src_root: Path = GATHERED, out_root: Path = BUILD) -> Path:
    cfg = SOURCES[key]
    src = src_root / "dataset" / cfg["folder"]
    out = out_root / cfg["repo"]
    if out.exists():
        for item in out.iterdir():
            if item.name != ".cache":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        out.mkdir(parents=True, exist_ok=True)
    (out / "holistic").mkdir(parents=True, exist_ok=True)
    (out / "isharati").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    yt = youtube_ids(cfg.get("videos") or src_root / cfg["folder"])
    aspects = frame_aspects(cfg.get("videos") or src_root / cfg["folder"])
    by_surah = defaultdict(list)
    for r in rows:
        by_surah[int(r["surah"])].append(r)
    manifest, surahs = [], []
    for s in sorted(by_surah):
        hol, ish = {}, {}
        for r in by_surah[s]:
            row = manifest_row(r, yt)
            clip = src / "clips" / f"{r['id']}.npz"
            if clip.exists():
                z = np.load(clip)
                for k in ("pose", "left_hand", "right_hand", "face", "time"):
                    hol[f"{r['id']}/{k}"] = z[k]
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
                aspect = (meta["width"] / meta["height"]) if meta.get("width") else aspects.get(r.get("video"))
                if aspect is None:
                    raise SystemExit(f"{key}: no frame size for {r.get('video')} ({r['id']}): the pose would be stretched")
                p = to_isharati(despike(blank_non_signer(square_units(z, aspect))))
                if p is not None:
                    ish[r["id"]] = p
                row["isharati"] = p is not None
            else:
                row["isharati"] = False
            manifest.append(row)
        np.savez_compressed(out / "holistic" / f"{s:03d}.npz", **hol)
        np.savez_compressed(out / "isharati" / f"{s:03d}.npz", **ish)
        seg = [m for m in manifest if m["surah"] == s]
        surahs.append({"surah": s, "surah_name": seg[0]["surah_name"], "segments": len(seg),
                       "ayahs": len({tuple(a) for m in seg for a in m["ayahs"]}),
                       "recitation": sum(m["type"] == "recitation" for m in seg),
                       "tafsir": sum(m["type"] == "tafsir" for m in seg), "intro": sum(m["type"] == "intro" for m in seg),
                       "seconds": round(sum(m["end"] - m["start"] for m in seg), 1)})
        print(f"  surah {s:3d}: {len(seg)} segments", flush=True)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
                                        encoding="utf-8")
    (out / "surahs.json").write_text(json.dumps(surahs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "README.md").write_text(card(key, manifest, surahs), encoding="utf-8")
    if len(manifest) != len(rows):
        raise SystemExit(f"{key}: {len(manifest)} rows written, {len(rows)} in the source manifest")
    return out


WHISPER_METHOD = ("The audio was transcribed with Whisper only for its **timing**. The text of each label is the fixed "
                  "Qur'an text\n   (Uthmani script), aligned to the timed transcript, so recognition errors never "
                  "become labels.")


def card(key: str, manifest: list, surahs: list) -> str:
    cfg = SOURCES[key]
    n = len(manifest)
    kinds = {k: sum(m["type"] == k for m in manifest) for k in ("recitation", "tafsir", "intro")}
    ayahs = len({tuple(a) for m in manifest for a in m["ayahs"]})
    hours = sum(m["end"] - m["start"] for m in manifest) / 3600
    poses = sum(m["isharati"] for m in manifest)
    det = {k: np.mean([m["detection_rate"].get(k, 0) for m in manifest]) for k in ("pose", "left_hand", "right_hand", "face")}
    covered = [s["surah"] for s in surahs]
    missing = [s for s in range(1, 115) if s not in covered]
    gaps = (f"{len(missing)} of 114 surahs are not in this source: " + ", ".join(map(str, missing))) if missing else "None."
    rows = "\n".join(f"| {s['surah']} | {s['surah_name']} | {s['segments']} | {s['ayahs']} | {s['recitation']} | "
                     f"{s['tafsir']} | {s['seconds']} |" for s in surahs)
    first = f"{surahs[0]['surah']:03d}"
    langs = "\n".join(f"- {l}" for l in cfg.get("languages", ["ar", "sgn"]))
    return f"""---
license: other
license_name: permission-pending
language:
{langs}
task_categories:
- other
tags:
- sign-language
- {cfg.get('sl_tag', 'arabic-sign-language')}
- quran
- tafsir
- mediapipe
- pose
pretty_name: "{cfg['title']}"
configs:
- config_name: default
  data_files: manifest.jsonl
---

# {cfg['title']}

Ground truth for the Qur'an in sign language: every segment of the source videos aligned to the ayah it signs, with
MediaPipe Holistic keypoints and the same motion as an Isharati pose sequence ready to drive a 3D avatar. Part of the
Isharati project (grounded answers about Islam, signed) and of the collection "{COLLECTION}".

**Source.** {cfg['source']}. The videos belong to {cfg['owner']}; they are **not** included here. Each row links to
the original video at the segment's timestamp.

**Licence status.** The videos state no licence. This dataset is private and shared for research while permission is
requested from the owner. Do not redistribute it.

## Figures

| Segments | Recitation | Tafsir | Intro | Distinct ayahs | Surahs | Hours | Segments with an Isharati pose |
|---|---|---|---|---|---|---|---|
| {n:,} | {kinds['recitation']:,} | {kinds['tafsir']:,} | {kinds['intro']:,} | {ayahs:,} | {len(surahs)} | {hours:.1f} | {poses:,} |

Mean detection rates (share of frames with each part detected): pose {det['pose']:.2f}, left hand
{det['left_hand']:.2f}, right hand {det['right_hand']:.2f}, face {det['face']:.2f}.

**Known gaps.** {gaps}

## How the segments were made

1. The source videos were downloaded with their metadata.
2. {cfg.get('method', WHISPER_METHOD)}
3. Each video was cut at the aligned boundaries into recitation, tafsir and intro segments.
4. MediaPipe Holistic was run on every frame (33 pose, 21 + 21 hand and 478 face landmarks; normalised image
   coordinates).
5. The Isharati pose keeps 8 upper-body joints and both hands, fills short detection gaps by interpolation, centres
   each frame on the neck, divides by the median shoulder width and resamples to 25 fps.

## Layout

```
manifest.jsonl          one row per segment
surahs.json             per-surah counts
holistic/<sss>.npz      MediaPipe Holistic per surah: <id>/pose [T,33,4], <id>/left_hand [T,21,3],
                        <id>/right_hand [T,21,3], <id>/face [T,478,3], <id>/time [T] (seconds in the video)
isharati/<sss>.npz      Isharati poses per surah: <id> -> [T,50,3] float16, 25 fps
```

Manifest fields: `id`, `type` (recitation | tafsir | intro), `surah`, `surah_name`, `ayahs` (list of [surah, ayah]),
`text_uthmani`, `tafsir_text` (the signed tafsir, tafsir and intro segments), `words` (word, start, end, frame range),
`start`, `end` (seconds in the source video), `fps_source`, `frames`, `detection_rate`, `youtube_id`, `youtube_url`,
`video_title`, `isharati` (whether an Isharati pose exists).

## Loading a surah

```python
import json, numpy as np
from huggingface_hub import hf_hub_download

repo = "FatimahEmadEldin/{cfg['repo']}"
rows = [json.loads(l) for l in open(hf_hub_download(repo, "manifest.jsonl", repo_type="dataset"), encoding="utf-8")]
poses = np.load(hf_hub_download(repo, "isharati/{first}.npz", repo_type="dataset"))
seg = next(r for r in rows if r["surah"] == {surahs[0]['surah']} and r["isharati"])
print(seg["text_uthmani"], poses[seg["id"]].shape)   # [T, 50, 3] at 25 fps
```

## Per-surah coverage

| Surah | Name | Segments | Ayahs | Recitation | Tafsir | Seconds |
|---|---|---|---|---|---|---|
{rows}
"""


def publish(key: str, folder: Path, update: bool) -> str:
    import winreg, os
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment')
        tok, _ = winreg.QueryValueEx(k, 'HF_TOKEN')
    except Exception:
        tok = os.environ.get('HF_TOKEN')
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    repo = f"{api.whoami()['name']}/{SOURCES[key]['repo']}"
    if api.repo_exists(repo, repo_type="dataset") and not update:
        raise SystemExit(f"{repo} already exists; pass --update to upload over it")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder), private=True)
    col = api.create_collection(COLLECTION, description="Qur'an recitation and tafsir in sign language, aligned per "
                                "ayah, with MediaPipe keypoints and avatar-ready poses.", private=True, exists_ok=True)
    api.add_collection_item(col.slug, item_id=repo, item_type="dataset", exists_ok=True)
    return repo


def main():
    args = sys.argv[1:]
    key = args[0] if args and args[0] in SOURCES else None
    if not key:
        raise SystemExit(__doc__)
    out = Path(args[args.index("--out") + 1]) if "--out" in args else BUILD
    folder = build(key, out_root=out)
    manifest = (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    print(f"{key}: {len(manifest)} segments, {size / 1e9:.2f} GB in {folder}")
    if "--dry-run" in args:
        return
    print("published (private):", publish(key, folder, "--update" in args))


if __name__ == "__main__":
    main()
