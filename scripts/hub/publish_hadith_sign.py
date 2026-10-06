"""Publish the hadith-in-sign-language dataset (gathring data/hadith_pipeline.py) as a private Hugging Face dataset.

  hadith-sign   one row per sample: a signed hadith clip, or a hadith quoted inside a signed lesson, labelled with the
                hadith as printed in its collection (Bukhari, Muslim, the Sunan, Malik, Nawawi's Forty), the source
                video's YouTube id and timestamp, MediaPipe Holistic keypoints and the same motion as an Isharati pose.

No video is uploaded. The videos belong to their publishers and state no licence, so the repository is private.
Safe to re-run while the pipeline is still extracting: it publishes the samples that have keypoints so far.

  py scripts/hub/publish_hadith_sign.py [--dry-run] [--update]

Run with `env -u HF_TOKEN` (the hub login token is used).
"""
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from publish_quran_sign import FPS, despike, to_isharati  # noqa: E402

GATHERED = Path(r"D:\islam\gathring data")
SRC = GATHERED / "dataset" / "hadith_videos"
OUT = Path(r"D:\islam\hub_build") / "hadith-sign"
REPO = "hadith-sign"
COLLECTION = "Hadith in sign language"
CHUNK = 100  # samples per packed npz
MIN_SEEN = 0.6  # share of frames, between the first and last with the signer's shoulders, that must show them


FACE_CONTOURS = json.loads((Path(__file__).resolve().parents[2] / "src" / "isharati" / "pose" / "face_contours.json")
                           .read_text(encoding="utf-8"))
POINTS = np.array(FACE_CONTOURS["points"])  # the 160 contour landmarks (lips, eyes, irises, brows, nose, oval)


def to_face(z) -> dict:
    """The signer's face on the Isharati pose's frames: contour points [T,160,3] in the pose's own units (centred on
    the neck, divided by the median shoulder width, as isharati.pose.keypoints.normalize) and, when extracted, the
    52 blendshape scores [T,52] (facial expression), both resampled to 25 fps on the same grid as to_isharati."""
    t = np.asarray(z["time"], np.float64)
    pose = np.asarray(z["pose"], np.float32)
    sh_r, sh_l = pose[:, 12, :3], pose[:, 11, :3]
    neck = (sh_r + sh_l) / 2
    widths = np.linalg.norm(sh_r - sh_l, axis=-1)
    widths = widths[np.isfinite(widths) & (widths > 1e-6)]
    if not widths.size or len(t) < 2:
        return {}
    ok = np.isfinite(neck[:, 0])
    neck = np.stack([np.interp(t, t[ok], neck[ok, k]) for k in range(3)], axis=1)  # hold the neck across gaps
    pts = (np.asarray(z["face"], np.float32)[:, POINTS] - neck[:, None, :]) / float(np.median(widths))
    grid = np.arange(t[0], t[-1] + 1e-6, 1.0 / FPS)

    def resample(a):  # per channel; frames with no face stay NaN (the avatar keeps its last expression)
        flat = a.reshape(len(t), -1).astype(np.float64)
        seen = np.isfinite(flat[:, 0])
        if seen.sum() < 2:
            return None
        out = np.stack([np.interp(grid, t[seen], flat[seen, k]) for k in range(flat.shape[1])], axis=1)
        idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        out[~seen[idx]] = np.nan  # no face there in the video: no invented expression
        return out.reshape(len(grid), *a.shape[1:]).astype(np.float16)

    out = {"points": resample(pts)}
    if "blend" in z:
        out["blend"] = resample(np.asarray(z["blend"], np.float32))
    return {k: v for k, v in out.items() if v is not None}


def build() -> tuple[Path, list]:
    # refresh the manifest from the pipeline state (cheap, no YouTube)
    subprocess.run([str(GATHERED / ".venv" / "Scripts" / "python.exe"), "hadith_pipeline.py", "manifest"],
                   cwd=GATHERED, check=True)
    rows = [json.loads(l) for l in (SRC / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    gloss = defaultdict(dict)
    if (SRC / "gloss.jsonl").exists():
        for g in map(json.loads, (SRC / "gloss.jsonl").open(encoding="utf-8")):
            gloss[g["ref"]][g["sign_language"]] = g
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "holistic").mkdir(parents=True)
    (OUT / "isharati").mkdir()
    (OUT / "face").mkdir()
    by_source = defaultdict(list)
    for r in rows:
        if (SRC / r["keypoints"]).exists():
            by_source[r["source"]].append(r)
    manifest = []
    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda r: r["id"])
        for c in range(0, len(group), CHUNK):
            part = f"{source}-{c // CHUNK:03d}"
            hol, ish, fac = {}, {}, {}
            for r in group[c:c + CHUNK]:
                z = np.load(SRC / r["keypoints"])
                for k in ("pose", "left_hand", "right_hand", "face", "time"):
                    hol[f"{r['id']}/{k}"] = z[k]
                # the avatar pose only covers the signing: drop the title cards before and after (no body seen), and
                # skip clips where the signer is still missing from most frames (cropped or off screen)
                # a frame counts as the signer only at the clip's usual body size: intro zooms, a tiny figure in a
                # title card or a one-frame misdetection (shoulders 5x narrower) would collapse the avatar
                width = np.linalg.norm(z["pose"][:, 11, :2] - z["pose"][:, 12, :2], axis=-1)
                med = np.nanmedian(width) if np.isfinite(width).any() else np.nan
                good = np.isfinite(width) & (width > 0.6 * med) & (width < 1.6 * med)
                seen = np.flatnonzero(good)
                p = None
                if len(seen) and good[seen[0]:seen[-1] + 1].mean() >= MIN_SEEN:
                    span = slice(seen[0], seen[-1] + 1)
                    clip = {k: np.array(z[k][span]) for k in ("pose", "left_hand", "right_hand", "face", "time", "blend")
                            if k in z.files}
                    for k in ("pose", "left_hand", "right_hand", "face", "blend"):  # glitch frames: NaN, interpolated over
                        if k in clip:
                            clip[k][~good[span]] = np.nan
                    # MediaPipe gives x as a share of the frame width and y of its height: on a 16:9 video one unit of
                    # x is 1.78 units of y, so without this the body comes out stretched tall (upper arm 1.24 shoulder
                    # widths instead of ~0.7) and the avatar's arms no longer fit it
                    meta = json.loads(str(z["meta"]))
                    aspect = meta["width"] / meta["height"]
                    for k in ("pose", "left_hand", "right_hand", "face"):
                        if k in clip:
                            clip[k] = clip[k].astype(np.float32)
                            clip[k][..., 0] *= aspect
                    p = to_isharati(despike({k: clip[k] for k in ("pose", "left_hand", "right_hand", "time")}))
                    face = to_face({k: clip[k] for k in ("pose", "face", "time", "blend") if k in clip})
                    for k, v in face.items():
                        fac[f"{r['id']}/{k}"] = v
                    r["face"] = "points" in face
                    r["expression"] = "blend" in face
                    if "blend" in face:
                        r["blend_names"] = json.loads(str(z["meta"])).get("blend_names")
                    # the sample becomes the signed span: start/end (and the YouTube timestamp) follow the trim, so the
                    # app's offsets into the pose (read_start - start) stay right
                    r["start"], r["end"] = round(float(z["time"][seen[0]]), 2), round(float(z["time"][seen[-1]]), 2)
                    r["youtube_url"] = f"https://www.youtube.com/watch?v={r['video_id']}&t={int(r['start'])}s"
                if p is not None:
                    ish[r["id"]] = p
                for h in r["hadiths"]:  # the gloss in the sample's own sign language only
                    g = gloss.get(h["ref"], {}).get(r["sign_language"])
                    h["gloss"] = {r["sign_language"]: g["gloss"]} if g else None
                manifest.append({**r, "keypoints": f"holistic/{part}.npz", "isharati": p is not None,
                                 "isharati_file": f"isharati/{part}.npz" if p is not None else None,
                                 "face_file": f"face/{part}.npz" if r.get("face") else None})
            np.savez_compressed(OUT / "holistic" / f"{part}.npz", **hol)
            np.savez_compressed(OUT / "isharati" / f"{part}.npz", **ish)
            np.savez_compressed(OUT / "face" / f"{part}.npz", **fac)
        print(f"  {source}: {len(group)} samples", flush=True)
    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
                                        encoding="utf-8")
    if (SRC / "gloss.jsonl").exists():
        shutil.copy(SRC / "gloss.jsonl", OUT / "gloss.jsonl")
    (OUT / "README.md").write_text(card(manifest), encoding="utf-8")
    return OUT, manifest


def card(m: list) -> str:
    n = len(m)
    refs = Counter(h["ref"] for r in m for h in r["hadiths"][:1])
    hours = sum(r["end"] - r["start"] for r in m) / 3600
    by_src = Counter(r["source_name"] for r in m)
    by_sl = Counter(r["sign_language"] for r in m)
    labels = Counter(r["label_source"] for r in m)
    colls = Counter(h["collection_name"] for r in m for h in r["hadiths"][:1])
    det = {k: np.mean([r["detection_rate"].get(k, 0) for r in m]) if m else 0
           for k in ("pose", "left_hand", "right_hand", "face")}
    src_rows = "\n".join(f"| {k} | {v} |" for k, v in by_src.most_common())
    lab_rows = "\n".join(f"| `{k}` | {v} |" for k, v in labels.most_common())
    col_rows = "\n".join(f"| {k} | {v} |" for k, v in colls.most_common())
    return f"""---
license: other
license_name: permission-pending
language:
- ar
- tr
- sgn
task_categories:
- other
tags:
- sign-language
- arabic-sign-language
- turkish-sign-language
- hadith
- mediapipe
- pose
pretty_name: "Hadith in sign language"
configs:
- config_name: default
  data_files: manifest.jsonl
---

# Hadith in sign language

Signed hadith gathered from YouTube channels for deaf Muslims, each sample labelled with the hadith as printed in its
collection, with MediaPipe Holistic keypoints (body, hands, 478 face landmarks) and the same motion as an Isharati pose
sequence ready to drive a 3D avatar. Part of the Isharati project.

**Licence status.** The videos belong to their publishers and state no licence; they are **not** included here. Each
row links to the original video at the sample's timestamp. Private, shared for research while permission is requested.

## Figures

| Samples | Distinct hadith | Hours | ArSL | TİD |
|---|---|---|---|---|
| {n:,} | {len(refs):,} | {hours:.1f} | {by_sl.get('ArSL', 0):,} | {by_sl.get('TİD', 0):,} |

Mean detection rates: pose {det['pose']:.2f}, left hand {det['left_hand']:.2f}, right hand {det['right_hand']:.2f},
face {det['face']:.2f}.

| Source | Samples |
|---|---|
{src_rows}

| Collection of the label | Samples |
|---|---|
{col_rows}

## How each sample is labelled

The label is always the collection's text (fawazahmed0/hadith-api editions), never a transcript:

| `label_source` | Samples |
|---|---|
{lab_rows}

- `speech_match`: the voice-over reads the hadith; Whisper (large-v3-turbo) gives word timings and the transcript is
  matched word by word to the collection, requiring 8+ matched words in order and 3+ distinctive word pairs (a chain of
  narrators alone never matches). `read_start` / `read_end` mark where the text itself is read.
- `translation_match`: Turkish voice-over; an LLM names the Arabic hadith and the Arabic collection must confirm it.
- `onscreen_match`: silent video; a vision model reads the hadith text shown on screen and the collection must confirm
  it. `title_conflict` marks samples whose screen reading disagrees with the Nawawi number in the title.
- `title`: silent video whose title names the hadith (Nawawi's Forty number, Riyad al-Salihin or Umdat al-Ahkam
  number in `title_ref`), with no text reading.
- `+title`: the publisher's title agrees in kind (`title_ref` is kept for checking).

`gloss` (per hadith, per sign language) is a machine gloss of the hadith text by the Isharati app's lexicon-constrained
glossers (ArSL for Arabic, TİD for Turkish); `oov` marks words with no recorded sign. It is a target for
text-to-sign, **not** an annotation of what the signer did.

## Layout

```
manifest.jsonl              one row per sample
gloss.jsonl                 one row per (hadith, sign language): text, gloss sequence, lexicon coverage
holistic/<source>-NNN.npz   MediaPipe Holistic: <id>/pose [T,33,4], <id>/left_hand [T,21,3], <id>/right_hand [T,21,3],
                            <id>/face [T,478,3], <id>/time [T] (seconds in the video)
isharati/<source>-NNN.npz   Isharati poses: <id> -> [T,50,3] float16, 25 fps
face/<source>-NNN.npz       the signer's face on the same 25 fps frames: <id>/points [T,160,3] face contours in the
                            pose's units (neck-centred, shoulder widths), <id>/blend [T,52] facial-expression
                            blendshape scores (names in the row's blend_names; rows with `expression` true)
```

Long silent lessons are sampled at every 2nd frame (`stride` 2).

## Loading

```python
import json, numpy as np
from huggingface_hub import hf_hub_download

repo = "FatimahEmadEldin/{REPO}"
rows = [json.loads(l) for l in open(hf_hub_download(repo, "manifest.jsonl", repo_type="dataset"), encoding="utf-8")]
r = next(r for r in rows if r["isharati"] and r["hadiths"])
poses = np.load(hf_hub_download(repo, r["isharati_file"], repo_type="dataset"))
print(r["hadiths"][0]["ref"], r["hadiths"][0]["text_ar"][:80], poses[r["id"]].shape)
```
"""


def publish(folder: Path, update: bool) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    repo = f"{api.whoami()['name']}/{REPO}"
    if api.repo_exists(repo, repo_type="dataset") and not update:
        raise SystemExit(f"{repo} already exists; pass --update to upload over it")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder), private=True)
    col = api.create_collection(COLLECTION, description="Hadith signed in Arabic and Turkish Sign Language, labelled "
                                "with the collection text, with MediaPipe keypoints and avatar-ready poses.",
                                private=True, exists_ok=True)
    api.add_collection_item(col.slug, item_id=repo, item_type="dataset", exists_ok=True)
    return repo


def main():
    folder, manifest = build()
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    print(f"{len(manifest)} samples, {size / 1e9:.2f} GB in {folder}")
    if "--dry-run" in sys.argv:
        return
    print("published (private):", publish(folder, "--update" in sys.argv))


if __name__ == "__main__":
    main()
