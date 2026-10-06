"""Publish the data the application reads as one private Hugging Face dataset, organised by language.

  en/  English  -> ASL           ar/  Arabic -> Arabic Sign Language
  tr/  Turkish  -> TİD           ur/  Urdu   -> Indian Sign Language
Each language folder holds its Qur'an + hadith corpus (corpus.jsonl), the E5 passage embeddings, its sign lexicon
table(s) and the poses of the signs (poses*.npz, keyed by the sign's path inside the project's data/ folder).
layout.json maps every file back to the project's data/ layout, which src/isharati/hub.py restores on the Space, so the
application code reads the same paths everywhere.

The dataset is private: several sign sources do not allow redistribution (licence table in the README).

  env -u HF_TOKEN .venv/Scripts/python scripts/hub/publish.py [--dry-run] [--repo NAME]
"""
import io
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
REPO = "isharati-data"

# per language: files copied as they are (repo path -> data path)
FILES = {
    "en": {"corpus.jsonl": "asl/corpus.jsonl", "embeddings.npy": "asl/embeddings.npy",
           "embeddings_ids.json": "asl/embeddings_ids.json", "lexicon.jsonl": "asl/lexicon/lexicon_all.jsonl",
           "templates.jsonl": "asl/lexicon/templates.jsonl",
           "alphabet.jsonl": "asl/lexicon/alphabet/entries.jsonl", "names.jsonl": "asl/lexicon/names/entries.jsonl"},
    "ar": {"corpus.jsonl": "ar/corpus.jsonl", "embeddings.npy": "ar/embeddings.npy",
           "embeddings_ids.json": "ar/embeddings_ids.json", "lexicon.jsonl": "lexicon_v2/lexicon_qa.jsonl",
           "lexicon_notes.json": "lexicon_v2/lexicon_qa_notes.json", "lemma_cache.json": "lexicon_v2/lemma_cache.json",
           # CAMeL analyses (lemma, POS, clitics) for verb/noun lemma matching: the Space has no CAMeL Tools
           "analysis_cache_ar.json": "lexicon_v2/analysis_cache_ar.json",
           "templates.jsonl": "lexicon/templates.jsonl"},
    "tr": {"corpus.jsonl": "tr/corpus.jsonl", "embeddings.npy": "tr/embeddings.npy",
           "embeddings_ids.json": "tr/embeddings_ids.json",
           "lexicon.jsonl": "lexicon_tr/dictionaries/entries_tid_dictionary.jsonl",
           "alphabet.jsonl": "lexicon_tr/alphabet/entries.jsonl", "names.jsonl": "lexicon_tr/names/entries.jsonl",
           # second TİD source, for the corpus words the dictionary lacks (scripts/lexicon/tid/tid_youtube.py)
           "lexicon_tid_youtube.jsonl": "lexicon_tr/tid_youtube/entries.jsonl",
           # AUTSL signs (scripts/lexicon/tid_datasets/autsl_build.py): the medoid recording of each sign the dictionary
           # lacks, plus "variant_of" rows (other signers, and AUTSL's recordings of dictionary signs) whose poses are
           # packed into tr/autsl_poses.npz with the rest, and faces into tr/faces.npz
           "lexicon_autsl.jsonl": "lexicon_tr/autsl/entries.jsonl",
           # the dictionary's opposite pairs re-cut into one sign per word (scripts/lexicon/tid/pairs_fix.py); they
           # replace the clip two opposite glosses shared (TIDLexicon.override_words)
           "lexicon_pairs_fixed.jsonl": "lexicon_tr/pairs_fixed/entries.jsonl",
           # gap fillers from «Our hands are speaking» and The Sign Polyglot (scripts/lexicon/tid/extra_sources.py), all
           # pending review; loaded last, "variant_of" rows are recordings of words the lexicon already signs
           "lexicon_extra.jsonl": "lexicon_tr/extra/entries.jsonl",
           # Cüneyt Küçükoğlu's «Hareketli Sözlük» channel (scripts/lexicon/tid/cuneyt.py), pending review: gap fillers
           "lexicon_cuneyt.jsonl": "lexicon_tr/cuneyt/entries.jsonl",
           # recordings chosen over a missing or weak sign (scripts/lexicon/arsl/reviewed_recordings.py): zekât, şart
           "lexicon_reviewed.jsonl": "lexicon_tr/reviewed/recordings.jsonl",
           # Zemberek lemmas of corpus words and glosses (isharati.text.turkish_morph), for when zeyrek is not installed
           "lemma_cache.json": "lexicon_tr/lemma_cache.json"},
    "ur": {"corpus.jsonl": "ur/corpus.jsonl", "embeddings.npy": "ur/embeddings.npy",
           "embeddings_ids.json": "ur/embeddings_ids.json",
           "lexicon_wslp.jsonl": "lexicon_isl/wslp/entries.jsonl", "lexicon_cislr.jsonl": "lexicon_isl/cislr/entries.jsonl",
           # Pakistan Sign Language signs from the FESF PSL dictionary (scripts/lexicon/isl/psl.py): English glosses
           # plus the Urdu word; loaded after the ISL sources, "variant_of" rows are other recordings
           "lexicon_psl.jsonl": "lexicon_isl/psl/entries.jsonl",
           # the ISLRTC dictionary (scripts/lexicon/isl/islrtc.py) and one signer's ISL alphabet (isl_alphabet.py)
           "lexicon_islrtc.jsonl": "lexicon_isl/islrtc/entries.jsonl", "alphabet.jsonl": "lexicon_isl/alphabet/entries.jsonl"},
}
# pose archives: repo path -> (lexicon table in data/, folder its pose paths are relative to; None = the table's folder)
POSES = {
    "en/poses.npz": ("asl/lexicon/lexicon_all.jsonl", None),
    "en/template_poses.npz": ("asl/lexicon/templates.jsonl", None),
    "ar/poses.npz": ("lexicon_v2/lexicon_qa.jsonl", None),
    "ar/template_poses.npz": ("lexicon/templates.jsonl", None),
    "tr/poses.npz": ("lexicon_tr/dictionaries/entries_tid_dictionary.jsonl", "lexicon_tr"),
    # one signer's alphabet and the names spelled from it (scripts/lexicon/alphabet/), loaded beside the lexicon
    "en/alphabet_poses.npz": ("asl/lexicon/alphabet/entries.jsonl", "asl/lexicon"),
    "en/names_poses.npz": ("asl/lexicon/names/entries.jsonl", "asl/lexicon"),
    "tr/alphabet_poses.npz": ("lexicon_tr/alphabet/entries.jsonl", "lexicon_tr"),
    "tr/names_poses.npz": ("lexicon_tr/names/entries.jsonl", "lexicon_tr"),
    "tr/tid_youtube_poses.npz": ("lexicon_tr/tid_youtube/entries.jsonl", "lexicon_tr"),
    "tr/autsl_poses.npz": ("lexicon_tr/autsl/entries.jsonl", "lexicon_tr"),
    "tr/pairs_fixed_poses.npz": ("lexicon_tr/pairs_fixed/entries.jsonl", "lexicon_tr"),
    "tr/extra_poses.npz": ("lexicon_tr/extra/entries.jsonl", "lexicon_tr"),
    "tr/cuneyt_poses.npz": ("lexicon_tr/cuneyt/entries.jsonl", "lexicon_tr"),
    "tr/reviewed_poses.npz": ("lexicon_tr/reviewed/recordings.jsonl", "lexicon_tr"),
    "ur/poses_wslp.npz": ("lexicon_isl/wslp/entries.jsonl", "lexicon_isl"),
    "ur/poses_cislr.npz": ("lexicon_isl/cislr/entries.jsonl", "lexicon_isl"),
    "ur/poses_psl.npz": ("lexicon_isl/psl/entries.jsonl", "lexicon_isl"),
    "ur/poses_islrtc.npz": ("lexicon_isl/islrtc/entries.jsonl", "lexicon_isl"),
    "ur/alphabet_poses.npz": ("lexicon_isl/alphabet/entries.jsonl", "lexicon_isl"),
}
# face archives: repo path -> folder of per-sign face files in data/ (scripts/face/attach.py). Packed compactly: each
# sign's blendshape scores and its 160 face contour points (isharati.pose.face.POINTS) instead of the full 478-point
# mesh, which is all the application uses; isharati.hub unpacks them to the same per-sign files.
FACES = {"en/faces.npz": "face/asl", "ar/faces.npz": "face/ar", "tr/faces.npz": "face/tr", "ur/faces.npz": "face/ur"}
LANG = {"en": ("English", "American Sign Language (ASL)"), "ar": ("Arabic", "Arabic Sign Language (ArSL)"),
        "tr": ("Turkish", "Turkish Sign Language (TİD)"), "ur": ("Urdu", "Indian Sign Language (ISL)")}
LICENCES = [
    ("Qur'an, Arabic (IslamicEval 2025 canonical text)", "shared-task release", "corpus"),
    ("Qur'an translations (QuranEnc: Saheeh Intl., Rowwad, Junagarhi)", "QuranEnc terms (to be confirmed)", "corpus"),
    ("Hadith (HadeethEnc)", "HadeethEnc terms (to be confirmed)", "corpus"),
    ("ASL Citizen", "Microsoft Research licence: non-commercial, no redistribution", "en lexicon"),
    ("WLASL, MS-ASL", "C-UDA, research use", "en lexicon"),
    ("GDM Islamic signs, Noor For Sign, Obscure ASL (YouTube)", "not stated; permission to be requested", "en lexicon"),
    ("KArSL-502", "research use", "ar lexicon"),
    ("Qur'an curriculum, Tawasol, Saudi/Arabic dictionary, ArabicSignLanguage, DisabilityApps, Levantine Shorts (YouTube)",
     "not stated; permission to be requested", "ar lexicon"),
    ("TİD Sözlüğü (YouTube)", "not stated; permission to be requested", "tr lexicon"),
    ("TiDiSLaM (YouTube, Islamic terms in TİD)", "not stated; permission to be requested", "tr lexicon"),
    ("AUTSL (Ankara University TSL dataset)", "academic and research use only, no commercial use", "tr lexicon"),
    ("WSLP 2025", "CC BY-NC-ND 4.0", "ur lexicon"),
    ("CISLR", "AFL-3.0", "ur lexicon"),
    ("PSL dictionary, FESF / Deaf Reach (psl.org.pk)", "all rights reserved; permission to be requested", "ur lexicon"),
]


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def poses_of(table, root):
    t = DATA / table
    base = DATA / root if root else t.parent
    out = {}
    for r in rows(t):
        p = (base / r["keypoints_path"]).resolve()
        if p.exists():
            out[p.relative_to(DATA.resolve()).as_posix()] = np.load(p)
    return out


def faces_of(folder):
    """{"<sign_id>/blend", "<sign_id>/points", "<sign_id>/ratio", "blend_names"} for every face file in data/<folder>."""
    from isharati.pose.face import POINTS
    out = {}
    for f in sorted((DATA / folder).glob("*.npz")):
        try:  # a face pass may still be writing: a half-written file is left for the next publish
            with np.load(f) as z:
                sign = {f"{f.stem}/blend": z["blend"].astype(np.float16),
                        f"{f.stem}/points": z["face"][:, POINTS].astype(np.float16),
                        f"{f.stem}/ratio": np.float32(z["ratio"]) if "ratio" in z.files else np.float32(np.nan)}
                names = z["blend_names"]
        except Exception as e:
            print("skipped", f.name, repr(e)[:80])
            continue
        out.update(sign)
        out.setdefault("blend_names", names)
    return out


def card(stats):
    cfg = []
    for lang, files in FILES.items():
        cfg.append(f"- config_name: {lang}_corpus\n  data_files: {lang}/corpus.jsonl")
        for name in files:
            if name.startswith("lexicon") and name.endswith(".jsonl"):
                cfg.append(f"- config_name: {lang}_{name[:-6]}\n  data_files: {lang}/{name}")
    lines = ["---", "license: other", "pretty_name: Isharati data", "language: [en, ar, tr, ur]",
             "tags: [sign-language, quran, hadith, retrieval, pose, asl, arsl]", "configs:", *cfg, "---", "",
             "# Isharati data", "",
             "Everything the Isharati application reads to answer a question about Islam in a sign language: the Qur'an and",
             "hadith passages it answers from (with their retrieval embeddings), and the sign lexicons and poses it signs with.",
             "**Private**: several sign sources do not allow redistribution (see Licences).", "",
             "## Structure", "",
             "| Folder | Text language | Sign language | Passages (Qur'an / hadith) | Lexicon glosses | Signs with poses |",
             "|---|---|---|---|---|---|"]
    for lang in FILES:
        s = stats[lang]
        lines.append(f"| `{lang}/` | {LANG[lang][0]} | {LANG[lang][1]} | {s['passages']:,} ({s['quran']:,} / {s['hadith']:,}) | "
                     f"{s['glosses']:,} | {s['poses']:,} |")
    lines += ["", "Files per language folder:", "",
              "- `corpus.jsonl`: one passage per line: `pid` (`q2:43` for a verse, `h5907` for a hadith; shared across",
              "  languages), `kind` (`quran` | `hadith`), `reference`, `text` (a hadith passage is its title, text and",
              "  scholarly explanation), `url` (source page).",
              "- `embeddings.npy`: float16 `[passages, 1024]`, multilingual-e5-large with the `passage: ` prefix, unit length;",
              "  `embeddings_ids.json` gives the `pid` of each row.",
              "- `lexicon*.jsonl`: one sign per line: `gloss`, `sign_id`, `dataset` (source), `keypoints_path`,",
              "  `review_status` (`approved` | `pending`), `is_religious`, `is_letter`. `templates.jsonl`: a second signer's",
              "  recording of a sign, used for back-translation.",
              "- `poses*.npz`: the poses, keyed by `keypoints_path` resolved inside the project's `data/` folder.",
              "  Each pose is `[T, 50, 3]` at 25 fps: 0 nose, 1 neck, 2-4 right shoulder/elbow/wrist, 5-7 left, 8-28 left",
              "  hand, 29-49 right hand (MediaPipe Holistic); neck-centred, shoulder-width units.",
              "- Arabic only: `lemma_cache.json` (CAMeL Tools lemmas of corpus words), `lexicon_notes.json` (dialect notes).",
              "- `layout.json` (root): the data/ path of every file, used by the application to restore its layout.", "",
              "## Use", "", "```python",
              "from huggingface_hub import snapshot_download",
              "root = snapshot_download('" + stats['repo'] + "', repo_type='dataset', allow_patterns=['ar/*', 'layout.json'])",
              "```", "",
              "## Sources and licences", "", "| Source | Licence | Used in |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in LICENCES]
    lines += ["", "## Citation", "",
              "Isharati technical report (Bathel AI Challenge 2026). Cite the original sources above when using their data:",
              "IslamicEval 2025 (Mubarak et al., ArabicNLP 2025), QuranEnc, HadeethEnc, ASL Citizen (Desai et al., 2023),",
              "WLASL (Li et al., 2020), KArSL (Sidig et al., 2021), CISLR (Joshi et al., 2022), WSLP 2025."]
    return "\n".join(lines) + "\n"


def main():
    from huggingface_hub import HfApi, get_token
    dry = "--dry-run" in sys.argv
    name = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else REPO
    api = HfApi(token=get_token())
    repo = f"{api.whoami()['name']}/{name}"
    stage = Path(tempfile.mkdtemp(prefix="isharati_data_"))
    layout, stats = {}, {"repo": repo}
    for lang, files in FILES.items():
        (stage / lang).mkdir(parents=True)
        for dst, src in files.items():
            if (DATA / src).exists():
                (stage / lang / dst).write_bytes((DATA / src).read_bytes())
                layout[f"{lang}/{dst}"] = src
        c = Counter(r["kind"] for r in rows(DATA / files["corpus.jsonl"]))
        glosses = {r["gloss"] for k, v in files.items() if k.startswith("lexicon") and k.endswith(".jsonl")
                   for r in rows(DATA / v) if not r.get("variant_of")}  # variant rows: more recordings, not glosses
        stats[lang] = {"passages": sum(c.values()), "quran": c["quran"], "hadith": c["hadith"], "glosses": len(glosses),
                       "poses": 0}
    for dst, (table, root) in POSES.items():
        poses = poses_of(table, root)
        buf = io.BytesIO()
        np.savez_compressed(buf, **poses)
        (stage / dst).write_bytes(buf.getvalue())
        if "template" not in dst:
            stats[dst.split("/")[0]]["poses"] += len(poses)
        print(f"{dst}: {len(poses)} poses, {len(buf.getvalue()) / 1e6:.1f} MB", flush=True)
    for dst, folder in FACES.items():
        faces = faces_of(folder)
        if not faces:
            continue
        buf = io.BytesIO()
        np.savez_compressed(buf, **faces)
        (stage / dst).write_bytes(buf.getvalue())
        n = sum(k.endswith("/blend") for k in faces)
        stats[dst.split("/")[0]]["faces"] = n
        print(f"{dst}: {n} signs with the signer's face, {len(buf.getvalue()) / 1e6:.1f} MB", flush=True)
    (stage / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    (stage / "README.md").write_text(card(stats), encoding="utf-8")
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1e6
    print({k: v for k, v in stats.items() if k != "repo"}, f"total {size:.0f} MB, staged in {stage}")
    if dry:
        return
    try:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(stage), commit_message="Isharati data: corpora, lexicons, poses")
        print("uploaded (private):", f"https://huggingface.co/datasets/{repo}")
    finally:  # the stage is a ~2 GB copy: left behind by every run, they filled the disk (14 GB on F:)
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
