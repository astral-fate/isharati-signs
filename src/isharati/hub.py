"""Fetch the application's data from the Hugging Face Hub (on the Space; a no-op in development).

ISHARATI_HUB_DATASET="user/isharati-data" names the dataset (scripts/hub/publish.py: one folder per language), and
ISHARATI_LANGS="en,ar" the languages to fetch. The files are downloaded with HF_TOKEN, copied to the data/ layout given
by layout.json, and each poses*.npz archive is unpacked to the paths the lexicon tables refer to, inside
ISHARATI_DATA_DIR. A marker per dataset revision makes a restart skip work already done.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

from isharati.config import DATA


def ensure_data() -> None:
    repo = os.environ.get("ISHARATI_HUB_DATASET", "").strip()
    if not repo:
        return
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get("HF_TOKEN")
    langs = [l.strip() for l in os.environ.get("ISHARATI_LANGS", "en,ar").split(",") if l.strip()]
    rev = HfApi(token=token).dataset_info(repo).sha
    marker = DATA / f".hub_{rev}_{'_'.join(langs)}"
    if marker.exists():
        return
    snap = Path(snapshot_download(repo, repo_type="dataset", revision=rev, token=token,
                                  allow_patterns=[f"{l}/*" for l in langs] + ["layout.json"]))
    layout = json.loads((snap / "layout.json").read_text(encoding="utf-8"))
    for src, dst in layout.items():
        if (snap / src).exists():
            (DATA / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snap / src, DATA / dst)
    for lang in langs:
        for archive in sorted((snap / lang).glob("*poses*.npz")):
            with np.load(archive) as z:
                for rel in z.files:
                    out = DATA / rel
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        np.save(out, z[rel])
        _unpack_faces(snap / lang / "faces.npz", lang)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(rev, encoding="utf-8")
    print("data ready:", repo, rev[:8], langs, flush=True)


def _unpack_faces(archive: Path, lang: str) -> None:
    """faces.npz (scripts/hub/publish.py) -> one data/face/<lexicon>/<sign_id>.npz per sign, as isharati.pose.face reads."""
    if not archive.exists():
        return
    from isharati.pose.face import LEXICON_DIR
    out_dir = DATA / "face" / LEXICON_DIR[lang]
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(archive) as z:
        names = z["blend_names"]
        for key in z.files:
            if key.endswith("/blend"):
                sid = key[:-len("/blend")]
                np.savez(out_dir / f"{sid}.npz", blend=z[key], face=z[f"{sid}/points"], ratio=z[f"{sid}/ratio"],
                         blend_names=names)
