"""Data files the application reads besides the corpora and lexicons: the Academy's alphabet clips and the Qur'an text.

They are published with their sources and licences in the private Hugging Face dataset isharati-app-data
(scripts/hub/publish_app_data.py), and fetched from there on first use; the code repository carries no data.

  ISHARATI_APP_DATA_DIR   a local folder with the same layout (development, tests); used instead of the Hub
  ISHARATI_APP_DATASET    the dataset id (default: <owner of ISHARATI_HUB_DATASET>/isharati-app-data)

path(name) returns the local path of one file, or None when it cannot be had (no token, offline): the caller then
leaves out what depends on it instead of failing.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)
_DEFAULT_OWNER = "FatimahEmadEldin"


def dataset_id() -> str:
    if os.environ.get("ISHARATI_APP_DATASET"):
        return os.environ["ISHARATI_APP_DATASET"]
    hub = os.environ.get("ISHARATI_HUB_DATASET", "")
    owner = hub.split("/")[0] if "/" in hub else _DEFAULT_OWNER
    return f"{owner}/isharati-app-data"


@lru_cache(maxsize=None)
def path(name: str) -> Path | None:
    """Local path of `name` (e.g. "quran/quran.json"), downloaded once from the dataset; None if unavailable."""
    local = os.environ.get("ISHARATI_APP_DATA_DIR")
    if local:
        p = Path(local) / name
        return p if p.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(dataset_id(), name, repo_type="dataset", token=os.environ.get("HF_TOKEN") or None))
    except Exception as e:  # noqa: BLE001 - offline or no access: the feature that needs it is left out
        log.warning("app data %s unavailable from %s: %s", name, dataset_id(), e)
        return None
