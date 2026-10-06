"""Signed hadith dataset: coverage, per-hadith samples and pose packs.

One private Hugging Face dataset, ``hadith-sign``, holding ``manifest.jsonl`` (one row per signed sample, labelled with
the hadith as printed in its collection) and ``isharati/<source>-NNN.npz`` (one [T, 50, 3] float16 pose array per sample
id, 25 fps, the same format as the Qur'an packs).

Local mode: ``HadithData(root)`` (or env ``ISHARATI_HADITH_DIR``) reads ``<root>/hadith-sign/...`` straight from disk; a
root that itself holds ``manifest.jsonl`` is the dataset folder. Hub mode: with no root, files come from
``hf_hub_download("<owner>/hadith-sign", ..., repo_type="dataset")``; the owner is env ``ISHARATI_HADITH_OWNER``, else
``ISHARATI_QURAN_OWNER`` (default ``FatimahEmadEldin``). A manifest that cannot be fetched is logged and retried on a
later access at most once per 60 s; a loaded one is re-read every 30 min, since the dataset is still growing.
Pose packs load lazily, outside the shared lock, and a failed pack is retried after the same 60 s.
The ``holistic/`` packs are never downloaded.

Only labelled samples with an Isharati pose are browsable: rows with ``label_source == "none"``, ``isharati`` false,
no hadith, or a primary hadith whose label conflicts with the video title (``title_conflict``) are left out and counted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

REPO = "hadith-sign"
FPS = 25
_MAX_PACKS = 8
_RETRY_SECONDS = 60
_RELOAD_SECONDS = 1800                                   # the dataset is republished every 30 min while it grows
_now = time.monotonic                                    # the retry clock; patched in tests
_DEFAULT_OWNER = "FatimahEmadEldin"
SIGN_LANGUAGES = ("ArSL", "TİD")
# browse order: Nawawi's Forty first, then the six books, then anything else by key
COLLECTION_ORDER = ("nawawi", "riyad", "bukhari", "muslim", "abudawud", "tirmidhi", "nasai", "ibnmajah", "malik")
REF_RE = re.compile(r"[a-z0-9_]{1,32}:[0-9]{1,6}[a-z]?")
_SHORT = 120


def _number(n) -> tuple:
    m = re.match(r"(\d+)(.*)", str(n))
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, str(n))


def _collection_key(c: str) -> tuple:
    return (COLLECTION_ORDER.index(c), "") if c in COLLECTION_ORDER else (len(COLLECTION_ORDER), c)


def short_text(text_ar: str | None, n: int = _SHORT) -> str:
    """The matn without the isnad where it can be told apart (the quoted saying, or what follows the salutation),
    cut to n characters."""
    t = (text_ar or "").replace("‏", "").strip()
    quoted = re.search(r'"\s*([^"]{8,}?)\s*"', t)
    if quoted:
        t = quoted.group(1)
    else:
        parts = re.split(r"صلى الله عليه و ?سلم", t, maxsplit=1)
        if len(parts) == 2 and len(parts[1].strip(" :ـ،.")) >= 8:
            t = parts[1]
    t = re.split(r"\s\.\s*(?:حَدِيثٌ|رَوَاهُ)", t, maxsplit=1)[0]   # the compiler's grading after the matn (Nawawi)
    t = re.sub(r"\s+", " ", t).strip(" :ـ،.")
    return t if len(t) <= n else t[:n].rstrip() + "…"


def _usable(r: dict) -> str | None:
    """Why a row is not browsable (None when it is)."""
    if r.get("label_source", "none") == "none" or not r.get("hadiths"):
        return "unlabelled"
    if not r.get("isharati") or not r.get("isharati_file"):
        return "no_pose"
    if r["hadiths"][0].get("title_conflict"):
        return "title_conflict"
    return None


class HadithData:
    def __init__(self, root: Path | None = None, repo: str = REPO):
        self.root = Path(root) if root is not None else None
        self.repo = repo
        self._lock = threading.RLock()
        self._packs: dict[str, dict] = {}
        self._pack_failed: dict[str, float] = {}
        self._pack_locks: dict[str, threading.Lock] = {}
        self.manifest: list[dict] | None = None
        self._by_id: dict[str, dict] = {}
        self._by_ref: dict[str, list[dict]] = {}
        self._alias: dict[str, str] = {}
        self._skipped: dict[str, int] = {}
        self._loaded_at: float | None = None
        self._failed_at: float | None = None
        self._loading = False
        self._refresh()

    def _file(self, name: str) -> Path:
        if self.root is not None:
            base = self.root if (self.root / "manifest.jsonl").exists() else self.root / self.repo
            return base / name
        from huggingface_hub import hf_hub_download
        owner = os.environ.get("ISHARATI_HADITH_OWNER") or os.environ.get("ISHARATI_QURAN_OWNER", _DEFAULT_OWNER)
        return Path(hf_hub_download(f"{owner}/{self.repo}", name, repo_type="dataset"))

    def _read(self) -> tuple:
        rows = [json.loads(line) for line in
                self._file("manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {r["id"]: r for r in rows}
        by_ref: dict[str, list[dict]] = {}
        alias: dict[str, str] = {}
        skipped: dict[str, int] = {}
        for r in rows:
            why = _usable(r)
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            by_ref.setdefault(r["hadiths"][0]["ref"], []).append(r)
        for ref, rs in by_ref.items():
            rs.sort(key=lambda r: (r["sign_language"] != "ArSL", r["source"], r["id"]))  # ArSL first; stable by source
            for a in rs[0]["hadiths"][0].get("also_in") or []:
                if a not in by_ref:
                    alias.setdefault(a, ref)
        return rows, by_id, by_ref, alias, skipped

    def _refresh(self) -> None:
        """Load the manifest when it is not loaded (retrying a failure at most once per _RETRY_SECONDS) or is older
        than _RELOAD_SECONDS. Picked under the lock, read outside it, swapped in under it; a failed reload keeps the
        index already served."""
        with self._lock:
            t = _now()
            if self._loading:
                return
            if self.manifest is None:
                if self._failed_at is not None and t - self._failed_at < _RETRY_SECONDS:
                    return
            elif self._loaded_at is not None and t - self._loaded_at < _RELOAD_SECONDS:
                return
            elif self._failed_at is not None and t - self._failed_at < _RETRY_SECONDS:
                return
            self._loading = True
        try:
            res = self._read()
        except Exception as e:                               # missing/private/offline/malformed: skip for now
            log.warning("hadith dataset unavailable: %s", e)
            with self._lock:
                self._loading = False
                self._failed_at = _now()
            return
        with self._lock:
            self._loading = False
            self._failed_at = None
            self._loaded_at = _now()
            changed = self.manifest is not None and res[0] != self.manifest
            self.manifest, self._by_id, self._by_ref, self._alias, self._skipped = res
            if changed:                                      # republished: packs may have grown, read them again
                self._packs.clear()
                self._pack_failed.clear()

    def resolve(self, ref: str) -> str:
        """The browsable ref for ref: itself, or the hadith it is a parallel narration of (``also_in``)."""
        self._refresh()
        return ref if ref in self._by_ref else self._alias.get(ref, ref)

    def coverage(self) -> dict:
        self._refresh()
        with self._lock:
            by_ref, skipped = self._by_ref, dict(self._skipped)
        samples = [r for rs in by_ref.values() for r in rs]
        sources: dict[str, dict] = {}
        langs = {sl: {"sign_language": sl, "samples": 0, "_refs": set()} for sl in SIGN_LANGUAGES}
        colls: dict[str, dict] = {}
        for r in samples:
            h = r["hadiths"][0]
            s = sources.setdefault(r["source"], {"key": r["source"], "name": r.get("source_name") or r["source"],
                                                 "sign_language": r["sign_language"], "samples": 0, "_refs": set(),
                                                 "seconds": 0.0})
            s["samples"] += 1
            s["_refs"].add(h["ref"])
            s["seconds"] += max(0.0, float(r.get("end") or 0) - float(r.get("start") or 0))
            sl = langs.setdefault(r["sign_language"], {"sign_language": r["sign_language"], "samples": 0, "_refs": set()})
            sl["samples"] += 1
            sl["_refs"].add(h["ref"])
            c = colls.setdefault(h["collection"], {"collection": h["collection"],
                                                   "name": h.get("collection_name") or h["collection"],
                                                   "samples": 0, "_refs": set()})
            c["samples"] += 1
            c["_refs"].add(h["ref"])

        def done(d: dict) -> dict:
            refs = d.pop("_refs")
            if "seconds" in d:
                d["hours"] = round(d.pop("seconds") / 3600, 2)
            return {**d, "hadith": len(refs)}
        return {"samples": len(samples), "hadith": len(by_ref),
                "hours": round(sum(max(0.0, float(r.get("end") or 0) - float(r.get("start") or 0))
                                   for r in samples) / 3600, 2),
                "skipped": skipped,
                "sources": [done(s) for s in sorted(sources.values(), key=lambda s: -s["samples"])],
                "sign_languages": [done(s) for s in langs.values()],
                "collections": [done(c) for _, c in sorted(colls.items(), key=lambda kv: _collection_key(kv[0]))]}

    def hadith_list(self, collection: str | None = None, sign_language: str | None = None) -> list[dict]:
        self._refresh()
        with self._lock:
            by_ref = self._by_ref
        out = []
        for ref, rows in by_ref.items():
            if sign_language:
                rows = [r for r in rows if r["sign_language"] == sign_language]
            h = rows[0]["hadiths"][0] if rows else None
            if h is None or (collection and h["collection"] != collection):
                continue
            out.append({"ref": ref, "collection": h["collection"], "collection_name": h.get("collection_name") or "",
                        "number": str(h.get("number", "")), "text": short_text(h.get("text_ar")),
                        "samples": len(rows), "sign_languages": sorted({r["sign_language"] for r in rows})})
        out.sort(key=lambda x: (_collection_key(x["collection"]), _number(x["number"])))
        return out

    def item(self, ref: str) -> dict | None:
        """The hadith (its text from the first sample) and every sample signing it, ArSL first."""
        ref = self.resolve(ref)
        with self._lock:
            rows = self._by_ref.get(ref)
        if not rows:
            return None
        h = rows[0]["hadiths"][0]
        samples = []
        for r in rows:
            rh = r["hadiths"][0]
            samples.append({"id": r["id"], "source": r["source"], "source_name": r.get("source_name") or r["source"],
                            "sign_language": r["sign_language"], "channel": r.get("channel"), "title": r.get("title"),
                            "youtube_url": r.get("youtube_url"), "video_id": r.get("video_id"),
                            "start": r.get("start"), "end": r.get("end"), "label_source": r.get("label_source"),
                            "read_start": rh.get("read_start"), "read_end": rh.get("read_end"),
                            "gloss": rh.get("gloss"), "isharati": True})
        return {"ref": ref, "collection": h["collection"], "collection_name": h.get("collection_name") or "",
                "number": str(h.get("number", "")), "also_in": h.get("also_in") or [],
                "text_ar": h.get("text_ar") or "", "text_en": h.get("text_en"), "text_tr": h.get("text_tr"),
                "samples": samples}

    def _pack(self, name: str) -> dict | None:
        with self._lock:
            if name in self._packs:
                return self._packs[name]
            if name in self._pack_failed and _now() - self._pack_failed[name] < _RETRY_SECONDS:
                return None
            klock = self._pack_locks.setdefault(name, threading.Lock())
        with klock:                                          # one loader per pack; other packs and the index stay free
            with self._lock:
                if name in self._packs:
                    return self._packs[name]
                if name in self._pack_failed and _now() - self._pack_failed[name] < _RETRY_SECONDS:
                    return None
            try:
                with np.load(self._file(name)) as z:
                    pack = {n: z[n] for n in z.files}
            except Exception as e:
                log.warning("hadith pose pack %s unavailable: %s", name, e)
                with self._lock:
                    self._pack_failed[name] = _now()
                return None
            with self._lock:
                self._pack_failed.pop(name, None)
                self._packs[name] = pack
                while len(self._packs) > _MAX_PACKS:
                    self._packs.pop(next(iter(self._packs)))
            return pack

    def pose(self, sample_id: str) -> dict | None:
        self._refresh()
        with self._lock:
            row = self._by_id.get(sample_id)
        if row is None or _usable(row):
            return None
        name = row["isharati_file"]
        if not re.fullmatch(r"isharati/[A-Za-z0-9_-]+\.npz", name):  # the manifest names the pack; never a path out
            return None
        pack = self._pack(name)
        if pack is None or sample_id not in pack:
            return None
        frames = np.nan_to_num(pack[sample_id].astype(np.float32), nan=0.0).round(4)
        out = {"fps": FPS, "frames": frames.tolist()}
        out.update(self._face(row, sample_id, len(frames)))
        return out

    def _face(self, row: dict, sample_id: str, length: int) -> dict:
        """The signer's face on the pose's frames, as the /pose endpoint gives it (isharati.pose.face): contour points
        for the skeleton view and, when extracted, the blendshape scores that drive the avatar's expression."""
        from isharati.pose import face
        name = row.get("face_file") or ""
        if not re.fullmatch(r"face/[A-Za-z0-9_-]+\.npz", name):
            return {}
        pack = self._pack(name)
        if not pack or f"{sample_id}/points" not in pack:
            return {}
        out = {"face": face.as_json(pack[f"{sample_id}/points"][:length, :, :2]), "face_edges": face.CONTOURS["edges"]}
        if f"{sample_id}/blend" in pack and row.get("blend_names"):
            out["blend"] = face.as_json(pack[f"{sample_id}/blend"][:length])
            out["blend_names"] = row["blend_names"]
        return out


_DATA: HadithData | None = None
_DATA_LOCK = threading.Lock()


def load() -> HadithData:
    global _DATA
    with _DATA_LOCK:
        if _DATA is None:
            d = os.environ.get("ISHARATI_HADITH_DIR")
            _DATA = HadithData(Path(d)) if d else HadithData()
        return _DATA
