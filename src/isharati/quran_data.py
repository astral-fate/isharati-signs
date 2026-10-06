"""Qur'an sign-language datasets: coverage, per-ayah segments and pose packs.

Three private Hugging Face datasets, one per source, each holding ``manifest.jsonl``, ``surahs.json`` and
``isharati/<sss>.npz`` (one [T, 50, 3] float16 pose array per segment id, 25 fps).

Local mode: ``QuranData(root)`` (or env ``ISHARATI_QURAN_DIR``) reads ``<root>/<repo>/...`` straight from disk.
Hub mode: with no root, files come from ``hf_hub_download("<owner>/<repo>", ..., repo_type="dataset")``; the owner is
env ``ISHARATI_QURAN_OWNER`` (default ``FatimahEmadEldin``). A source that cannot be fetched (missing, private,
still uploading, malformed) is logged and skipped, then retried on a later access at most once per 60 s.
Nothing is read at import time; pose packs load lazily, outside the shared lock, and a failed pack is retried
after the same 60 s.
The ``holistic/`` packs are never downloaded.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SOURCES: dict[str, dict] = {
    "kfc": {"repo": "quran-sign-kfc", "label": "King Fahd Complex"},
    "curriculum": {"repo": "quran-sign-curriculum", "label": "Al-Kharj curriculum"},
    "mukhtasar": {"repo": "tafsir-mukhtasar-sign", "label": "al-Mukhtasar fi Tafsir"},
    "tebyan": {"repo": "quran-sign-tebyan", "label": "Tebyan"},
    "diyanet": {"repo": "quran-sign-diyanet-tid", "label": "Diyanet (Turkish Sign Language)"},
}
FPS = 25
_MAX_PACKS = 8
_RETRY_SECONDS = 60
_now = time.monotonic                                    # the retry clock; patched in tests
_DEFAULT_OWNER = "FatimahEmadEldin"


class QuranData:
    def __init__(self, root: Path | None = None, repos: dict | None = None):
        self.root = Path(root) if root is not None else None
        self.repos = repos or {k: v["repo"] for k, v in SOURCES.items()}
        self._lock = threading.RLock()
        self._packs: dict[tuple[str, int], dict] = {}
        self.manifest: dict[str, list[dict]] = {}
        self.surahs: dict[str, list[dict]] = {}
        self._by_id: dict[str, dict[str, dict]] = {}
        self._per_source: dict[str, dict[tuple[int, int], list[dict]]] = {}
        self._failed_at: dict[str, float] = {}
        self._loading: set[str] = set()
        self._pack_failed: dict[tuple[str, int], float] = {}
        self._pack_locks: dict[tuple[str, int], threading.Lock] = {}
        self._index: dict[tuple[int, int], list[dict]] = {}
        self._covered: list[tuple[int, int]] = []
        self._refresh()

    def _read_source(self, key: str) -> tuple:
        rows = [json.loads(line) for line in
                self._file(key, "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        surahs = json.loads(self._file(key, "surahs.json").read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in rows}
        index: dict[tuple[int, int], list[dict]] = {}
        for r in rows:
            for s, a in r["ayahs"]:
                index.setdefault((s, a), []).append({**r, "source": key})
        return rows, surahs, by_id, index

    def _refresh(self) -> None:
        """Load every source not loaded yet (retrying a failed one at most once per _RETRY_SECONDS).

        Due sources are picked under the lock, read outside it, and merged under it again."""
        with self._lock:
            due = [k for k in SOURCES if k in self.repos and k not in self.manifest and k not in self._loading
                   and not (k in self._failed_at and _now() - self._failed_at[k] < _RETRY_SECONDS)]
            self._loading.update(due)
        if not due:
            return
        results = {}
        for key in due:
            try:
                results[key] = self._read_source(key)
            except Exception as e:                           # missing/private/offline/malformed: skip for now
                results[key] = e
                log.warning("Qur'an source %s unavailable: %s", key, e)
        with self._lock:
            for key, res in results.items():
                self._loading.discard(key)
                if isinstance(res, Exception):
                    self._failed_at[key] = _now()
                    continue
                self._failed_at.pop(key, None)
                self.manifest[key], self.surahs[key], self._by_id[key], self._per_source[key] = res
            index: dict[tuple[int, int], list[dict]] = {}
            for key in SOURCES:                              # source order kfc, curriculum, mukhtasar
                for sa, rows in self._per_source.get(key, {}).items():
                    index.setdefault(sa, []).extend(rows)
            for rows in index.values():                      # recitation first, then tafsir; stable by source
                rows.sort(key=lambda r: (r["type"] != "recitation", r["type"] != "tafsir"))
            self._index, self._covered = index, sorted(index)

    def _file(self, key: str, name: str) -> Path:
        repo = self.repos[key]
        if self.root is not None:
            return self.root / repo / name
        from huggingface_hub import hf_hub_download
        owner = os.environ.get("ISHARATI_QURAN_OWNER", _DEFAULT_OWNER)
        return Path(hf_hub_download(f"{owner}/{repo}", name, repo_type="dataset"))

    def coverage(self) -> dict:
        self._refresh()
        names: dict[int, str] = {}
        counts = {n: {k: {"recitation": 0, "tafsir": 0} for k in SOURCES} for n in range(1, 115)}
        sources = []
        for key in SOURCES:
            surahs = self.surahs.get(key)
            if surahs is None:
                continue
            for s in surahs:
                names.setdefault(s["surah"], s.get("surah_name", ""))
                if 1 <= s["surah"] <= 114:
                    counts[s["surah"]][key] = {"recitation": s["recitation"], "tafsir": s["tafsir"]}
            sources.append({"key": key, "label": SOURCES[key]["label"],
                            "segments": sum(s["segments"] for s in surahs),
                            "recitation": sum(s["recitation"] for s in surahs),
                            "tafsir": sum(s["tafsir"] for s in surahs),
                            "ayahs": sum(s["ayahs"] for s in surahs), "surahs": len(surahs),
                            "hours": round(sum(s["seconds"] for s in surahs) / 3600, 2)})
        return {"sources": sources,
                "surahs": [{"surah": n, "name": names.get(n, ""), **counts[n]} for n in range(1, 115)]}

    def segments(self, surah: int, ayah: int) -> list[dict]:
        self._refresh()
        return [dict(r) for r in self._index.get((surah, ayah), [])]

    def nearest(self, surah: int, ayah: int) -> list[int] | None:
        """The closest covered [surah, ayah] in mushaf order: the next one at or after it, else the one before."""
        self._refresh()
        covered = self._covered
        if not covered:
            return None
        for sa in covered:
            if sa >= (surah, ayah):
                return list(sa)
        return list(covered[-1])

    def _pack(self, source: str, surah: int) -> dict | None:
        k = (source, surah)
        with self._lock:
            if k in self._packs:
                return self._packs[k]
            if k in self._pack_failed and _now() - self._pack_failed[k] < _RETRY_SECONDS:
                return None
            klock = self._pack_locks.setdefault(k, threading.Lock())
        with klock:                                          # one loader per pack; other packs and coverage stay free
            with self._lock:
                if k in self._packs:
                    return self._packs[k]
                if k in self._pack_failed and _now() - self._pack_failed[k] < _RETRY_SECONDS:
                    return None
            try:
                with np.load(self._file(source, f"isharati/{surah:03d}.npz")) as z:
                    pack = {n: z[n] for n in z.files}
            except Exception as e:
                log.warning("Qur'an pose pack %s/%03d unavailable: %s", source, surah, e)
                with self._lock:
                    self._pack_failed[k] = _now()
                return None
            with self._lock:
                self._pack_failed.pop(k, None)
                self._packs[k] = pack
                while len(self._packs) > _MAX_PACKS:
                    self._packs.pop(next(iter(self._packs)))
            return pack

    def pose(self, source: str, segment_id: str) -> dict | None:
        self._refresh()
        row = self._by_id.get(source, {}).get(segment_id)
        if row is None or not row.get("isharati"):
            return None
        pack = self._pack(source, int(segment_id[:3]))
        if pack is None or segment_id not in pack:
            return None
        frames = np.nan_to_num(pack[segment_id].astype(np.float32), nan=0.0).round(4)
        return {"fps": FPS, "frames": frames.tolist()}


_DATA: QuranData | None = None
_DATA_LOCK = threading.Lock()


def load() -> QuranData:
    global _DATA
    with _DATA_LOCK:
        if _DATA is None:
            d = os.environ.get("ISHARATI_QURAN_DIR")
            _DATA = QuranData(Path(d)) if d else QuranData()
        return _DATA
