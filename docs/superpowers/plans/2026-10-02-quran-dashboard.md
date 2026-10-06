# Qur'an in Sign Language Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A landing-page section that presents the three Qur'an sign-language datasets — counters, a 114-surah
coverage map, and an ayah viewer that plays each segment on the avatar (cartoon or realistic) next to the original
interpreter's YouTube video.

**Architecture:** A small Python module `isharati.quran_data` loads each dataset's `manifest.jsonl` and `surahs.json`
(from local build folders or the private Hub repositories) and serves per-surah Isharati poses on demand; three
FastAPI endpoints expose coverage, the segments for an ayah, and a segment's pose. A React section `Quran.tsx` reuses
`AvatarView`, `KeypointCanvas`, `StyleSwitch`, `usePlayer` and `Counter`.

**Tech Stack:** Python 3.11+ (FastAPI, numpy, huggingface_hub), React 18 + TypeScript + Framer Motion, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-02-quran-sign-datasets-design.md` (section 2). Builds on
`docs/superpowers/specs/2026-10-02-landing-page-design.md`.

## Global Constraints

- Python tests: `py -m pytest` from the repo root; only `tests/test_extract.py` ×3 may fail (mediapipe missing).
- Web: `cd web && npx vitest run && npm run build`. Vitest gotcha: with `vi.mock` in a file, import RTL as
  `import * as rtl` (an identifier named `render` gets mis-hoisted).
- Interface strings in `en`, `ar`, `tr`, `ur`; every en key exists, non-empty, in the other three (enforced by
  `web/src/i18n/i18n.test.tsx`). Arabic/Urdu are right-to-left; Qur'an text is always `dir="rtl" lang="ar"`.
- The datasets are private; the Space reads them with its `HF_TOKEN`. The Space never downloads the Holistic packs.
- Hugging Face commands with `env -u HF_TOKEN`; never print keys.
- Sources (keys → repositories): `kfc` → `quran-sign-kfc`, `mukhtasar` → `tafsir-mukhtasar-sign`,
  `curriculum` → `quran-sign-curriculum`, all under the dataset owner `FatimahEmadEldin`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MCZ7kRB6SCynLU61xgepiB
  ```

## Review Focus

1. **An ayah no source covers** (e.g. 2:255 in curriculum-only builds, or al-Fatiha in al-Mukhtasar): the viewer says
   so and offers the nearest covered ayah, never a blank player (Task 2 test).
2. **A segment with `isharati: false`:** the YouTube video still shows, with a note, and the avatar panel does not
   error (Task 2 test).
3. **A segment without a YouTube id:** no broken iframe (Task 2 test).
4. **Unsafe `source` or `segment_id` in the pose URL:** 404, no file access outside the data (Task 1 test).
5. **A dataset that has not finished uploading on the Space:** coverage lists only the loaded sources; the page does
   not break (Task 1 test: missing source folder).

---

### Task 1: `isharati.quran_data` and the three endpoints

**Files:**
- Create: `src/isharati/quran_data.py`, `tests/test_quran_api.py`
- Modify: `src/isharati/app/server.py`

**Interfaces:**
- `SOURCES: dict[str, dict]` — key → `{"repo": str, "label": str}` (labels: "King Fahd Complex", "al-Mukhtasar fi Tafsir", "Al-Kharj curriculum").
- `QuranData(root: Path | None = None, repos: dict | None = None)`:
  - local mode when `root` is given (or env `ISHARATI_QURAN_DIR`): reads `<root>/<repo>/manifest.jsonl`,
    `surahs.json`, `isharati/<sss>.npz`;
  - hub mode otherwise: `hf_hub_download(f"{owner}/{repo}", <file>, repo_type="dataset")` with owner from env
    `ISHARATI_QURAN_OWNER` (default `FatimahEmadEldin`); a source whose download fails is skipped (logged), not fatal.
  - `.coverage() -> dict`: `{"sources": [{"key", "label", "segments", "recitation", "tafsir", "ayahs", "surahs", "hours"}],
    "surahs": [{"surah", "name", "kfc": {"recitation": n, "tafsir": n}, "mukhtasar": {...}, "curriculum": {...}}]}`
    for surahs 1..114 (every surah present, counts 0 when absent; name from any source that has it, else "").
  - `.segments(surah: int, ayah: int) -> list[dict]` — manifest rows (minus nothing; they are small) of every loaded
    source whose `ayahs` contains `[surah, ayah]`, each with an added `"source"` key; ordered recitation first, then
    tafsir, then source order kfc, curriculum, mukhtasar.
  - `.nearest(surah: int, ayah: int) -> list[int] | None` — the closest covered `[surah, ayah]` in mushaf order
    (forward first, then backward), or None if nothing is loaded.
  - `.pose(source: str, segment_id: str) -> dict | None` — `{"fps": 25, "frames": [[...]]}` from the surah pack
    (surah = int of the id's first three digits), rounded to 4 decimals, NaN→0; None if unknown. Packs are cached in
    memory per (source, surah) with at most 8 kept (simple dict + insertion-order eviction).
- `load() -> QuranData` — module-level singleton built on first use.
- Endpoints in `server.py`:
  - `GET /api/quran/coverage` → `load().coverage()`
  - `GET /api/quran/ayah?surah=&ayah=` → `{"surah", "ayah", "segments": [...], "nearest": [s, a] | null}`; 400 when
    surah ∉ 1..114 or ayah < 1.
  - `GET /api/quran/pose/{source}/{segment_id}.json` → the pose; 404 when `source ∉ SOURCES`, when `segment_id` does
    not match `^[0-9]{3}[A-Za-z0-9_-]*$`, or when not found.

- [ ] **Step 1: Write the failing tests** — `tests/test_quran_api.py` builds a fixture root with two sources:

```python
"""Qur'an dataset API: coverage, ayah lookup, poses, path safety, missing sources."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _source(root, repo, rows):
    d = root / repo
    (d / "isharati").mkdir(parents=True)
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    by = {}
    for r in rows:
        by.setdefault(r["surah"], []).append(r)
    surahs = [{"surah": s, "surah_name": rs[0]["surah_name"], "segments": len(rs),
               "ayahs": len({tuple(a) for r in rs for a in r["ayahs"]}),
               "recitation": sum(r["type"] == "recitation" for r in rs), "tafsir": sum(r["type"] == "tafsir" for r in rs),
               "intro": 0, "seconds": 10.0} for s, rs in sorted(by.items())]
    (d / "surahs.json").write_text(json.dumps(surahs, ensure_ascii=False), encoding="utf-8")
    for s, rs in by.items():
        np.savez(d / "isharati" / f"{s:03d}.npz",
                 **{r["id"]: np.full((5, 50, 3), 0.5, np.float16) for r in rs if r["isharati"]})


def _row(id, typ, surah, ayahs, isharati=True, yt="abc"):
    return {"id": id, "type": typ, "surah": surah, "surah_name": f"s{surah}", "ayahs": ayahs, "text_uthmani": "نص",
            "tafsir_text": "تفسير" if typ == "tafsir" else None, "words": [], "start": 10.0, "end": 14.0,
            "fps_source": 25.0, "frames": 100, "detection_rate": {}, "youtube_id": yt,
            "youtube_url": f"https://www.youtube.com/watch?v={yt}&t=10s" if yt else None, "video_title": "v",
            "isharati": isharati}


@pytest.fixture
def root(tmp_path):
    _source(tmp_path, "quran-sign-kfc", [_row("114_001", "recitation", 114, [[114, 1]]),
                                         _row("114_tafsir_001-002", "tafsir", 114, [[114, 1], [114, 2]], isharati=False)])
    _source(tmp_path, "tafsir-mukhtasar-sign", [_row("002_tafsir_255-255", "tafsir", 2, [[2, 255]], yt=None)])
    return tmp_path                                  # curriculum missing on purpose


def test_quran_data_coverage_ayahs_nearest_pose(root):
    from isharati.quran_data import QuranData
    q = QuranData(root)
    cov = q.coverage()
    assert [s["key"] for s in cov["sources"]] == ["kfc", "mukhtasar"]        # curriculum not loaded, not fatal
    assert len(cov["surahs"]) == 114
    assert cov["surahs"][113]["kfc"] == {"recitation": 1, "tafsir": 1}
    assert cov["surahs"][0]["kfc"] == {"recitation": 0, "tafsir": 0}
    seg = q.segments(114, 1)
    assert [(s["source"], s["type"]) for s in seg] == [("kfc", "recitation"), ("kfc", "tafsir")]
    assert q.segments(1, 1) == [] and q.nearest(1, 1) == [2, 255]
    p = q.pose("kfc", "114_001")
    assert p["fps"] == 25 and len(p["frames"]) == 5 and abs(p["frames"][0][0][0] - 0.5) < 1e-3
    assert q.pose("kfc", "114_tafsir_001-002") is None and q.pose("curriculum", "114_001") is None


@pytest.fixture
def client(root, monkeypatch):
    monkeypatch.setenv("ISHARATI_QURAN_DIR", str(root))
    from isharati import quran_data
    monkeypatch.setattr(quran_data, "_DATA", None)
    from isharati.app import server
    return TestClient(server.app)


def test_endpoints(client):
    assert len(client.get("/api/quran/coverage").json()["surahs"]) == 114
    r = client.get("/api/quran/ayah", params={"surah": 114, "ayah": 2}).json()
    assert [s["id"] for s in r["segments"]] == ["114_tafsir_001-002"] and r["segments"][0]["isharati"] is False
    r = client.get("/api/quran/ayah", params={"surah": 1, "ayah": 1}).json()
    assert r["segments"] == [] and r["nearest"] == [2, 255]
    assert client.get("/api/quran/ayah", params={"surah": 115, "ayah": 1}).status_code == 400
    assert client.get("/api/quran/pose/kfc/114_001.json").json()["fps"] == 25
    assert client.get("/api/quran/pose/nope/114_001.json").status_code == 404
    assert client.get("/api/quran/pose/kfc/..%2F..%2Fx.json").status_code == 404
    assert client.get("/api/quran/pose/kfc/114_tafsir_001-002.json").status_code == 404
```

- [ ] **Step 2:** run `py -m pytest tests/test_quran_api.py -v` — FAIL (module missing).
- [ ] **Step 3: Implement** `src/isharati/quran_data.py` to the interfaces above (module docstring explaining local
  vs hub mode; `_DATA = None` module-level singleton; `load()` builds `QuranData(Path(os.environ["ISHARATI_QURAN_DIR"]))`
  when that env var is set, else hub mode). In hub mode download `manifest.jsonl` and `surahs.json` per source at first
  use and `isharati/<sss>.npz` lazily in `pose()`. Add the three endpoints to `server.py` next to `/api/stats`, with
  the validation above (`re.fullmatch(r"[0-9]{3}[A-Za-z0-9_-]*", segment_id)`).
- [ ] **Step 4:** `py -m pytest -q` — new tests pass; nothing else regresses.
- [ ] **Step 5:** local check against the real builds: `ISHARATI_QURAN_DIR=D:\islam\hub_build PYTHONPATH=src py -c
  "from isharati.quran_data import QuranData; from pathlib import Path; q=QuranData(Path(r'D:\islam\hub_build'));
  print([s['key'] for s in q.coverage()['sources']], len(q.segments(114,1)), q.nearest(2,1))"`. Report the output.
- [ ] **Step 6: Commit** — "Qur'an datasets API: coverage, ayah segments, poses".

---

### Task 2: The "Qur'an in sign language" section

**Files:**
- Create: `web/src/sections/Quran.tsx`, `web/src/sections/quran.ts` (pure helpers), `web/src/sections/Quran.test.tsx`,
  `web/src/sections/quran.test.ts`
- Modify: `web/src/api.ts` (three calls), `web/src/types.ts` (types), `web/src/App.tsx` (section after Lexicon),
  `web/src/sections/Nav.tsx` (link `quran`), `web/src/styles.css`, `web/src/i18n/{en,ar,tr,ur}.json`, `web/vite.config.ts`
  (no change needed if `/api` is already proxied — confirm)

**Interfaces:**
- types: `QuranSourceKey = "kfc" | "mukhtasar" | "curriculum"`; `QuranCoverage` (as Task 1); `QuranSegment` (manifest
  row + `source`); `QuranAyah = { surah: number; ayah: number; segments: QuranSegment[]; nearest: [number, number] | null }`.
- api: `getQuranCoverage(): Promise<QuranCoverage>`, `getQuranAyah(surah, ayah): Promise<QuranAyah>`,
  `getQuranPose(source, id): Promise<PoseFrames>` (all via the existing `call()`).
- helpers (`quran.ts`): `cellKind(s): "both" | "recitation" | "tafsir" | "none"` from a coverage surah row (sums over
  sources); `wordAt(seg, t): number` — index of the word whose `[start-seg.start, end-seg.start)` contains t, or -1;
  `embedUrl(seg): string | null` — `https://www.youtube-nocookie.com/embed/<id>?start=<floor(start)>&end=<ceil(end)>&rel=0`
  or null without `youtube_id`; `AYAH_COUNTS: number[]` — the 114 surah ayah counts (standard Hafs numbering) used to
  bound the ayah picker.
- Section `id="quran"`:
  1. heading + lead; three source cards with `Counter`s (segments, ayahs, surahs, hours) and a "private dataset —
     permission requested" note;
  2. surah map: 114 buttons (`.surah-cell`, class by `cellKind`), title attribute "<n> · <name>"; clicking selects
     the surah and ayah 1 in the viewer;
  3. viewer: surah select (1–114 with names when known) + ayah number input bounded by `AYAH_COUNTS`; the list of
     segments for that ayah as chips "<source label> · <recitation|tafsir>"; selecting one loads its pose;
     left panel: Uthmani text (`dir="rtl" lang="ar"`, words wrapped in spans, the current word highlighted with
     `wordAt` while playing for recitation segments) and the tafsir text for tafsir segments; right panel: the
     avatar (`AvatarView` with the shared avatar/outfit state, `StyleSwitch`, Avatar/Keypoints toggle, play/scrub)
     and, below or beside it, the YouTube embed (iframe with `title`, `allow="encrypted-media; picture-in-picture"`,
     `loading="lazy"`) when `embedUrl` is not null;
  4. empty state: no segments → message + "Go to <nearest>" button; `isharati: false` → note that only the original
     video exists for this segment, avatar panel hidden.
- i18n keys (add to all four files; give real translations): `nav.quran`, `quran.title`, `quran.lead`,
  `quran.private`, `quran.segments`, `quran.ayahs`, `quran.surahs`, `quran.hours`, `quran.map`, `quran.legend.both`,
  `quran.legend.recitation`, `quran.legend.tafsir`, `quran.legend.none`, `quran.surah`, `quran.ayah`,
  `quran.recitation`, `quran.tafsir`, `quran.none`, `quran.goNearest`, `quran.noPose`, `quran.original`,
  `quran.source.kfc`, `quran.source.mukhtasar`, `quran.source.curriculum`.
- Tests: `quran.test.ts` — `cellKind`, `wordAt` (boundaries), `embedUrl` (null without id; start/end rounding),
  `AYAH_COUNTS` length 114 and sum 6236. `Quran.test.tsx` (mock `../api`): renders 114 cells from a stubbed coverage;
  clicking a cell requests that surah's ayah 1; an empty ayah shows the nearest button and clicking it requests the
  nearest ayah; a segment with `isharati: false` shows `quran.noPose` and no avatar canvas; a segment without
  `youtube_id` renders no iframe.

- [ ] **Step 1:** write the tests; **Step 2:** see them fail; **Step 3:** implement; **Step 4:** `npx vitest run &&
  npm run build`; **Step 5:** visual check with the API in local mode (`ISHARATI_QURAN_DIR=D:\islam\hub_build`) and
  `npx vite`: open `/?ui=en#quran`, pick 114:1, play on realistic then cartoon, confirm the YouTube embed starts at
  the segment time; repeat in `?ui=ar`; save screenshots; **Step 6:** commit — "Landing page: Qur'an in sign language
  dashboard".

---

### Task 3: Space wiring, e2e, README, deploy

**Files:**
- Modify: `deploy/space/Dockerfile` (env `ISHARATI_QURAN_OWNER=FatimahEmadEldin`), `scripts/web/e2e.py` (open `#quran`,
  select 114:1, assert a pose loads and the iframe `src` contains `start=`; screenshot `docs/paper/figures/ui_quran.png`),
  `README.md` (the dataset contribution: three repositories, figures from the data cards, the dashboard screenshot).
- [ ] Steps: implement; run `py -m pytest -q`, `cd web && npx vitest run && npm run build`, and the e2e against a local
  server in local mode; commit; push `main`; redeploy the Space (`env -u HF_TOKEN py scripts/hub/deploy_space.py
  --update`) once all three datasets are uploaded; smoke-test `/api/quran/coverage` and one pose on the Space.
