# Qur'an in sign language: three dataset repositories and a contributions dashboard

Date: 2026-10-02 · Status: design approved in conversation, awaiting spec review

Builds on: `2026-10-02-landing-page-design.md` (the dashboard is a section of that page and reuses its avatar,
style switch and keypoint viewer).

## Goal

Publish the Qur'an sign-language ground truth already processed in `D:\islam\gathring data\dataset` as three
standalone Hugging Face dataset repositories, and show them on the landing page as Isharati's dataset contribution:
figures, per-surah coverage, and an ayah viewer that plays each segment on the avatar next to the original
interpreter. That viewer is the first version of a later ayah library: pick an ayah, watch its sign-language
recitation, choose the style.

## Decisions taken

| Question | Decision |
|---|---|
| Visibility | **Private** repositories. The videos belong to the King Fahd Complex, the Mukhtasar tafsir centre and the curriculum channel, and none states a licence; the cards say "permission to be requested", as for the other YouTube-derived sets. |
| Host | Hugging Face datasets, one repository per source, grouped in a private collection |
| Videos | Not uploaded. Every segment links to the original YouTube video at its timestamp. |
| Tafsir name | The playlist is *al-Mukhtasar fi Tafsir al-Qur'an al-Karim* (المختصر في تفسير القرآن الكريم), not Tafsir al-Muyassar |

## 1 · The three repositories

| Repository | Source | Segments | Surahs |
|---|---|---|---|
| `FatimahEmadEldin/quran-sign-kfc` | King Fahd Complex, translation of the meanings of the Qur'an in sign language (6 releases) | 592 recitation, 468 tafsir, 6 intro | 38 (1, 78–114) |
| `FatimahEmadEldin/tafsir-mukhtasar-sign` | al-Mukhtasar fi Tafsir al-Qur'an in sign language (playlist `PLofjBfpgjcQxcM8eMrsiNUFjip13cfCin`) | 6,091 tafsir, 35 intro | 112 (1 and 7 missing) |
| `FatimahEmadEldin/quran-sign-curriculum` | Al-Kharj school curriculum, 22 short surahs | 207 recitation | 22 |

**Layout** (identical in all three):

```
README.md                   data card
manifest.jsonl              one row per segment (below)
surahs.json                 per-surah counts: segments, ayahs, recitation/tafsir, seconds
holistic/<sss>.npz          full MediaPipe Holistic per surah, keyed by segment id:
                            <id>/pose [T,33,4], <id>/left_hand [T,21,3], <id>/right_hand [T,21,3],
                            <id>/face [T,478,3], <id>/time [T]
isharati/<sss>.npz          the same segments as Isharati poses [T,50,3] float16 at 25 fps, keyed by segment id
```

**Manifest row**: `id`, `type` (`recitation` | `tafsir` | `intro`), `surah`, `surah_name`, `ayahs` (list of
`[surah, ayah]`), `text_uthmani` (the ayah or ayahs), `tafsir_text` (tafsir segments: the signed tafsir text from
the aligned transcript), `words` (word, start, end, frame range), `start`, `end` (seconds in the source video),
`fps_source`, `frames`, `detection_rate` (pose, hands, face), `youtube_id`, `youtube_url` (with `&t=<start>s`),
`video_title`. Rows whose video has no `info.json` keep `youtube_id: null`.

**Isharati conversion**: body joints 0, 12, 14, 16, 11, 13, 15 and the neck (mid-shoulders), the two hands; NaN for
undetected joints, interpolated over time (`pose.keypoints.interpolate_missing`), neck-centred and divided by the
median shoulder width (`pose.keypoints.normalize`), then resampled from the segment's `time` stamps to 25 fps (the
tafsir videos are 30 fps with stride 2). Segments with no shoulders detected are kept in `holistic/` and listed in
the manifest with `isharati: false`.

**Data card**: what each source is and who owns it, the licence status, how segments were made (Whisper timing
aligned to the fixed Qur'an text, so ASR errors never become labels; only timing is used), the layout above, figures
(segments, ayahs, hours, detection rates), per-surah coverage, known gaps, and how to load a surah in Python.

**Publishing script**: `scripts/hub/publish_quran_sign.py <source> [--dry-run]` builds the folder in a temporary
directory from `D:\islam\gathring data\dataset\<source>` and the video `info.json` files, verifies counts against the
manifest, uploads, and adds the repository to the collection "Qur'an in sign language". It refuses to overwrite a
repository whose card was edited on the Hub unless `--update` is given (same rule as `deploy_space.py`).

## 2 · Contributions dashboard (landing page section)

A section **"Qur'an in sign language"** (`id="quran"`, nav link), after the lexicon section:

- **Three cards**, one per repository, with animated counters (segments, ayahs, surahs, hours) and the licence note.
- **Surah map**: 114 cells in mushaf order; each cell's colour shows recitation, tafsir, both or none, across all
  three sources; hover shows the surah name and counts; clicking opens it in the viewer.
- **Ayah viewer**:
  - choose surah and ayah, then the source and type available for it (KFC recitation, KFC tafsir, curriculum
    recitation, Mukhtasar tafsir);
  - the Uthmani text with each word highlighted while it is signed (recitation segments have word timings), and the
    tafsir text for tafsir segments;
  - the signing on the avatar with the Cartoon / Realistic switch, or as keypoints, using the landing page's
    `AvatarView`, `KeypointCanvas`, `StyleSwitch` and `usePlayer`;
  - the original interpreter's video embedded from YouTube (`youtube-nocookie.com`, starting at the segment's
    time), side by side with the avatar, so a judge can compare them; hidden when `youtube_id` is null.

Interface strings in the four interface languages, like the rest of the page.

**API** (FastAPI, same app):

- `GET /api/quran/coverage` → per source: totals and per-surah counts (from each repository's `surahs.json`).
- `GET /api/quran/ayah?surah=&ayah=` → the segments that cover that ayah, across sources: manifest rows without the
  keypoints.
- `GET /api/quran/pose/{source}/{segment_id}.json` → `{fps: 25, frames}` from `isharati/<sss>.npz`.

**Data on the Space**: `isharati.quran_data` downloads `manifest.jsonl` and `surahs.json` for each source at start-up
(small), and a surah's `isharati/<sss>.npz` on first request, cached on disk (`hf_hub_download`, the Space's
`HF_TOKEN`). The 8.5 GB of Holistic data is never downloaded by the Space. Locally, the same module reads the
repositories' build folders when `ISHARATI_QURAN_DIR` is set, so the dashboard works offline.

## Errors

A surah not in any source: its cell is dark and the viewer says so. A segment whose Isharati pose is missing
(`isharati: false`): the viewer shows the interpreter's video only, with a note. A download failure: a readable
message, and a retry on the next request. The private-Space HTML and quota errors use the landing page's handling.

## Testing

- Python: Holistic → Isharati conversion (NaN interpolation, 30 fps stride 2 → 25 fps, neck-centred, shoulder
  scale), manifest row building (ayah lists, YouTube URL with timestamp, null when no `info.json`), per-surah packing
  round trip, `surahs.json` totals equal the manifest, the three endpoints on a small fixture dataset, path safety for
  `{source}` and `{segment_id}`.
- Publishing: `--dry-run` on each source prints counts that match the manifests (1,066; 6,126; 207).
- Front end: Vitest for the surah-map colouring and the ayah → segments selection; the end-to-end check opens
  an-Nas 114:1 (KFC recitation), plays it on both styles, and checks the YouTube embed's start time.

## Out of scope

The full ayah library (search, bookmarks, reciter choice, playlists), making the repositories public, new
recordings, and al-Fatiha / al-A'raf for al-Mukhtasar (not in the playlist download).
