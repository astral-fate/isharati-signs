# Isharati landing page and four-language translator: design

Date: 2026-10-02 · Status: approved in conversation, awaiting spec review

## Goal

A landing page good enough to impress the Bathel AI Challenge judges (build days Oct 4–6, 2026), with the working
translator embedded in it, real signing videos from every avatar, animated figures for the lexicon and coverage, and
all four supported languages translating: English → ASL, Arabic → ArSL, Turkish → TİD, Urdu → ISL.

Success means:

- a visitor understands what Isharati is from the first screen and reaches a working translator in one click;
- a question in any of the four languages returns a cited answer, its glosses and a signed avatar/keypoint video;
- every number on the page comes from the data (no invented figures), including honest coverage per language;
- it runs on the existing private HF Space, at one URL, with no new hosting.

## Decisions taken

| Question | Decision |
|---|---|
| Urdu sign sources | CISLR (AFL-3.0) **and** WSLP (CC BY-NC-ND), WSLP behind `ISHARATI_URDU_WSLP` (on while the Space is private; turn off before making it public) |
| Interface languages | English, Arabic, Turkish, Urdu; Arabic and Urdu right-to-left |
| Visual direction | Dark and immersive: night-sky navy, glowing teal keypoints, gradient headings |
| Layout | Story-first hero, translator one scroll down ("Try it") |
| Stack | React + Vite + TypeScript + Framer Motion, built into the Space's Docker image and served by FastAPI |

## 1 · Backend: Turkish and Urdu

**Answering** already exists in `isharati.retrieval.turkish_urdu` (Qur'an in QuranEnc translations: Turkish Rowwad,
Urdu Junagarhi; the same graded HadeethEnc hadith; the same citation and support checks). `Pipeline("tr" | "ur")`
uses `turkish_urdu.corpus(lang)` and the shared `engine.answer`.

**Glossing.** `glossing/asl.py`'s `ASLLLMGlosser` becomes a language-parameterised `LLMGlosser(lexicon, client,
spec)`, where `spec` carries the system prompt, the gloss normaliser and the sign-language name. The rule is unchanged:
the model may use only glosses from the allowed list, and the lexicon (not the model) decides what is signed.

- Turkish → TİD: glosses are Turkish words from the 1,373-sign TİD dictionary
  (`lexicon_tr/dictionaries/entries_tid_dictionary.jsonl`); matching uses Turkish casing (İ/i, I/ı) via
  `turkish_urdu.tr_norm`.
- Urdu → ISL: glosses are English labels from CISLR (3,964 signs) plus WSLP (4,055) when `ISHARATI_URDU_WSLP=1`.
  The two lists are merged at load time; CISLR wins a tie, so turning WSLP off only removes signs.
- English keeps its current prompt and behaviour; `ASLLLMGlosser` remains as a thin alias so existing tests and the
  paper's reported runs are unchanged.

**Signing** reuses `MissingAwarePoser`, `proportions.apply`, `SkeletonRenderer` and `pose.npy` unchanged (all four
lexicons store `[T,50,3]` poses the same way). Words with no sign are marked and held, as now.

**Back-translation** for Turkish and Urdu (no recorded templates exist) uses a `SampledRecognizer`: the answer's own
signs plus a fixed sample of 200 distractor signs. 1-NN over every sign (8,000 for Urdu) would take minutes per answer.
The score stays labelled circular, as for English without templates.

**Cache stamp.** `stamp(lang)` and `question_id(lang)` cover `tr` and `ur` (lexicon path per language, plus the
Turkish/Urdu retrieval and glossing modules).

**API.**

- `POST /api/ask` accepts `lang ∈ {en, ar, tr, ur}`.
- `GET /api/stats` returns, per language: sign-language name, number of signs, number of distinct glosses, corpus
  passages, measured coverage, plus the dataset licences and the avatar count. The figures are generated once by
  `scripts/eval/app_stats.py` into `src/isharati/app/static/stats.json` (the Space has no `results/` folder). Coverage
  is one measure for all four languages: content-word token coverage of the Qur'an and hadith corpus
  (`results/eda.json`, `all.content_token_coverage`; Urdu through ISL's English glosses, `isl_on_en`). With
  `ISHARATI_URDU_WSLP` off, the Urdu sign count drops to CISLR's.
- The avatar clips are listed in a static `clips/clips.json` next to the clips (section 2.4); no endpoint.

**Text to sign (Studio).** `POST /api/sign {text, lang}` signs the user's own text, up to 1,000 characters, without
retrieval or answer generation: `Pipeline.sign_text(text)` → glossing → pose → avatar, reusing `_sign`. The report
has `"mode": "text"` and no sources; saved runs live under their own ids (`t<lang><hash>`), so a text never replays
an answer. Glossing still uses the LLM (sign order, lexicon-only signs). For English, when every LLM backend fails
(e.g. the free quota), a `FallbackGlosser` uses the rule-based `ASLRuleGlosser` (word by word, no reordering) and
the report says so (`glosser: "rule"`). Arabic, Turkish and Urdu have no rule glosser and report the error. A text
none of whose words has a sign returns 422 with a readable message.

**Space data.** `ISHARATI_LANGS=en,ar,tr,ur`, so `hub.ensure_data()` also fetches `tr/` and `ur/` (about 0.3 GB more).

## 2 · Front end

### 2.1 Structure

A new `web/` package: React 18, Vite, TypeScript, Framer Motion, three.js and `@pixiv/three-vrm` from npm.

```
web/
  src/
    main.tsx, App.tsx
    i18n/{en,ar,tr,ur}.json, i18n.ts        interface strings; sets <html lang dir>
    api.ts                                  typed calls; turns HTML/429/503 replies into readable errors
    sections/  Nav, Hero, Translator, HowItWorks, Lexicon, Avatars, Sources, Footer
    components/ Counter, KeypointCanvas, AvatarView, AvatarPicker, GlossChips, ClipCard, LangTabs
    avatar/     retarget.ts (port of static/avatar.js), outfits.ts
    assets/     hero-pose.json (a bundled real pose for the hero figure)
  public/clips/ <avatar>-<phrase>.webm + .jpg poster
```

Interface language switcher: English, العربية, Türkçe, اردو. Arabic and Urdu set `dir="rtl"`.

### 2.2 Sections (top to bottom)

1. **Nav**: logo, section links, interface-language switcher.
2. **Hero**: dark navy; a glowing keypoint figure signing a real bundled pose on a loop (`KeypointCanvas`); headline
   and one line; counters that roll up to the `/api/stats` figures (4 languages, total signs, approved passages); a reel of the
   avatar clips beside the figure; "Try it" scrolls to the translator.
3. **Translator**:
   - language tabs EN→ASL · AR→ArSL · TR→TİD · UR→ISL, defaulting to the interface language;
   - question box with example chips per language;
   - answer with source links;
   - gloss chips that light up as each sign plays, red for no sign yet;
   - video panel with an Avatar / Keypoints toggle (Keypoints draws the glowing skeleton from the pose frames),
     avatar thumbnails for the six avatars, play / scrub / speed / export.
   - two tabs at the top of the section: **Ask a question** (the flow above) and **Translate my text**, a
     textarea for the user's own content in the chosen language. It shows the text, its glosses and the same viewer
     (style switch, playback, export), labelled "your text, not checked against the sources". The nav's **Studio**
     link (`#studio`) opens this tab.
4. **How it works**: five steps animated on scroll: question → approved sources → cited answer → glosses → signed
   pose and avatar.
5. **Lexicon and coverage**: a card per language with counters (signs, sources, corpus size) and a coverage bar.
6. **Avatars**: one card per avatar playing its clip on hover (autoplay muted on mobile when in view); clicking a card
   selects that avatar in the translator and scrolls there.
7. **Sources and licences**, **team**, **footer**.

The current page stays at `/classic`.

### 2.3 Avatar component

`AvatarView` ports `static/avatar.js` (rest-direction retargeting from bone positions, so the Rocketbox A-pose works)
and `outfits.js` into a React component driven by pose frames and a time value. One WebGL renderer is reused when the
avatar changes. If WebGL or the VRM fails, the panel switches to Keypoints with a short note.

### 2.3a Style switch: cartoon or realistic

A demo-ready switch under the signing avatar (and the same two groups in the avatar gallery): **Cartoon** (the
stylised pixiv VRM, with an outfit choice of hijab, shemagh and thobe, or original) or **Realistic** (the five
Microsoft Rocketbox characters converted in Blender, with their own modest clothes). Switching mid-answer keeps
signing from the same moment, because the pose clock is shared; each style remembers the last avatar chosen in it, so
flipping back and forth during a presentation lands on the same two avatars. A one-line note under the switch says
what each style is, for explaining it in the demo.

### 2.4 Avatar signing clips

Real videos of every avatar signing, for the hero reel and the avatar gallery.

- `scripts/avatars/record_clips.py` stitches short phrases from the lexicons with the existing poser (for example
  ASL "ALLAH / PRAYER", ArSL «الصلاة», TİD and ISL equivalents), loads the built page headless (Playwright), plays the
  pose on each avatar and records the canvas with `MediaRecorder` to WebM (VP9), plus a JPEG poster.
- Each avatar gets at least one clip; the five realistic avatars between them cover all four sign languages.
- Budget: 3–6 s per clip, ≤ 1.5 MB each, ≤ 15 MB in total; generated once and committed to `web/public/clips/`, with a
  `clips.json` manifest (avatar, sign language, phrase, glosses).

## 3 · Deployment, errors, testing

**Deployment.** A two-stage Dockerfile: stage 1 (`node:22-slim`) runs `npm ci && npm run build` in `web/`; stage 2 is
the current Python image and copies `web/dist`. FastAPI serves the React app at `/`, `/classic` for the old page, and
the API under `/api`, `/pose`, `/video`, `/static`. `deploy_space.py` also stages `web/` (without `node_modules`).

**Errors.** Readable messages for: an HTML reply (sign-in or proxy page), the free-quota 503, "referred" (rulings go
to a scholar) and "unanswered" (no supported answer). A missing pose or failed avatar falls back to Keypoints. If
`/api/stats` fails, counters use built-in figures from the last measurement so the hero is never empty.

**Testing.**

- Python (pytest): Turkish and Urdu glossers with a fake LLM; Turkish casing; WSLP flag on and off; `/api/stats`
  shape and values; cached replay for `tr` and `ur`; the existing 85 tests stay green.
- Front end: `tsc` and `vite build`; Vitest for `GlossChips` timing, `Counter`, the API error mapping and RTL
  switching.
- End to end (Playwright against the built app on a local server): cached English and Arabic answers, one live
  Turkish and one live Urdu question; avatar loads, switching avatars works, clips play, no page errors. Fresh
  screenshots go into the README and the report.

## Out of scope

Facial expressions, accounts, a separate Vercel deployment, new sign recordings, and raising Turkish coverage
(currently about 16% of words; shown as measured).
