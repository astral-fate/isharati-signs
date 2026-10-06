---
title: Isharati
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Islamic answers signed in ASL, ArSL, TİD and ISL
---

# Isharati (إشارتي)

The landing page is a React app in web/, built by the Dockerfile's first stage; the previous page is at /classic.

Ask a question about Islam in English, Arabic, Turkish or Urdu; the answer is built only from cited Qur'an verses and
authenticated hadith, translated into sign glosses, and signed in the matching sign language: English to ASL, Arabic to
ArSL, Turkish to TİD, Urdu to ISL. Studio signs any text you write, without retrieval. The signer is a skeleton or a 3D
avatar, either a cartoon or a realistic one (original, hijab, or shemagh and thobe).

## Before making anything public

- Turn `ISHARATI_URDU_WSLP` off. Urdu coverage figures are measured with it on.
- The committed clips and hero pose (`web/public/clips/poses/*.json`, `web/src/assets/hero-pose.json` and the rendered
  clips) use signs from ASL Citizen (non-commercial, no redistribution) and sources whose permission is still to be
  requested (GDM Islamic signs, the Qur'an curriculum, the TİD dictionary). Re-render them from redistributable sources,
  or get permission, first.

## How this Space runs

| Part | Where |
|---|---|
| Data: Qur'an and hadith corpora, embeddings, sign lexicons and poses | private dataset `ISHARATI_HUB_DATASET`, downloaded at start-up with `HF_TOKEN` |
| Language models | OpenRouter (`OPENROUTER_API_KEY`); free models by default, override with `ISHARATI_MODELS` |
| Question embeddings | Hugging Face inference, `intfloat/multilingual-e5-large` (same model as the passage index) |
| Application | FastAPI server and single-page interface (this container, port 7860) |

## Settings

Secrets: `HF_TOKEN` (read access to the dataset), `OPENROUTER_API_KEY`.
Variables: `ISHARATI_HUB_DATASET` (e.g. `user/isharati-data`), optional `ISHARATI_LANGS` (`en,ar,tr,ur`) and `ISHARATI_MODELS`
(comma-separated OpenRouter model ids, tried in order).

The first start downloads and unpacks the data (a few minutes); later starts of the same revision skip it.
