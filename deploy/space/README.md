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

Ask a question about Islam in English, Arabic, Turkish or Urdu. The answer is built only from cited Qur'an verses and
authenticated hadith, translated into sign glosses, and signed in the matching sign language: English to ASL, Arabic to
ArSL, Turkish to TİD, Urdu to ISL. Studio signs any text you write, without retrieval. The Academy teaches the
alphabet and the signed Qur'an and hadith. The signer is a skeleton or a 3D avatar, cartoon or realistic (original,
hijab, or shemagh and thobe).

Code, technical report and data description: https://github.com/astral-fate/isharati-signs

## How this Space runs

| Part | Where |
|---|---|
| Data: Qur'an and hadith corpora, embeddings, sign lexicons and poses | dataset `ISHARATI_HUB_DATASET`, downloaded at start-up with `HF_TOKEN` |
| Academy alphabet, Qur'an text, precomputed answers to the suggested questions | dataset `isharati-app-data`, fetched on first use |
| Signed Qur'an and hadith | the `quran-sign-*` and `hadith-sign` datasets |
| Language models | OpenRouter (`OPENROUTER_API_KEY`); models in `ISHARATI_MODELS`, tried in order |
| Question embeddings | Hugging Face inference, `intfloat/multilingual-e5-large` (the model of the passage index) |
| Application | FastAPI server and React page (this container, port 7860) |

The first start downloads and unpacks the data (a few minutes); later starts of the same revision skip it.

## Run it yourself

The keys of this demo are the authors'. To run the app on your own machine or your own Space, use your own keys:
[`.env.example`](.env.example) lists every setting, what it is for and where to get each key (OpenRouter, NVIDIA NIM,
Groq for Whisper speech-to-text, a Hugging Face token, and optionally a Neon database for the review dashboard).

```bash
git clone https://github.com/astral-fate/isharati-signs && cd isharati-signs
pip install -e .
cp .env.example .env          # fill in your keys
python -c "from isharati import hub; hub.ensure_data()"
PYTHONPATH=src python -m isharati.app.server     # http://127.0.0.1:8765
```

The datasets are private (several sign sources do not allow redistribution): ask the authors for access, and your own
read token works.

On a Space of your own, set the keys under Settings → Variables and secrets: `HF_TOKEN` and `OPENROUTER_API_KEY` as
secrets, `ISHARATI_HUB_DATASET` (e.g. `user/isharati-data`) and optionally `ISHARATI_MODELS` as variables.
