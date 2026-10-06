# Sources, rights and use of AI · المصادر والحقوق واستخدام الذكاء الاصطناعي

Every source Isharati uses: the Qur'an and hadith texts, the sign dictionaries and datasets of the four sign languages,
the signed Qur'an and hadith videos, the models and tools, the avatars and the infrastructure. For each one: what it is,
where it comes from, what Isharati uses it for, and its terms. The last sections state the rights and how AI was used to
build the system. An Arabic version of the essentials follows at the end ([العربية](#التوثيق-بالعربية)).

Numbers are those of the current release (October 2026); `src/isharati/app/static/stats.json` and the dataset cards on
the Hugging Face Hub hold the live figures.

**Contents**
1. [Qur'an and hadith texts](#1-quran-and-hadith-texts)
2. [Sign lexicons, by language](#2-sign-lexicons-by-language)
3. [Signed Qur'an and hadith videos](#3-signed-quran-and-hadith-videos)
4. [Evaluation data](#4-evaluation-data)
5. [Models](#5-models)
6. [Tools and libraries](#6-tools-and-libraries)
7. [Avatars](#7-avatars)
8. [Infrastructure and published datasets](#8-infrastructure-and-published-datasets)
9. [Rights and terms](#9-rights-and-terms)
10. [Use of AI](#10-use-of-ai)
11. [Academic references](#11-academic-references)
12. [التوثيق بالعربية](#التوثيق-بالعربية)

---

## 1. Qur'an and hadith texts

The answers are built only from these texts, and every answer cites the passage it comes from.

| Source | What Isharati uses | Where | Terms |
|---|---|---|---|
| **IslamicEval 2025** canonical Qur'an text (Mubarak et al., ArabicNLP 2025) | The Arabic Qur'an (6,236 ayat, Uthmani script) for Arabic retrieval and citation | IslamicEval 2025 shared task release | Shared-task release |
| **Tanzil Project** text, through the **AlQuran Cloud API** | The Qur'an by surah and ayah (Uthmani and Simple-Clean) shown in the app's Qur'an pages | [tanzil.net](https://tanzil.net) · [api.alquran.cloud](https://alquran.cloud/api) | Creative Commons Attribution 3.0 (verbatim, with attribution) |
| **QuranEnc** (Encyclopedia of the Translations of the Meanings of the Qur'an) | Translations: English (Saheeh International), Turkish (Rowwad), Urdu (Junagarhi) | [quranenc.com](https://quranenc.com) API | QuranEnc terms (to be confirmed) |
| **HadeethEnc** (Encyclopedia of Translated Prophetic Hadiths) | Authenticated hadith with grades and explanations: 3,574 (Arabic), 2,328 (English), 2,150 (Turkish), 2,220 (Urdu) | [hadeethenc.com](https://hadeethenc.com) API | HadeethEnc terms (to be confirmed) |
| **hadith-api** (fawazahmed0) | The text of each signed hadith as printed in its collection (Bukhari, Muslim, Abu Dawud, Tirmidhi, Nasa'i, Ibn Majah, Malik, Nawawi's Forty, Forty Qudsi): the label of every sample in `hadith-sign` | [github.com/fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) | Public domain (Unlicense) |

## 2. Sign lexicons, by language

A lexicon entry is one recorded sign converted to Isharati's pose format (MediaPipe Holistic keypoints, 50 joints, 25 fps).
No video is stored or redistributed: only keypoints, with a link to the source.

**Coverage** (content words of the Qur'an and hadith that the lexicon signs): English 83.1%, Turkish 75.7%, Arabic 72.9%
(74.2% with proper names fingerspelled), Urdu 72.1%.

### Arabic → Arabic Sign Language (ArSL) · 6,663 signs

When several sources have the same gloss, the first in this order is kept (Saudi signers first, other dialects to fill
gaps; `scripts/lexicon/arsl/merge.py`).

| Source | Signs used | Where | Terms |
|---|---|---|---|
| Reviewed recordings and fingerspelling decisions (project team) | 26 | this repository | project |
| **KArSL-502** (King Fahd University) | 840 | [kaggle.com/datasets/yousefdotpy/karsl-502](https://www.kaggle.com/datasets/yousefdotpy/karsl-502) | Research use |
| Qur'an curriculum in sign language (Al-Kharj) | 304 | [YouTube channel](https://www.youtube.com/channel/UCRpE_0EQWEJhf_8WCW6hxKg) | Not stated; permission to be requested |
| **Tawasol Center** vocabulary lessons (Saudi interpreters) | 196 | [youtube.com/@tawasol9282](https://www.youtube.com/@tawasol9282) | Not stated; permission to be requested |
| Saudi and Arabic dictionaries, explained entry by entry | 82 | [YouTube channel](https://www.youtube.com/channel/UCQiTnBVnBfvVYilcfjpKkMA) | Not stated; permission to be requested |
| **Unified Arabic Sign Dictionary** (ArabicSignLanguage) | 1,424 | [YouTube channel](https://www.youtube.com/channel/UCg85d6jX2GANVz1msIvFvnA) | Not stated; permission to be requested |
| **Kuwaiti**, children's and scouts' sign dictionaries (DisabilityApps) | 978 + 87 + 140 | [YouTube channel](https://www.youtube.com/channel/UCiILcu4Zyk_j4L8tEe2UmVQ) | Not stated; permission to be requested |
| «إشارتي هي لغتي» (Ayman Abbas), captioned vocabulary videos | 5 | [youtube.com/@aymanabbas3623](https://www.youtube.com/@aymanabbas3623) | Not stated; permission to be requested |
| Heba Abdelrahman, signed song (the sign «الدنيا», cut at the Whisper-timed word) | 1 | [YouTube Short](https://www.youtube.com/shorts/hpbp2V1IGgA) | Not stated; permission to be requested |
| Arabic Deaf instruction channels (20 channels, 249 videos, aligned by speech recognition) | 2,406 | see the list below | Not stated; permission to be requested |
| Levantine Shorts (Jordan / Palestine) | 149 | [YouTube playlist](https://www.youtube.com/playlist?list=PLqZ7C4R4F6EJOnp0lRkKSM9RswCr6WBSG) | Not stated; permission to be requested |
| **ArabSign** sentence clips, glosses vetted against the signed sentence | 56 | [github.com/Hamzah-Luqman/ArabSign](https://github.com/Hamzah-Luqman/ArabSign) | Not stated in the paper |

<details>
<summary><b>The 20 Arabic Deaf instruction channels and playlists</b> (click to open)</summary>

| Channel or playlist | Link |
|---|---|
| المصطلحات الإشارية العربية | [playlist](https://www.youtube.com/playlist?list=PLYFAA2KANsca-IuKR2YJnknBCXjtfYQhf) |
| الدين | [playlist](https://www.youtube.com/playlist?list=PLYFAA2KANscZJzmWk_yWveUxUl4DSFpsM) |
| السيرة النبوية | [playlist](https://www.youtube.com/playlist?list=PLYFAA2KANscaIpzGb8P9pnxBT-Ie2V9qd) |
| تفسير جزء عم | [playlist](https://www.youtube.com/playlist?list=PLYFAA2KANscaHyllcipTc7KiTHLY1RlIV) |
| مصطلحات دينية بلغة الإشارة | [video](https://youtu.be/Qs22zNf-1B0) |
| الإشارة لغتي | [playlist](https://www.youtube.com/playlist?list=PLtFfPbdJciOfIPuolrn4SkSCiED6g9hHY) |
| الدين وما يتعلق به بلغة الإشارة | [video](https://youtu.be/WXqQssArQ6E) |
| أساسيات لغة الإشارة | playlist (the identifier recorded in our download log is incomplete) |
| حسين العورتاني، خبير لغة إشارة الصم | [channel](https://www.youtube.com/channel/UCARkwpk-jW6UwHyNXcd7wyw) |
| كلمات متنوعة بلغة الإشارة (دبي) | [playlist](https://www.youtube.com/playlist?list=PLAW205Tn_URx_yBou0HH-9SxNaoIEzcDV) |
| تعلم لغة الإشارة، وفاء الريامي | [playlist](https://www.youtube.com/playlist?list=PLJqrb8bafBc8nylNiSMvLlB-6Wi0gPqSH) |
| إشارتي هي لغتي، أيمن عباس | [channel](https://www.youtube.com/@aymanabbas3623) |
| لغة الإشارة الأردنية، أسامة طهراوي | [channel](https://www.youtube.com/channel/UC0iJnWIOaDRopy-U5nH6iTw) |
| صوت الأنامل | [channel](https://www.youtube.com/@sawtalanamil7234) |
| المعلم للأشخاص ذوي الإعاقة السمعية | [channel](https://www.youtube.com/@teacher_of_deaf) |
| قائمة إشارة المختارة (دين، أنبياء، عبادات) | playlist (the identifier recorded in our download log is incomplete) |
| صفة العمرة بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY6QJObgFM11lyxJoIsIjqSN) |
| أحكام الحج بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY7u820mdbW7DXHYcwX8K1q2) |
| شرح كتاب الدروس المهمة لعامة الأمة بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY46PpMhLtVCxlvsXkmjIqdg) |
| من أجل حياة سعيدة بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY7bYeZbeK7RbPOGpD-Ao5UM) |
| ترجمة خطب يوم عرفة والعيدين بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY7aRFV9me76HqRG2yLX8fps) |
| الدورة العلمية 14 بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY7BMfqm7fA_HPOoikdsJr88) |
| محبة النبي ﷺ على السنة بلغة الإشارة | [playlist](https://www.youtube.com/playlist?list=PL4sH_2mzNxY74EsrJsNqwvswnqOBwseBM) |
| شرح كتاب الآداب الإسلامية بلغة الإشارة | [video](https://youtu.be/sGW3nQ-0ZRg) |

</details>

### English → American Sign Language (ASL) · 3,677 signs

| Source | What Isharati uses | Where | Terms |
|---|---|---|---|
| **ASL Citizen** (Microsoft Research, Desai et al. 2023) | Isolated signs, the main English lexicon | [microsoft.com/en-us/research/project/asl-citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Microsoft Research licence: non-commercial, no redistribution |
| **WLASL** (Li et al. 2020) | Isolated signs | [github.com/dxli94/WLASL](https://github.com/dxli94/WLASL) | C-UDA, research use |
| **MS-ASL** (Joze & Koller 2019) | Isolated signs | [microsoft.com/en-us/research/project/ms-asl](https://www.microsoft.com/en-us/research/project/ms-asl/) | C-UDA, research use |
| **YouTube-ASL** (Uthus et al. 2023) | Continuous signing: 186 missing scriptural terms mined with a weakly supervised spotter; converted keypoints published as `youtube-asl-isharati` | [github.com/google-research/google-research/tree/master/youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl) | CC BY 4.0 |
| GDM, 40 Islamic signs | Islamic terms, one Deaf signer | [video](https://www.youtube.com/watch?v=L-lWEt7OeJk) | Not stated; permission to be requested |
| Noor For Sign, Ramadan vocabulary | Ramadan terms | [video](https://www.youtube.com/watch?v=VWzAH0j56f4) | Not stated; permission to be requested |
| Obscure ASL | One word per video | [youtube.com/@obscureasl](https://www.youtube.com/@obscureasl) | Not stated; permission to be requested |

### Turkish → Turkish Sign Language (TİD) · 4,827 signs

| Source | What Isharati uses | Where | Terms |
|---|---|---|---|
| **Türk İşaret Dili Sözlüğü** (Turkish Sign Language Dictionary) | The main TİD lexicon (~5,000 clips), opposite pairs re-cut into one sign per word | [YouTube channel](https://www.youtube.com/channel/UCq2tlBndwWjJb2Zlf9QoOzQ) | Not stated; permission to be requested |
| **TiDiSLaM** | Islamic terms in TİD | [YouTube channel](https://www.youtube.com/channel/UC7zHDrJRMxSd65cE3vxfksg) | Not stated; permission to be requested |
| **AUTSL** (Ankara University, Sincan & Keles 2020) | Signs the dictionary lacks (the medoid recording of each) and further signers | [huggingface.co/datasets/aipieces/AUTSL](https://huggingface.co/datasets/aipieces/AUTSL) | Academic and research use only, no commercial use |
| Cüneyt Küçükoğlu, «Hareketli Sözlük» | Gap fillers, pending review | YouTube channel | Not stated; permission to be requested |
| «Ellerimiz Konuşuyor» (Our hands are speaking) and The Sign Polyglot | Gap fillers, pending review | YouTube | Not stated; permission to be requested |
| TİD alphabet and Companion names | Fingerspelling and name signs | one signer | Not stated |

### Urdu → Indian and Pakistani Sign Language (ISL / PSL) · 10,233 signs

| Source | Signs used | Where | Terms |
|---|---|---|---|
| **CISLR** (Joshi et al. 2022) | 3,965 | [github.com/cislr/cislr](https://github.com/cislr/cislr) | AFL-3.0 |
| **WSLP 2025** | 4,056 | [huggingface.co/datasets/Exploration-Lab/WSLP](https://huggingface.co/datasets/Exploration-Lab/WSLP) | CC BY-NC-ND 4.0 |
| **ISLRTC** dictionary (Indian Sign Language Research and Training Centre) and manual alphabet | 966 | [YouTube channel](https://www.youtube.com/channel/UC3AcGIlqVI4nJWCwHgHFXtg) | Not stated; permission to be requested |
| **PSL dictionary**, FESF / Deaf Reach | 1,246 | [psl.org.pk](https://psl.org.pk) | All rights reserved; permission to be requested |

## 3. Signed Qur'an and hadith videos

Each segment is aligned to the ayah or hadith it signs; the label is always the fixed text (Qur'an or the hadith as printed
in its collection), never a transcript. Only keypoints and links are published; the videos belong to their publishers.

### Qur'an in sign language · 7,980 segments, 44.9 hours

| Dataset | Source | Segments | Hours | Where |
|---|---|---|---|---|
| `quran-sign-kfc` | **King Fahd Glorious Qur'an Printing Complex**, translation of the meanings in sign language | 1,066 | 7.4 | [qurancomplex.gov.sa](https://qurancomplex.gov.sa/) · [playlist](https://www.youtube.com/playlist?list=PL5yk4nUwSQItYt7cz09HBYzbvgIX4tg-g) |
| `tafsir-mukhtasar-sign` | **al-Mukhtasar fi Tafsir al-Qur'an al-Karim** in sign language | 6,127 | 34.7 | [playlist](https://www.youtube.com/playlist?list=PLofjBfpgjcQxcM8eMrsiNUFjip13cfCin) |
| `quran-sign-curriculum` | **Al-Kharj** school Qur'an curriculum, 22 short surahs | 207 | 0.3 | [YouTube channel](https://www.youtube.com/channel/UCRpE_0EQWEJhf_8WCW6hxKg) |
| `quran-sign-tebyan` | **Tebyan Qur'an**, per-ayah videos | 567 | 2.4 | [tebyanquran.com](https://tebyanquran.com/) (api.tebyanquran.com) |
| `quran-sign-diyanet-tid` | **Diyanet İşleri Başkanlığı**, «İşaret Dili ile Kur'an» (TİD) | 13 | 0.1 | [playlist](https://www.youtube.com/playlist?list=PLtApkPE49w-tQ8-zFm1QVrwrmYKauonS9) |

The Academy's Arabic alphabet comes from the same Diyanet series (episode 30, the Arabic finger alphabet), published in
`isharati-app-data`.

### Hadith in sign language · 375 samples, 301 distinct hadith, 15.5 hours

Gathered from 824 videos; the speech is transcribed (faster-whisper) and matched to the hadith-api text, and silent
videos are labelled from on-screen text or titles and checked against the corpus.

<details>
<summary><b>The hadith channels and playlists</b> (click to open)</summary>

| Channel or playlist | Sign language | Links |
|---|---|---|
| Islamweb for Deaf: الحديث الشريف بلغة الإشارة | ArSL | [playlist](https://www.youtube.com/playlist?list=PLZ25ESuPeIC7s-lWyM7Zm6GzCvXJVyj5e) · [playlist](https://www.youtube.com/playlist?list=PLP3ke6jz11_Bej-DbS2yJ-I4Ap00HFlqw) · [channel](https://www.youtube.com/channel/UCf61i7CZLHYBSm3wz7dY_yg) |
| جمعية إنسان: الأربعون النووية بلغة الإشارة | ArSL | [playlist](https://www.youtube.com/playlist?list=PLgJnHaEtnve4L-Xm3KqlYvWMAwPCAL0v7) |
| محمود حافظ: الأربعين النووية بلغة الإشارة | ArSL | [channel](https://www.youtube.com/channel/UCdSijcBzDntB9z12InQKlNw) |
| القناة التعليمية للصم: رياض الصالحين | ArSL | [playlist](https://www.youtube.com/playlist?list=PLQgY-GER8072iESE3EpXzj1ERzbKDnbaa) · [playlist](https://www.youtube.com/playlist?list=PLQgY-GER8071N5tv5bDvGB-XNqzVkmJUd) |
| القناة التعليمية للصم: شرح 30 حديثاً للنساء | ArSL | [playlist](https://www.youtube.com/playlist?list=PLQgY-GER8071C3NOVwHYrDFSQzFrg_xEZ) |
| القناة التعليمية للصم: شرح عمدة الأحكام | ArSL | [playlist](https://www.youtube.com/playlist?list=PLQgY-GER8070RzWhd20DTklkAXNkpZSOD) |
| القناة التعليمية للصم: شرح الأربعين النووية | ArSL | [playlist](https://www.youtube.com/playlist?list=PLQgY-GER8071Ur41UwYkCgZ7JLDe5vQyu) |
| حمزة ودويد للصم: الأحاديث | ArSL | [playlist](https://www.youtube.com/playlist?list=PLcc5qc9jFS6FPmXyMxgTuWvwzmMVlg351) |
| حمزة ودويد للصم: شرح الأربعين النووية | ArSL | [playlist](https://www.youtube.com/playlist?list=PLcc5qc9jFS6G24Xnn6P5Fwtb6Vy3tglqG) |
| عمرو عباس للصم: شرح الأربعين النووية | ArSL | [playlist](https://www.youtube.com/playlist?list=PLdgbEjNH7MHkEC8gjdIAiaqwhkvuhOJih) |
| نصرة محمد ﷺ بلغة الإشارة: شرح حديث للشيخ عبد الرحمن الفهيد | ArSL | [playlist](https://www.youtube.com/playlist?list=PLe2CIyE27iWC0Kani7idYAXss9ITCuWAv) |
| أحاديث متفرقة بلغة الإشارة | ArSL | [1](https://youtu.be/sOLPcM0wnHs) · [2](https://youtu.be/ECxN5ei8kYI) · [3](https://youtu.be/y1074REjsEM) |
| Eray Demir: İşaret Dili ile 40 Hadis | TİD | [playlist](https://www.youtube.com/playlist?list=PL4WaOloufEidV83XNVSgwSOLaYCKPpYHD) |
| İşaret dili ile hadisler (Türkiye) | TİD | [1](https://youtu.be/57tu4gKEGoM) · [2](https://youtu.be/6brtE5q7c-o) · [3](https://youtu.be/QhMx0toXu4Q) · [4](https://youtu.be/vLvE_2D0nVQ) · [5](https://youtu.be/1PWuKRpAPK0) · [6](https://youtu.be/k7FttKFl2yw) · [7](https://youtu.be/oJrcGZR3AbM) |

</details>

## 4. Evaluation data

Used only to measure the system; nothing from them is published.

| Source | Use | Where | Terms |
|---|---|---|---|
| **Isharah** (Saudi ArSL; 2,000 sentence annotations) | Gold glosses for the text-to-sign evaluation (`scripts/eval/text2gloss_eval.py`, `matching_eval.py`): 300 held-out and 54 religious sentences | [huggingface.co/datasets/snalyami/Isharah](https://huggingface.co/datasets/snalyami/Isharah) | CC BY-NC-ND 4.0 |
| **ArabSign** (Luqman, FG 2023) | 50 sentences with the ArSL gloss as signed | [github.com/Hamzah-Luqman/ArabSign](https://github.com/Hamzah-Luqman/ArabSign) | Not stated in the paper |
| **EGCBT** (Deaf Bible Society) | Benchmark of continuous-signing segmentation | [deafbible.com/EGCBT](https://deafbible.com/EGCBT) | Not stated |

## 5. Models

### Inside the application

| Model | Provider | What it does in Isharati |
|---|---|---|
| **Qwen 3.8 27B** (`qwen/qwen3.8-27b`), then **Qwen 3.8 Flash** | OpenRouter (paid) | Writes the cited answer from the retrieved passages, and turns it into sign glosses restricted to the lexicon |
| **NVIDIA Nemotron 3** Super 120B and Ultra 550B | NVIDIA NIM and OpenRouter (free tier) | Fallback for the two above when they are unavailable |
| **Whisper large-v3** | Groq | Speech-to-text in the audio / video mode (khutbah, lesson, recording) |
| **multilingual-e5-large** (`intfloat/multilingual-e5-large`) | Hugging Face inference | Embeddings of questions and passages for semantic retrieval (with BM25, fused by reciprocal rank) |

### In building the data (offline)

| Model | What it did |
|---|---|
| **faster-whisper large-v3-turbo** (local GPU) | Transcribed the hadith videos with word timestamps, to find which hadith each video signs; timed single signs in songs and lessons |
| **Claude Sonnet 5.5** (Anthropic, through OpenRouter) | Read on-screen hadith text in silent videos (vision) and checked translations, as proposals; every label was then verified against the hadith-api text |
| **Qwen 3.8 27B** (OpenRouter) | Glossed every signed hadith for the dataset; proposed same-meaning signs for missing Arabic words (each reviewed by hand); judged sign equivalence in the evaluation |
| **MediaPipe Holistic** (Google), `holistic_landmarker.task` | Body, hand and face keypoints and 52 facial blendshape scores from every video |
| Weakly supervised sign spotter (project's own, multiple-instance learning over YouTube-ASL keypoints) | Found 186 missing ASL signs in continuous signing |

## 6. Tools and libraries

| Tool | Use | Licence |
|---|---|---|
| **CAMeL Tools** (NYU Abu Dhabi), MSA morphology database | Arabic lemmas, part of speech, clitics and roots for matching words to signs | MIT |
| **zeyrek** | Turkish morphology (lemmas) | MIT |
| **BM25** (implemented in `src/isharati/retrieval/engine.py`) | Keyword retrieval | project |
| **yt-dlp** | Downloading the public videos the keypoints were extracted from (the videos are not kept or published) | Unlicense |
| **OpenCV**, **NumPy**, **imageio-ffmpeg**, **FFmpeg** | Video reading, keypoint processing, video export | Apache-2.0, BSD, BSD, LGPL/GPL |
| **FastAPI**, **Uvicorn**, **Pydantic** | The web server and API | MIT, BSD, MIT |
| **React**, **Vite**, **TypeScript** | The web page | MIT, MIT, Apache-2.0 |
| **three.js**, **@pixiv/three-vrm** | 3D avatar rendering in the browser | MIT |
| **Blender** and the **VRM Add-on for Blender** | Converting Rocketbox avatars to VRM 1.0 | GPL (tools only; the output avatars keep their own licence) |
| **Playwright** | End-to-end browser checks and screenshots | Apache-2.0 |
| **sacrebleu**, **jiwer** | BLEU, chrF and WER in the evaluation | Apache-2.0 |
| **XeLaTeX**, **polyglossia**, Amiri and TeX Gyre fonts | The technical report | LPPL, OFL |

## 7. Avatars

| Avatar | Source | Licence |
|---|---|---|
| Five realistic avatars (hijab, abaya, thobe with cap, red and white shemagh) | **Microsoft Rocketbox** ([github.com/microsoft/Microsoft-Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)), converted to VRM 1.0 | MIT |
| Stylised avatar, with hijab or shemagh-and-thobe outfits added | **VRM 1.0 sample** (pixiv, `VRM1_Constraint_Twist_Sample`) | VRM Public License 1.0: modification, redistribution, commercial and religious use allowed |

Details: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 8. Infrastructure and published datasets

| Service | Use |
|---|---|
| **Hugging Face Hub** | The datasets below; hosted inference of the E5 embeddings |
| **Hugging Face Spaces** (Docker) | The live demo ([spaces/FatimahEmadEldin/isharati-app](https://huggingface.co/spaces/FatimahEmadEldin/isharati-app)) |
| **OpenRouter**, **NVIDIA NIM**, **Groq** | The language and speech models of section 5 |
| **Neon** (PostgreSQL) | The sign-review dashboard's database |
| **GitHub** | This repository |

Datasets published by the project (Hugging Face, `FatimahEmadEldin/…`): `isharati-data` (corpora, embeddings, lexicons and
poses, one folder per language), `isharati-app-data` (Academy alphabet, Qur'an text, precomputed answers),
`hadith-sign`, `quran-sign-kfc`, `tafsir-mukhtasar-sign`, `quran-sign-curriculum`, `quran-sign-tebyan`,
`quran-sign-diyanet-tid`, `youtube-asl-isharati`; collections «Qur'an in sign language» and «Hadith in sign language».
Each dataset card names its sources and their terms.

## 9. Rights and terms

- **Code**: MIT License ([LICENSE](../LICENSE)).
- **Data**: each source keeps its own terms, as listed in the tables above and in each dataset card. Several sign
  sources are non-commercial (ASL Citizen, WSLP, AUTSL, Isharah) or state no licence; for those that state none, permission
  is to be requested from the publisher, and the datasets are private.
- **Videos**: no video from any source is stored in the datasets or redistributed. The datasets hold keypoints (numbers
  describing body, hand and face positions) and a link to the original video at its timestamp, so the publisher remains
  the place to watch it.
- **Religious texts**: the Qur'an and hadith are quoted verbatim with their reference and a link to the source page; the
  application never paraphrases a verse as if it were the text.
- **Review**: no sign has yet been approved by Deaf reviewers for this project; every sign is marked *pending review*
  until it is.
- **Removal**: a publisher who wants their material removed can open an issue in this repository; it is removed from the
  datasets and the lexicon.

## 10. Use of AI

**In the product.** Language models write the answer and the glosses (section 5), always under these rules:

- an answer is built only from retrieved Qur'an verses and authenticated hadith, and a sentence is kept only if 80% of
  its content words occur in the passages it cites; a sentence citing only Qur'an verses must reproduce a verse's wording;
- questions asking for a religious ruling (fatwa) are referred to scholars before any model is called;
- glosses are restricted to signs that exist in the lexicon; a word without a sign is reported, never invented;
- the signing itself is never generated: it is assembled from recorded signs of human signers.

**In building the data.** Models proposed, people and fixed texts decided:

- every hadith label is the text of the collection (hadith-api); speech recognition, vision and translation only point to
  which hadith a video signs, and a proposal is accepted only when the collection's text confirms it;
- the 249 Arabic synonym lemmas and the reviewed synonym lists were proposed by a model from the lexicon's own signs and
  reviewed one by one (100 of 241 proposals rejected in the last batch);
- evaluation uses an LLM judge only to decide whether two differently named signs mean the same; the judgments are cached
  and published with the results.

**In developing the software.** The code, data pipelines, tests and documentation were written with the help of an AI
coding assistant (Claude Code, Anthropic), directed and reviewed by the team. Every change is in the Git history, tested
(184 automated tests) and checked against the data before release.

**Limits.** Automatic checks are not a substitute for Deaf signers and scholars: review by native signers and religious
scholars is part of the plan before any curricular use.

## 11. Academic references

The technical report ([docs/paper/main.pdf](paper/main.pdf)) cites every dataset and model above with its paper;
the BibTeX entries are in [docs/paper/refs.bib](paper/refs.bib).

---

<div dir="rtl">

## التوثيق بالعربية

يوثّق هذا الملف جميع المصادر التي تعتمد عليها «إشارتي»، وحقوق كلٍّ منها، وكيفية استخدام الذكاء الاصطناعي في بناء النظام.

### المصادر النصية (القرآن والحديث)

- **نص القرآن الكريم**: النص العثماني المعتمد في مسابقة IslamicEval 2025 (6,236 آية)، ونص مشروع «تنزيل» عبر واجهة
  AlQuran Cloud لعرض السور والآيات في التطبيق (رخصة المشاع الإبداعي CC BY 3.0 مع ذكر المصدر).
- **ترجمات معاني القرآن**: موسوعة «QuranEnc» (الإنجليزية: صحيح إنترناشونال، التركية: روّاد، الأردية: جوناكري).
- **الحديث الشريف**: «موسوعة الأحاديث النبوية المترجمة» HadeethEnc بأحاديثها الصحيحة ودرجاتها وشروحها، ومكتبة
  hadith-api (fawazahmed0) لنص كل حديث كما ورد في كتابه (البخاري، مسلم، السنن، الموطأ، الأربعون النووية، الأربعون
  القدسية)، وهو الوسم المعتمد لكل عينة في مدوّنة الحديث بالإشارة.

### المعاجم الإشارية

| اللغة | لغة الإشارة | عدد الإشارات | التغطية | أبرز المصادر |
|---|---|---|---|---|
| العربية | لغة الإشارة العربية | 6,663 | 72.9٪ | KArSL، القاموس الإشاري العربي الموحّد، القاموس الكويتي، مركز تواصل، منهج الخرج، قنوات تعليم الصم (20 قناة) |
| الإنجليزية | لغة الإشارة الأمريكية | 3,677 | 83.1٪ | ASL Citizen، WLASL، MS-ASL، YouTube-ASL |
| التركية | لغة الإشارة التركية | 4,827 | 75.7٪ | قاموس لغة الإشارة التركية، TiDiSLaM، AUTSL |
| الأردية | لغة الإشارة الهندية والباكستانية | 10,233 | 72.1٪ | CISLR، WSLP، ISLRTC، قاموس PSL |

### مدوّنة القرآن والحديث بالإشارة

- **القرآن الكريم**: 7,980 مقطعاً (44.9 ساعة) من خمسة مصادر: مجمع الملك فهد لطباعة المصحف الشريف، والمختصر في تفسير
  القرآن الكريم، ومنهج الخرج، وتبيان، ورئاسة الشؤون الدينية التركية (ديانت).
- **الحديث الشريف**: 375 عينة لـ301 حديث (15.5 ساعة) من 824 مقطعاً لجمعيات الصم والقنوات الإسلامية (القائمة الكاملة في
  القسم 3 أعلاه).

### الحقوق

- **الشيفرة البرمجية**: رخصة MIT.
- **البيانات**: يحتفظ كل مصدر بشروطه كما في الجداول أعلاه وفي بطاقة كل مجموعة بيانات. بعض المصادر للاستخدام غير التجاري
  (ASL Citizen، WSLP، AUTSL، إشارة)، وبعضها لم يُعلن رخصة، ويُطلب الإذن من ناشريها، ولذلك فمجموعات البيانات خاصة.
- **المقاطع المرئية**: لا يُخزَّن ولا يُعاد نشر أي مقطع مرئي؛ تحتفظ البيانات بالنقاط المفصلية (أرقام تصف مواضع الجسم
  واليدين والوجه) مع رابط المقطع الأصلي في موضعه، فيبقى الناشر هو مكان المشاهدة.
- **النصوص الشرعية**: تُقتبس الآيات والأحاديث بنصّها مع مرجعها ورابط مصدرها، ولا تُعاد صياغة آية على أنها نصّها.
- **المراجعة**: لم تُعتمد أي إشارة بعدُ من مراجعين صمّ، وكل إشارة موسومة «بانتظار المراجعة» حتى تُعتمد.
- **الإزالة**: لأي ناشر يرغب في إزالة مادته أن يفتح بلاغاً (Issue) في هذا المستودع، فتُزال من البيانات والمعجم.

### استخدام الذكاء الاصطناعي

- **داخل المنصة**: تكتب النماذج اللغوية الإجابة والإشارات وفق قيود صارمة: لا تُبنى الإجابة إلا من آيات وأحاديث مسترجعة،
  ولا تُقبل جملة لا يدعمها مصدرها، وتُحال أسئلة الفتوى إلى أهل العلم قبل استدعاء أي نموذج، وتقتصر الإشارات على ما هو
  مسجَّل في المعجم، ولا تُولَّد حركة الإشارة نفسها بل تُركَّب من تسجيلات لمترجمين بشريين.
- **في بناء البيانات**: اقترحت النماذج والقرار للنصوص الثابتة وللفريق؛ فوسم كل حديث هو نصّه في كتابه، ولا يُقبل اقتراح
  التفريغ الصوتي أو قراءة النص المعروض إلا إذا أكّده نص الحديث، والمرادفات العربية (249 جذراً معجمياً) اقترحها نموذج
  من إشارات المعجم نفسه وراجعها الفريق واحدةً واحدة.
- **في تطوير البرمجيات**: كُتبت الشيفرة ومسارات البيانات والاختبارات والتوثيق بمساعدة مساعد برمجي ذكي (Claude Code من
  Anthropic) بتوجيه الفريق ومراجعته، وكل تغيير موثّق في سجل Git ومختبَر (184 اختباراً آلياً).
- **الحدود**: لا تغني الفحوص الآلية عن الصمّ وأهل العلم؛ فمراجعة المترجمين الصمّ والعلماء جزء من الخطة قبل أي استخدام
  تعليمي.

</div>
