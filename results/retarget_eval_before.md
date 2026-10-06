# Avatar retargeting vs keypoints (before, avatar.vrm)

Median over 21 hand tracks in 15 signs: height error 0.27 (nose heights), side error 0.1 (shoulder widths), palm angle 0.4°, share of frames with the palm flipped 2%.

| sign | where | hand | height signer → avatar | contact signer → avatar | palm ° | flipped |
|---|---|---|---|---|---|---|
| FAST (noor_for_sign) | mouth | right | 0.83 → 0.52 | 0.12 → 0.12 | 0.5 | 0% |
| RAMADAN (noor_for_sign) | mouth | right | 0.8 → 0.51 | 0.09 → 0.24 | 0.2 | 0% |
| EAT1 (asl_citizen) | mouth | left | 0.6 → 0.41 | 0.24 → 0.48 | 0.2 | 0% |
| MOTHER (asl_citizen) | chin | left | 1.05 → 0.69 | 0.2 → 0.47 | 0.2 | 0% |
| FATHER (asl_citizen) | forehead | left | 1.98 → 1.06 | 0.49 → 0.55 | 0.3 | 0% |
| THANK YOU (wlasl) | chin | right | -0.66 → -0.33 | 0.33 → 0.4 | 0.4 | 0% |
| GOD (asl_citizen) | above head | right | 0.3 → -0.13 | 0.24 → 0.44 | 0.6 | 15% |
| PRAYER (asl_citizen) | chest | left | 0.44 → 0.35 | 0.22 → 0.37 | 0.3 | 0% |
| PRAYER (asl_citizen) | chest | right | 0.47 → 0.4 | 0.22 → 0.39 | 0.5 | 0% |
| HAPPY (asl_citizen) | chest | left | -0.24 → -0.24 | 0.59 → 0.61 | 0.4 | 0% |
| SALAH (gdm_islamic_signs) | chest | left | 1.19 → 0.97 | 0.21 → 0.28 | 0.6 | 0% |
| SALAH (gdm_islamic_signs) | chest | right | 1.14 → 0.96 | 0.33 → 0.34 | 0.4 | 3% |
| QURAN (gdm_islamic_signs) | chest | left | 0.16 → 0.02 | 0.37 → 0.48 | 0.4 | 0% |
| QURAN (gdm_islamic_signs) | chest | right | 0.27 → 0.15 | 0.25 → 0.4 | 0.5 | 0% |
| صوم (children_dictionary) | mouth | right | 0.72 → 0.47 | 0.18 → 0.36 | 0.2 | 17% |
| رمضان (tawasol) | face | left | -1.16 → -0.54 | 1.07 → 1.05 | 0.7 | 0% |
| رمضان (tawasol) | face | right | 0.74 → 0.24 | 0.04 → 0.54 | 0.3 | 0% |
| الصلاة (karsl) | chest | left | 1.4 → 0.75 | 0.4 → 0.49 | 0.1 | 0% |
| الصلاة (karsl) | chest | right | 1.45 → 0.68 | 0.38 → 0.44 | 0.2 | 0% |
| الله (quran_curriculum) | above head | left | -1.05 → -0.78 | 0.74 → 1.13 | 0.6 | 0% |
| الله (quran_curriculum) | above head | right | 1.21 → 0.53 | 0.11 → 0.33 | 0.6 | 0% |
