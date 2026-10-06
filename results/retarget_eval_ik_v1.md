# Avatar retargeting vs keypoints (ik_v1, avatar.vrm)

Median over 21 hand tracks in 15 signs: height error 0.2 (nose heights), side error 0.05 (shoulder widths), palm angle 0.4°, share of frames with the palm flipped 2%.

| sign | where | hand | height signer → avatar | contact signer → avatar | palm ° | flipped |
|---|---|---|---|---|---|---|
| FAST (noor_for_sign) | mouth | right | 0.83 → 0.84 | 0.12 → 0.09 | 0.4 | 0% |
| RAMADAN (noor_for_sign) | mouth | right | 0.8 → 0.78 | 0.09 → 0.07 | 0.2 | 0% |
| EAT1 (asl_citizen) | mouth | left | 0.6 → 0.44 | 0.24 → 0.38 | 0.5 | 0% |
| MOTHER (asl_citizen) | chin | left | 1.05 → 0.9 | 0.2 → 0.32 | 0.3 | 0% |
| FATHER (asl_citizen) | forehead | left | 1.98 → 1.74 | 0.49 → 0.55 | 0.3 | 0% |
| THANK YOU (wlasl) | chin | right | -0.66 → -0.77 | 0.33 → 0.41 | 0.4 | 0% |
| GOD (asl_citizen) | above head | right | 0.3 → -0.27 | 0.24 → 0.23 | 0.6 | 15% |
| PRAYER (asl_citizen) | chest | left | 0.44 → 0.33 | 0.22 → 0.36 | 0.4 | 0% |
| PRAYER (asl_citizen) | chest | right | 0.47 → 0.23 | 0.22 → 0.39 | 0.4 | 0% |
| HAPPY (asl_citizen) | chest | left | -0.24 → -0.31 | 0.59 → 0.62 | 0.5 | 0% |
| SALAH (gdm_islamic_signs) | chest | left | 1.19 → 1.27 | 0.21 → 0.29 | 0.8 | 0% |
| SALAH (gdm_islamic_signs) | chest | right | 1.14 → 1.23 | 0.33 → 0.34 | 1.1 | 3% |
| QURAN (gdm_islamic_signs) | chest | left | 0.16 → -0.05 | 0.37 → 0.5 | 0.4 | 0% |
| QURAN (gdm_islamic_signs) | chest | right | 0.27 → 0.06 | 0.25 → 0.35 | 0.5 | 0% |
| صوم (children_dictionary) | mouth | right | 0.72 → 0.61 | 0.18 → 0.26 | 0.3 | 17% |
| رمضان (tawasol) | face | left | -1.16 → -0.79 | 1.07 → 1.14 | 1.1 | 0% |
| رمضان (tawasol) | face | right | 0.74 → 0.5 | 0.04 → 0.27 | 0.6 | 0% |
| الصلاة (karsl) | chest | left | 1.4 → 1.04 | 0.4 → 0.45 | 0.3 | 0% |
| الصلاة (karsl) | chest | right | 1.45 → 0.98 | 0.38 → 0.34 | 0.3 | 0% |
| الله (quran_curriculum) | above head | left | -1.05 → -0.86 | 0.74 → 1.12 | 0.6 | 0% |
| الله (quran_curriculum) | above head | right | 1.21 → 0.8 | 0.11 → 0.23 | 1.9 | 0% |
