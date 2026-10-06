# Avatar retargeting vs keypoints (ik_v6_fingers, avatar.vrm)

Median over 21 hand tracks in 15 signs: height error 0.09 (nose heights), side error 0.04 (shoulder widths), palm angle 0.4°, share of frames with the palm flipped 1%.

| sign | where | hand | height signer → avatar | contact signer → avatar | palm ° | flipped |
|---|---|---|---|---|---|---|
| FAST (noor_for_sign) | mouth | right | 0.83 → 0.77 | 0.12 → 0.08 | 0.4 | 0% |
| RAMADAN (noor_for_sign) | mouth | right | 0.8 → 0.88 | 0.09 → 0.08 | 0.2 | 0% |
| EAT1 (asl_citizen) | mouth | left | 0.6 → 0.44 | 0.24 → 0.37 | 0.2 | 0% |
| MOTHER (asl_citizen) | chin | left | 1.05 → 1.01 | 0.2 → 0.31 | 0.2 | 0% |
| FATHER (asl_citizen) | forehead | left | 1.98 → 1.89 | 0.49 → 0.35 | 0.3 | 3% |
| THANK YOU (wlasl) | chin | right | -0.66 → -0.8 | 0.33 → 0.35 | 0.5 | 0% |
| GOD (asl_citizen) | above head | right | 0.3 → 0.17 | 0.24 → 0.23 | 0.4 | 2% |
| PRAYER (asl_citizen) | chest | left | 0.44 → 0.37 | 0.22 → 0.28 | 0.3 | 0% |
| PRAYER (asl_citizen) | chest | right | 0.47 → 0.34 | 0.22 → 0.31 | 0.5 | 0% |
| HAPPY (asl_citizen) | chest | left | -0.24 → -0.28 | 0.59 → 0.57 | 0.5 | 0% |
| SALAH (gdm_islamic_signs) | chest | left | 1.19 → 1.27 | 0.21 → 0.24 | 0.8 | 0% |
| SALAH (gdm_islamic_signs) | chest | right | 1.14 → 1.23 | 0.33 → 0.3 | 0.6 | 3% |
| QURAN (gdm_islamic_signs) | chest | left | 0.16 → 0.05 | 0.37 → 0.4 | 0.5 | 0% |
| QURAN (gdm_islamic_signs) | chest | right | 0.27 → 0.19 | 0.25 → 0.29 | 0.4 | 0% |
| صوم (children_dictionary) | mouth | right | 0.72 → 0.67 | 0.18 → 0.11 | 0.2 | 5% |
| رمضان (tawasol) | face | left | -1.16 → -0.95 | 1.07 → 1.25 | 0.8 | 0% |
| رمضان (tawasol) | face | right | 0.74 → 0.51 | 0.04 → 0.12 | 0.6 | 0% |
| الصلاة (karsl) | chest | left | 1.4 → 1.07 | 0.4 → 0.28 | 0.1 | 0% |
| الصلاة (karsl) | chest | right | 1.45 → 1.37 | 0.38 → 0.32 | 0.1 | 0% |
| الله (quran_curriculum) | above head | left | -1.05 → -0.77 | 0.74 → 1.07 | 0.4 | 0% |
| الله (quran_curriculum) | above head | right | 1.21 → 1.06 | 0.11 → 0.04 | 1.1 | 0% |
