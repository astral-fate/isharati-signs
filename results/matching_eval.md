# What Isharati does with Arabic text, against Deaf signers' annotations

Only the signer's signs that come from a word of the sentence (same root) are compared; the signer's added
signs (an added «انا», question markers) are left out, since no text-to-sign system produces them.

## isharah (300 sentences; 39% of the signer's signs come from the text)

**Word -> sign** (459 word pairs whose sign is in the lexicon)

| matcher | right sign | no sign | wrong sign |
|---|---|---|---|
| exact | 44.7% | 54.9% | 0.4% |
| affixes | 85.4% | 10.7% | 3.9% |
| full | 90.8% | 6.5% | 2.6% |

**Sentence** (the signer's signs that come from the text)

| system | precision | recall | F1 |
|---|---|---|---|
| copy | 0.284 | 0.637 | 0.393 |
| rule | 0.352 | 0.671 | 0.462 |
| llm | 0.351 | 0.681 | 0.464 |

## religious (54 sentences; 48% of the signer's signs come from the text)

**Word -> sign** (118 word pairs whose sign is in the lexicon)

| matcher | right sign | no sign | wrong sign |
|---|---|---|---|
| exact | 55.9% | 44.1% | 0.0% |
| affixes | 90.7% | 9.3% | 0.0% |
| full | 98.3% | 1.7% | 0.0% |

**Sentence** (the signer's signs that come from the text)

| system | precision | recall | F1 |
|---|---|---|---|
| copy | 0.385 | 0.697 | 0.496 |
| rule | 0.481 | 0.772 | 0.593 |
| llm | 0.477 | 0.779 | 0.592 |

## arabsign (25 sentences; 70% of the signer's signs come from the text)

**Word -> sign** (39 word pairs whose sign is in the lexicon)

| matcher | right sign | no sign | wrong sign |
|---|---|---|---|
| exact | 74.4% | 25.6% | 0.0% |
| affixes | 92.3% | 7.7% | 0.0% |
| full | 97.4% | 2.6% | 0.0% |

**Sentence** (the signer's signs that come from the text)

| system | precision | recall | F1 |
|---|---|---|---|
| copy | 0.589 | 0.796 | 0.677 |
| rule | 0.656 | 0.778 | 0.712 |
| llm | 0.611 | 0.815 | 0.698 |

## Errors of the full matcher (first 25 each)

| set | word | signer's sign | matched | kind |
|---|---|---|---|---|
| isharah | المدرسي | مدرسه | مدرس | wrong sign |
| isharah | الشركه | شركه | الشرك | wrong sign |
| isharah | عمره | عمر | عمرة | wrong sign |
| isharah | عمره | عمر | عمرة | wrong sign |
| isharah | الدرس | مدرسه | درَّس | wrong sign |
| isharah | الدرس | مدرسه | درَّس | wrong sign |
| isharah | الرياضه | رياضه | الرياض | wrong sign |
| isharah | لبسه | لبس | بسه | wrong sign |
| isharah | تذكره | ذاكره | ذكر | wrong sign |
| isharah | تجار | تجاره | جار | wrong sign |
| isharah | الدرس | مدرسه | درَّس | wrong sign |
| isharah | الرياضه | رياضه | الرياض | wrong sign |
| isharah | اشعر | شعور | — | no sign |
| isharah | خمسين | خمسون | — | no sign |
| isharah | السعودي | السعوديه | — | no sign |
| isharah | مزدحم | ازدحام | — | no sign |
| isharah | زهور | زهره | — | no sign |
| isharah | يدرس | مدرسه | — | no sign |
| isharah | اكره | كره | — | no sign |
| isharah | مزدحم | ازدحام | — | no sign |
| isharah | سرق | سرقه | — | no sign |
| isharah | تدخن | دخان | — | no sign |
| isharah | تدخن | دخان | — | no sign |
| isharah | معا | مع | — | no sign |
| isharah | ترتدي | وردي | — | no sign |
| isharah | اصغر | صغير | — | no sign |
| isharah | سرق | سرقه | — | no sign |
| isharah | خطير | خطر | — | no sign |
| isharah | المساجين | سجن | — | no sign |
| isharah | المدرسيه | مدرسه | — | no sign |
| isharah | ادرس | مدرسه | — | no sign |
| isharah | صحيه | صحه | — | no sign |
| isharah | السعودي | السعوديه | — | no sign |
| isharah | مشمس | شمس | — | no sign |
| isharah | الرابعه | اربع | — | no sign |
| isharah | المترجم | ترجمه | — | no sign |
| isharah | اترجم | ترجمه | — | no sign |
