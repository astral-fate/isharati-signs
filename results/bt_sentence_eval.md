# Sentence-level back-translation, Arabic -> ArSL (10 sentences)

3 run(s); mean ± sd over runs. Recogniser: dtw-1nn-heldout-signer. Back-translation LLM: nvidia/nemotron-3-super-120b-a12b.

| path | BLEU | chrF | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|
| gloss -> text (glossing loss only) | 50.6 ± 0.0 | 57.2 ± 0.0 | 44.6 ± 0.0 | 31.7 ± 0.0 | 44.6 ± 0.0 |
| pose -> text (back-translation) | 2.4 ± 0.3 | 10.7 ± 0.4 | 4.1 ± 1.3 | 0.0 ± 0.0 | 4.1 ± 1.3 |

Sign-level back-translation on the same sentences: gloss accuracy 8%, gloss WER 0.92. Only 12 of the 27 signed glosses (run 1) have a held-out-signer template, so the recogniser cannot name the others: the pose -> text row is bounded by the recogniser, not only by the signing.

Run 1, per sentence:

| source | sentence | glosses | gloss -> text | recognised | pose -> text | BLEU g/p | chrF g/p |
|---|---|---|---|---|---|---|---|
| Bukhari 1 | إنما الأعمال بالنيات | النية | النية | غدر | غدر | 0 / 0 | 14 / 0 |
| Muslim 55 | الدين النصيحة | نصيحة دين | نصيحة في الدين | سكين التهاب | سكين التهاب | 28 / 0 | 48 / 19 |
| Bukhari 2442 | المسلم أخو المسلم | مسلم أخ مسلم | مسلم أخ مسلم | وجه إسهال وجه | وجه - إسهال - وجه | 0 / 0 | 42 / 7 |
| Muslim 223 | الطهور شطر الإيمان | طهارة نصف إيمان | نصف الإيمان طهارة | طهارة غدر واجبات | طهارة - غدر - واجبات | 28 / 0 | 38 / 10 |
| Bukhari 6094 | الصدق يهدي إلى البر | صدق بر | صدق - بر | أناني إمام | أنا أناني - إمام | 0 / 0 | 14 / 6 |
| al-Fatihah 1:2 | الحمد لله رب العالمين | الحمد هو رب عالم | الحمد لله رب العالمين | التهاب فروض الشهادتين أركان الإسلام | التهاب فروض الشهادتين أركان الإسلام | 100 / 0 | 100 / 15 |
| al-Baqarah 2:43 | أقيموا الصلاة وآتوا الزكاة | الصلاة الزكاة | الصلاة والزكاة | مرتبك الزكاة | الزكاة مرتبك | 18 / 18 | 47 / 20 |
| statement | صوم رمضان فرض على كل مسلم | صوم رمضان فرض مسلم | صوم رمضان فرض على المسلم | ينام إمام فروض وجه | الإمام يواجه فروضه وهو نائم | 55 / 0 | 79 / 10 |
| statement | الحج ركن من أركان الإسلام | الحج رُكن إسلام | الحج ركن من أركان الإسلام | حول فلاح 2 | حول الفلاح 2 | 100 / 0 | 100 / 8 |
| statement | الله رحيم بعباده | الله رحيم عابد | الله رحيم عابد | إمام نبض القلب يدخل | الإمام يدخل نبض القلب | 55 / 0 | 53 / 11 |
