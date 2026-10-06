"""Isharati Academy: a faith-based sign curriculum over the lexicons the app already has.

Each item is one concept with its word in every sign language (ArSL by Arabic, TİD by Turkish, ASL and ISL by English
glosses). An item is taught in a sign language only where that lexicon has a real recording of it (the lexicon's own
matcher decides); otherwise the dialect switch leaves that language out. Nothing is fingerspelled or composed here.

  curriculum(ui)        paths -> lessons -> items, each item with the sign languages that have it
  sign_pose(lang, id)   the item's sign with a rest pose before and after, and the signer's face (as /pose/{id}.json)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from isharati.types import FPS, GlossItem, GlossResult

SIGN_LANGS = {"ar": "ArSL", "en": "ASL", "tr": "TİD", "ur": "ISL"}

# concept id -> words per lexicon language (ur: the ISL lexicon is glossed in English) and its meaning per UI language
ITEMS = {
    "salam": {"ar": "السلام عليكم", "en": "PEACE", "tr": "selam", "ur": "PEACE",
              "mean": {"en": "Peace be upon you (the Islamic greeting)", "ar": "تحية الإسلام", "tr": "Selamün aleyküm", "ur": "السلام علیکم"}},
    "thanks": {"ar": "شكرا", "en": "THANK YOU", "tr": "teşekkür", "ur": "THANK YOU",
               "mean": {"en": "Thank you", "ar": "شكرًا", "tr": "Teşekkür ederim", "ur": "شکریہ"}},
    "inshallah": {"ar": "إن شاء الله", "en": "INSHALLAH", "tr": "inşallah", "ur": "INSHALLAH",
                  "mean": {"en": "If Allah wills", "ar": "إن شاء الله", "tr": "İnşallah", "ur": "ان شاء اللہ"}},
    "mashallah": {"ar": "ما شاء الله", "en": "MASHALLAH", "tr": "maşallah", "ur": "MASHALLAH",
                  "mean": {"en": "What Allah has willed", "ar": "ما شاء الله", "tr": "Maşallah", "ur": "ما شاء اللہ"}},
    "allah": {"ar": "الله", "en": "ALLAH", "tr": "allah", "ur": "ALLAH",
              "mean": {"en": "Allah, God", "ar": "الله", "tr": "Allah", "ur": "اللہ"}},
    "subhanallah": {"ar": "سبحان الله", "en": "SUBHANALLAH", "tr": "sübhanallah", "ur": "SUBHANALLAH",
                    "mean": {"en": "Glory be to Allah", "ar": "سبحان الله", "tr": "Sübhanallah", "ur": "سبحان اللہ"}},
    "alhamdulillah": {"ar": "الحمد لله", "en": "ALHAMDULILLAH", "tr": "elhamdülillah", "ur": "ALHAMDULILLAH",
                      "mean": {"en": "All praise is for Allah", "ar": "الحمد لله", "tr": "Elhamdülillah", "ur": "الحمد للہ"}},
    "allahuakbar": {"ar": "الله أكبر", "en": "ALLAHU AKBAR", "tr": "allahu ekber", "ur": "ALLAHU AKBAR",
                    "mean": {"en": "Allah is the Greatest", "ar": "الله أكبر", "tr": "Allahu ekber", "ur": "اللہ اکبر"}},
    "tawhid": {"ar": "لا إله إلا الله", "en": "THERE IS NO GOD BUT ALLAH", "tr": "kelime i şehadet", "ur": "SHAHADA",
               "mean": {"en": "There is no god but Allah", "ar": "لا إله إلا الله", "tr": "Kelime-i şehadet", "ur": "لا الہ الا اللہ"}},
    "istighfar": {"ar": "أستغفر الله", "en": "FORGIVE", "tr": "tövbe", "ur": "FORGIVE",
                  "mean": {"en": "Seeking Allah's forgiveness", "ar": "أستغفر الله", "tr": "Tövbe, istiğfar", "ur": "استغفار"}},
    "dua": {"ar": "دعاء", "en": "PRAY", "tr": "dua", "ur": "PRAY",
            "mean": {"en": "Supplication (du'a)", "ar": "الدعاء", "tr": "Dua", "ur": "دعا"}},
    "wudu": {"ar": "وضوء", "en": "WUDU", "tr": "abdest", "ur": "WUDU",
             "mean": {"en": "Ablution before prayer", "ar": "الوضوء", "tr": "Abdest", "ur": "وضو"}},
    "salah": {"ar": "الصلاة", "en": "PRAYER", "tr": "namaz", "ur": "PRAYER",
              "mean": {"en": "The prayer (salah)", "ar": "الصلاة", "tr": "Namaz", "ur": "نماز"}},
    "mosque": {"ar": "مسجد", "en": "MOSQUE", "tr": "cami", "ur": "MOSQUE",
               "mean": {"en": "Mosque", "ar": "المسجد", "tr": "Cami", "ur": "مسجد"}},
    "adhan": {"ar": "أذان", "en": "ADHAN", "tr": "ezan", "ur": "ADHAN",
              "mean": {"en": "The call to prayer", "ar": "الأذان", "tr": "Ezan", "ur": "اذان"}},
    "qibla": {"ar": "قبلة", "en": "QIBLA", "tr": "kıble", "ur": "QIBLA",
              "mean": {"en": "The direction of prayer", "ar": "القبلة", "tr": "Kıble", "ur": "قبلہ"}},
    "sujud": {"ar": "سجود", "en": "PROSTRATE", "tr": "secde", "ur": "PROSTRATE",
              "mean": {"en": "Prostration", "ar": "السجود", "tr": "Secde", "ur": "سجدہ"}},
    "imam": {"ar": "إمام", "en": "IMAM", "tr": "imam", "ur": "IMAM",
             "mean": {"en": "The prayer leader", "ar": "الإمام", "tr": "İmam", "ur": "امام"}},
    "friday": {"ar": "الجمعة", "en": "FRIDAY", "tr": "cuma", "ur": "FRIDAY",
               "mean": {"en": "Friday, the day of congregation", "ar": "يوم الجمعة", "tr": "Cuma", "ur": "جمعہ"}},
    "khutbah": {"ar": "خطبة", "en": "SERMON", "tr": "hutbe", "ur": "SERMON",
                "mean": {"en": "The Friday sermon", "ar": "الخطبة", "tr": "Hutbe", "ur": "خطبہ"}},
    "quran": {"ar": "القرآن", "en": "QURAN", "tr": "kur'an", "ur": "QURAN",
              "mean": {"en": "The Qur'an", "ar": "القرآن الكريم", "tr": "Kur'an", "ur": "قرآن"}},
    "prophet": {"ar": "نبي", "en": "PROPHET", "tr": "peygamber", "ur": "PROPHET",
                "mean": {"en": "Prophet", "ar": "النبي", "tr": "Peygamber", "ur": "نبی"}},
    "muhammad": {"ar": "محمد", "en": "MUHAMMAD", "tr": "muhammed", "ur": "MUHAMMAD",
                 "mean": {"en": "Prophet Muhammad ﷺ", "ar": "محمد ﷺ", "tr": "Hz. Muhammed", "ur": "محمد ﷺ"}},
    "musa": {"ar": "موسى", "en": "MOSES", "tr": "musa", "ur": "MOSES",
             "mean": {"en": "Prophet Musa (Moses)", "ar": "موسى عليه السلام", "tr": "Hz. Musa", "ur": "موسیٰ"}},
    "isa": {"ar": "عيسى", "en": "JESUS", "tr": "isa", "ur": "JESUS",
            "mean": {"en": "Prophet Isa (Jesus)", "ar": "عيسى عليه السلام", "tr": "Hz. İsa", "ur": "عیسیٰ"}},
    "ibrahim": {"ar": "إبراهيم", "en": "ABRAHAM", "tr": "ibrahim", "ur": "ABRAHAM",
                "mean": {"en": "Prophet Ibrahim (Abraham)", "ar": "إبراهيم عليه السلام", "tr": "Hz. İbrahim", "ur": "ابراہیم"}},
    "nuh": {"ar": "نوح", "en": "NOAH", "tr": "nuh", "ur": "NOAH",
            "mean": {"en": "Prophet Nuh (Noah)", "ar": "نوح عليه السلام", "tr": "Hz. Nuh", "ur": "نوح"}},
    "sabr": {"ar": "صبر", "en": "PATIENCE", "tr": "sabır", "ur": "PATIENCE",
             "mean": {"en": "Patience (sabr)", "ar": "الصبر", "tr": "Sabır", "ur": "صبر"}},
    "zakat": {"ar": "الزكاة", "en": "ZAKAT", "tr": "zekât", "ur": "ZAKAT",
              "mean": {"en": "Obligatory charity (zakat)", "ar": "الزكاة", "tr": "Zekât", "ur": "زکوٰۃ"}},
    "sadaqa": {"ar": "صدقة", "en": "CHARITY", "tr": "sadaka", "ur": "CHARITY",
               "mean": {"en": "Voluntary charity (sadaqa)", "ar": "الصدقة", "tr": "Sadaka", "ur": "صدقہ"}},
    "fasting": {"ar": "صوم", "en": "FAST", "tr": "oruç", "ur": "FAST",
                "mean": {"en": "Fasting", "ar": "الصوم", "tr": "Oruç", "ur": "روزہ"}},
    "ramadan": {"ar": "رمضان", "en": "RAMADAN", "tr": "ramazan", "ur": "RAMADAN",
                "mean": {"en": "Ramadan", "ar": "شهر رمضان", "tr": "Ramazan", "ur": "رمضان"}},
    "hajj": {"ar": "حج", "en": "HAJJ", "tr": "hac", "ur": "HAJJ",
             "mean": {"en": "The pilgrimage (hajj)", "ar": "الحج", "tr": "Hac", "ur": "حج"}},
    "jannah": {"ar": "الجنة", "en": "HEAVEN", "tr": "cennet", "ur": "HEAVEN",
               "mean": {"en": "Paradise (jannah)", "ar": "الجنة", "tr": "Cennet", "ur": "جنت"}},
    "iman": {"ar": "إيمان", "en": "FAITH", "tr": "iman", "ur": "FAITH",
             "mean": {"en": "Faith (iman)", "ar": "الإيمان", "tr": "İman", "ur": "ایمان"}},
}

# The Arabic alphabet in sign: Diyanet's «İşaret Dili ile Kur'an» episode 30 (Arabic Sign Language finger alphabet,
# taught to Turkish learners of the Mushaf), cut per letter by scripts/academy/diyanet_letters.py. Taught under ArSL
# and TİD, with the clip itself as the sign (not a lexicon entry).
LETTERS_NAME = "academy/diyanet_letters.npz"  # in the isharati-app-data dataset (isharati.app_data), with its attribution
LETTER_LANGS = ("ar", "tr")
_LETTERS = [  # letter, Arabic name, English name, Turkish name
    ("ا", "ألف", "Alif", "Elif"), ("ب", "باء", "Bā", "Be"), ("ت", "تاء", "Tā", "Te"), ("ث", "ثاء", "Thā", "Se"),
    ("ج", "جيم", "Jīm", "Cim"), ("ح", "حاء", "Ḥā", "Ha"), ("خ", "خاء", "Khā", "Hı"), ("د", "دال", "Dāl", "Dal"),
    ("ذ", "ذال", "Dhāl", "Zel"), ("ر", "راء", "Rā", "Ra"), ("ز", "زاي", "Zāy", "Ze"), ("س", "سين", "Sīn", "Sin"),
    ("ش", "شين", "Shīn", "Şın"), ("ص", "صاد", "Ṣād", "Sad"), ("ض", "ضاد", "Ḍād", "Dad"), ("ط", "طاء", "Ṭā", "Tı"),
    ("ظ", "ظاء", "Ẓā", "Zı"), ("ع", "عين", "ʿAyn", "Ayın"), ("غ", "غين", "Ghayn", "Gayın"), ("ف", "فاء", "Fā", "Fe"),
    ("ق", "قاف", "Qāf", "Kaf"), ("ك", "كاف", "Kāf", "Kef"), ("ل", "لام", "Lām", "Lam"), ("م", "ميم", "Mīm", "Mim"),
    ("ن", "نون", "Nūn", "Nun"), ("ه", "هاء", "Hā", "He"), ("و", "واو", "Wāw", "Vav"), ("ي", "ياء", "Yā", "Ye"),
]
for _k, (_l, _ar, _en, _tr) in enumerate(_LETTERS, 1):
    ITEMS[f"ltr{_k:02d}"] = {"ar": _l, "en": _l, "tr": _l, "ur": _l, "clip": f"l{_k:02d}",
                             "names": {"ar": _ar, "en": _en, "tr": _tr, "ur": _ar},
                             "mean": {"en": f"The letter {_en}", "ar": f"حرف {_ar}", "tr": f"{_tr} harfi", "ur": f"حرف {_ar}"}}

# matches the lexicon makes but must not teach: a different word with the same letters once hamza is folded
WRONG_SENSE = {("imam", "ar"),   # إمام (imam) matched أمام (in front of)
               ("salam", "ur")}  # PEACE there is a PSL recording of peace/calm, not the greeting

PATHS = [
    {"id": "greetings", "icon": "👋", "title": {"en": "Greetings in Islam", "ar": "التحية في الإسلام", "tr": "İslam'da selamlaşma", "ur": "اسلامی سلام"},
     "lessons": [["salam", "thanks", "inshallah", "mashallah"]]},
    {"id": "athkar", "icon": "📿", "title": {"en": "Daily athkar", "ar": "الأذكار اليومية", "tr": "Günlük zikirler", "ur": "روزانہ اذکار"},
     "lessons": [["allah", "subhanallah", "alhamdulillah", "allahuakbar"], ["tawhid", "istighfar", "dua"]]},
    {"id": "salah", "icon": "🕌", "title": {"en": "Salah basics", "ar": "أساسيات الصلاة", "tr": "Namazın temelleri", "ur": "نماز کی بنیادی باتیں"},
     "lessons": [["wudu", "salah", "mosque", "adhan"], ["qibla", "sujud", "imam"], ["friday", "khutbah", "quran"]]},
    {"id": "prophets", "icon": "📖", "title": {"en": "Prophets & Qur'anic stories", "ar": "الأنبياء وقصص القرآن", "tr": "Peygamberler ve Kur'an kıssaları", "ur": "انبیاء اور قرآنی قصے"},
     "lessons": [["prophet", "muhammad", "musa", "isa"], ["ibrahim", "nuh"]]},
    {"id": "values", "icon": "🤲", "title": {"en": "Pillars & values", "ar": "الأركان والقيم", "tr": "Şartlar ve değerler", "ur": "ارکان اور اقدار"},
     "lessons": [["sabr", "zakat", "sadaqa"], ["fasting", "ramadan", "hajj"], ["jannah", "iman"]]},
    {"id": "letters", "icon": "🔤", "title": {"en": "The Arabic alphabet (reading the Mushaf)", "ar": "الحروف العربية: طريقك إلى المصحف",
                                            "tr": "Arap harfleri: Kur'an okumaya ilk adım", "ur": "عربی حروف: مصحف پڑھنے کی پہلی سیڑھی"},
     "lessons": [[f"ltr{k:02d}" for k in range(i, min(i + 4, 29))] for i in range(1, 29, 4)]},
]


@lru_cache(maxsize=4)
def _lexicon(lang):
    from isharati.pipeline import load_lexicon
    return load_lexicon(lang)


def _entry(lang, word):
    lex = _lexicon(lang)
    e = lex.lookup(word)
    if e is None and " " not in word.strip():
        e = lex.match_token(word)
    return e


@lru_cache(maxsize=1)
def availability():
    """item id -> {sign language: gloss of its recorded sign}, only where the lexicon has a real recording."""
    out = {}
    have_letters = _letters_file() is not None
    for iid, item in ITEMS.items():
        out[iid] = {}
        if "clip" in item:
            if have_letters:
                out[iid] = {lang: item[lang] for lang in LETTER_LANGS}
            continue
        for lang in SIGN_LANGS:
            try:
                e = _entry(lang, item[lang])
            except Exception:  # a lexicon that cannot load here (e.g. its data not published) has nothing to teach
                e = None
            if e is not None and not getattr(e, "is_letter", False) and (iid, lang) not in WRONG_SENSE:
                out[iid][lang] = e.gloss
    return out


def texts(iid):
    """The item written in each sign language's spoken language, for under the sign: Arabic (the word itself),
    English, Turkish and Urdu (the ISL lexicon is glossed in English, so its text is the Urdu wording)."""
    if "names" in ITEMS[iid]:                                  # a letter: the letter, then its name
        n = ITEMS[iid]["names"]
        return {lang: f"{ITEMS[iid]['ar']} · {n[lang]}" for lang in SIGN_LANGS}
    m = ITEMS[iid]["mean"]
    return {"ar": ITEMS[iid]["ar"], "en": m["en"].split(" (")[0], "tr": m["tr"], "ur": m["ur"]}


def sources(iid, avail):
    """Per sign language, the sign language of the actual recording: the ur lexicon mixes ISL and PSL clips."""
    if "clip" in ITEMS[iid]:
        return {lang: "Diyanet" for lang in avail[iid]}
    out = {}
    for lang, gloss in avail[iid].items():
        e = _lexicon(lang).lookup(gloss) if lang == "ur" else None
        if e is not None and e.dataset.startswith("psl"):
            out[lang] = "PSL"
    return out


def curriculum(ui: str = "en"):
    avail = availability()
    pick = lambda d: d.get(ui) or d["en"]  # noqa: E731
    paths = []
    for p in PATHS:
        lessons = []
        for k, ids in enumerate(p["lessons"]):
            items = [{"id": i, "meaning": pick(ITEMS[i]["mean"]), "words": {l: ITEMS[i][l] for l in SIGN_LANGS},
                      "texts": texts(i), "langs": sorted(avail[i]), "src": sources(i, avail)} for i in ids if avail[i]]
            if items:
                lessons.append({"id": f"{p['id']}-{k + 1}", "items": items})
        if lessons:
            paths.append({"id": p["id"], "icon": p["icon"], "title": pick(p["title"]), "lessons": lessons})
    return {"sign_languages": SIGN_LANGS, "paths": paths}


@lru_cache(maxsize=256)
def sign_pose(lang: str, item_id: str):
    """The item's recorded sign between rest poses, with the signer's face: the /pose/{id}.json payload."""
    if item_id not in ITEMS or lang not in SIGN_LANGS:
        return None
    gloss = availability()[item_id].get(lang)
    if gloss is None:
        return None
    from isharati.pose import face
    from isharati.pose.poser import MissingAwarePoser
    clip = ITEMS[item_id].get("clip")
    lex = _ClipLexicon(clip) if clip else _lexicon(lang)
    pose, segments = MissingAwarePoser(lex).pose(GlossResult([GlossItem(gloss)], "academy"))
    pose = np.nan_to_num(pose).astype(float).round(4)
    out = {"fps": FPS, "frames": pose.tolist(), "gloss": gloss}
    signer_face = face.track(segments, len(pose), lang, lex, loader=lex.face if clip else None)
    if signer_face is not None:
        out["blend"] = face.as_json(signer_face["blend"])
        out["blend_names"] = signer_face["names"]
        out["face"] = face.as_json(signer_face["points"][..., :2])
        out["face_edges"] = face.CONTOURS["edges"]
    return out


@lru_cache(maxsize=1)
def _letters_file():
    from isharati import app_data
    return app_data.path(LETTERS_NAME)


def _letters():
    z = np.load(_letters_file())
    return {k: z[k] for k in z.files}


class _ClipLexicon:
    """One Academy clip (a letter) behind the lexicon interface MissingAwarePoser and face.track use."""

    def __init__(self, clip):
        from isharati.types import SignEntry
        self.clip = clip
        self.entry = SignEntry(gloss=clip, sign_id=clip, dataset="diyanet_letters", keypoints_path="",
                               review_status="approved", is_religious=True, is_letter=False)

    def lookup(self, text):
        return self.entry

    def keypoints(self, entry):
        return _letters()[f"{self.clip}/pose"].astype(np.float32)

    def face(self, lang, sign_id):
        z = _letters()
        return (z[f"{self.clip}/blend"].astype(np.float32), z[f"{self.clip}/face"].astype(np.float32),
                [str(n) for n in z["blend_names"]])
