"""TİD word signs from two YouTube sources, as gap fillers for the TİD lexicon (all pending review).

  «Our hands are speaking» (an EU youth-exchange project): six topic videos, each a sequence of single signs with an
      English caption naming the sign (burned in bottom-left; top-right in Animals; title cards before each sign in
      Family members). The signers are participants, probably learners rather than native signers.
  «The Sign Polyglot»: single-word TİD videos, the word in the title; the instructor demonstrates the sign amid
      explanation, often twice. Where it is repeated, the cut is the most consistent performance (DTW between the
      repeats, then hand detection).

Each span was found from the caption windows and the tracked hand activity inside them, then checked on a contact sheet
(the sign alone: no fade, no crossfade, no next signer). Cutting follows reviewed_recordings.py: the video is tracked
with the Holistic task (scripts/face/track.py), the clip is window_frame() of fifty() of the span with hands lost for
more than 3 frames resting at the wrist (rest_undetected_hands), and the face is saved in the same frame.

tid_lexicon() loads extra/entries.jsonl last (add_words), so these only sign words the lexicon lacks. A gloss the
lexicon already signs gets a "variant_of" row instead (another recording, kept for review/recognition; add_words skips
it), decided here against the lexicon without this source.

  py scripts/lexicon/tid/extra_sources.py
Output: data/lexicon_tr/extra/{signs/<sign_id>.npy, entries.jsonl}, data/face/tr/<sign_id>.npz (videos and tracks on F:)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "face"), str(ROOT / "scripts" / "lexicon" / "arsl")]
from isharati.config import DATA  # noqa: E402
from isharati.pose import proportions  # noqa: E402
from reviewed_recordings import FPS, MP_PYTHON, YTDLP, rest_undetected_hands  # noqa: E402

WORK = Path(r"F:\ourhands")
LEX_ROOT = DATA / "lexicon_tr"
OUT = LEX_ROOT / "extra"
FACES = DATA / "face" / "tr"
OURHANDS, POLYGLOT = "tid_ourhands", "tid_signpolyglot"
LEARNER_NOTE = ("signed by a participant of the «Our hands are speaking» youth exchange, who may be a learner rather "
                "than a native signer; pending review by a native TİD signer")

VIDEOS = {  # youtube id -> (dataset, source name)
    "K2A-e2D6uHQ": (OURHANDS, "Our hands are speaking: Health"),
    "eu88_Ha0frg": (OURHANDS, "Our hands are speaking: Verbs"),
    "PZE2tEUZUKM": (OURHANDS, "Our hands are speaking: Sports"),
    "Uw4sFVpTLcE": (OURHANDS, "Our hands are speaking: Months"),
    "KZ_U3aR88R4": (OURHANDS, "Our hands are speaking: Family members"),
    "31jCMtMOjZs": (OURHANDS, "Our hands are speaking: Animals"),
    "xJxy4t9GDec": (POLYGLOT, "The Sign Polyglot: How to say SIGN LANGUAGE in Turkish Sign Language"),
    "6QMph68LDAc": (POLYGLOT, "The Sign Polyglot: How to sign DEAF in Turkish Sign Language"),
    "Nn2ehWjwtNY": (POLYGLOT, "The Sign Polyglot: How to sign HOME in Turkish Sign Language"),
    "iQ9kEKs2n7I": (POLYGLOT, "The Sign Polyglot: How to sign Turkey in Turkish Sign Language"),
}

# (youtube id, english label as captioned, gloss, start_s, end_s[, note])
SIGNS = [
    # Health
    ("K2A-e2D6uHQ", "PAIN", "ağrı", 7.68, 10.12),
    ("K2A-e2D6uHQ", "BE PATIENT", "sabretmek", 11.36, 13.80),
    ("K2A-e2D6uHQ", "HOSPITAL", "hastane", 14.72, 17.00),
    ("K2A-e2D6uHQ", "PHAMARCY (pharmacy)", "eczane", 17.88, 20.24),
    ("K2A-e2D6uHQ", "OPERATION", "ameliyat", 21.20, 22.96, "starts after the crossfade from the previous signer"),
    ("K2A-e2D6uHQ", "PILL", "hap", 23.44, 25.40),
    ("K2A-e2D6uHQ", "FIRST AID", "ilk yardım", 26.20, 28.80),
    ("K2A-e2D6uHQ", "MICROBE", "mikrop", 29.80, 31.60, "starts after the crossfade"),
    ("K2A-e2D6uHQ", "PLASTER", "alçı", 33.36, 38.36, "ends before the crossfade to the next scene"),
    ("K2A-e2D6uHQ", "AMBULANCE", "ambulans", 39.24, 43.08),
    ("K2A-e2D6uHQ", "FLUE (flu)", "grip", 43.76, 47.00),
    ("K2A-e2D6uHQ", "CREAM", "krem", 48.64, 52.48),
    ("K2A-e2D6uHQ", "X-RAY", "röntgen", 53.92, 55.20),
    # Verbs
    ("eu88_Ha0frg", "TO BUY", "satın almak", 6.84, 8.44),
    ("eu88_Ha0frg", "TO CALL", "telefon etmek", 9.64, 11.04, "phone handshape at the ear: a phone call"),
    ("eu88_Ha0frg", "TO SCREAM", "bağırmak", 12.32, 13.88),
    ("eu88_Ha0frg", "TO LOOK", "bakmak", 15.32, 16.52),
    ("eu88_Ha0frg", "TO LIKE", "beğenmek", 18.32, 19.48),
    ("eu88_Ha0frg", "TO FINISH", "bitirmek", 21.00, 22.20),
    ("eu88_Ha0frg", "LOOK LIKE", "benzemek", 24.08, 25.12),
    ("eu88_Ha0frg", "TO WORK", "çalışmak", 26.64, 29.04),
    ("eu88_Ha0frg", "TO LISTEN", "dinlemek", 29.80, 33.12, "starts after the cut to the new signer"),
    ("eu88_Ha0frg", "TO BE LATE", "geç kalmak", 33.88, 36.08),
    ("eu88_Ha0frg", "TO COME", "gelmek", 40.32, 41.48),
    ("eu88_Ha0frg", "TO GO", "gitmek", 42.08, 44.60),
    ("eu88_Ha0frg", "TO WEAR", "giymek", 46.36, 49.68),
    ("eu88_Ha0frg", "TO SEE", "görmek", 51.44, 55.20),
    ("eu88_Ha0frg", "TO WANT", "istemek", 56.04, 59.20, "starts after the cut to the new signer"),
    ("eu88_Ha0frg", "TO WATCH", "izlemek", 60.52, 65.44),
    ("eu88_Ha0frg", "TO RUN", "koşmak", 66.84, 69.88),
    ("eu88_Ha0frg", "TO READ", "okumak", 71.40, 74.36),
    ("eu88_Ha0frg", "TO WRITE", "yazmak", 77.12, 79.72),
    ("eu88_Ha0frg", "TO LEARN", "öğrenmek", 82.44, 85.84),
    ("eu88_Ha0frg", "TO FORGET", "unutmak", 87.44, 89.56),
    # Sports (HIKING skipped: 6 s of continuous movement whose halves do not repeat, no clear single sign)
    ("PZE2tEUZUKM", "BASKETBALL", "basketbol", 6.80, 9.16),
    ("PZE2tEUZUKM", "BOXING", "boks", 9.52, 12.36),
    ("PZE2tEUZUKM", "FOOTBALL", "futbol", 13.60, 15.76),
    ("PZE2tEUZUKM", "TABLE TENIS (table tennis)", "masa tenisi", 16.56, 19.84),
    ("PZE2tEUZUKM", "VOLLEYBALL", "voleybol", 20.60, 22.56),
    ("PZE2tEUZUKM", "HANDBALL", "hentbol", 23.36, 26.96, "ends before the cut to the next signer"),
    ("PZE2tEUZUKM", "SWIMMING", "yüzme", 28.08, 30.44),
    ("PZE2tEUZUKM", "SKATING", "paten", 31.00, 33.20),
    ("PZE2tEUZUKM", "ICE SKATE", "buz pateni", 33.28, 36.04, "follows SKATING without a rest; similar movement"),
    ("PZE2tEUZUKM", "SURFING", "sörf", 44.08, 46.08),
    ("PZE2tEUZUKM", "SKIING", "kayak", 46.56, 49.40),
    ("PZE2tEUZUKM", "BILLIARD", "bilardo", 49.28, 51.56),
    ("PZE2tEUZUKM", "TENNIS", "tenis", 53.84, 56.08),
    ("PZE2tEUZUKM", "TRABZONSPOR", "trabzonspor", 56.80, 58.96),
    ("PZE2tEUZUKM", "FENERBAHCE", "fenerbahçe", 59.76, 61.80),
    ("PZE2tEUZUKM", "GALATASARAY", "galatasaray", 63.44, 66.52),
    ("PZE2tEUZUKM", "BESIKTAS", "beşiktaş", 67.84, 70.40, "ends before the fade to the end card"),
    # Months: one signer per run of months without rests; the span is the caption window
    ("Uw4sFVpTLcE", "JANUARY", "ocak", 7.36, 8.88),
    ("Uw4sFVpTLcE", "FEBRUARY", "şubat", 9.72, 11.80),
    ("Uw4sFVpTLcE", "MARCH", "mart", 11.96, 14.08),
    ("Uw4sFVpTLcE", "APRIL", "nisan", 14.32, 16.76),
    ("Uw4sFVpTLcE", "MAY", "mayıs", 17.48, 19.64),
    ("Uw4sFVpTLcE", "JUNE", "haziran", 19.88, 22.32),
    ("Uw4sFVpTLcE", "JULY", "temmuz", 22.68, 24.08),
    ("Uw4sFVpTLcE", "AUGUST", "ağustos", 24.60, 27.00),
    ("Uw4sFVpTLcE", "SEPTEMBER", "eylül", 28.64, 31.04),
    ("Uw4sFVpTLcE", "OCTOBER", "ekim", 31.36, 32.76),
    ("Uw4sFVpTLcE", "NOVEMBER", "kasım", 33.92, 35.76),
    ("Uw4sFVpTLcE", "DECEMBER", "aralık", 35.72, 37.24),
    # Family members: a title card, then the sign (fade-in and fade-out frames trimmed)
    ("KZ_U3aR88R4", "BROTHER", "erkek kardeş", 12.96, 14.72),
    ("KZ_U3aR88R4", "BIG BROTHER", "ağabey", 17.00, 18.56),
    ("KZ_U3aR88R4", "SISTER", "kız kardeş", 20.60, 22.60),
    ("KZ_U3aR88R4", "BIG SISTER", "abla", 24.88, 26.88),
    ("KZ_U3aR88R4", "MOTHER", "anne", 29.40, 31.60),
    ("KZ_U3aR88R4", "FATHER", "baba", 34.20, 36.84),
    ("KZ_U3aR88R4", "MOTHER'S SIDE UNCLE", "dayı", 39.56, 40.96),
    ("KZ_U3aR88R4", "FATHER'S SIDE UNCLE", "amca", 43.44, 47.00),
    ("KZ_U3aR88R4", "MOTHER'S SIDE AUNT", "teyze", 49.28, 50.56),
    ("KZ_U3aR88R4", "FATHER'S SIDE AUNT", "hala", 52.76, 54.48),
    ("KZ_U3aR88R4", "MOTHER'S SIDE GRANDMOTHER", "anneanne", 56.60, 57.88),
    ("KZ_U3aR88R4", "FATHER'S SIDE GRANDMOTHER", "babaanne", 60.12, 61.40),
    ("KZ_U3aR88R4", "BOTH SIDES GRANDFATHER", "dede", 63.60, 65.28),
    ("KZ_U3aR88R4", "HUMAN", "insan", 67.44, 68.48),
    ("KZ_U3aR88R4", "WIFE OR HUSBAND", "eş", 70.80, 73.04),
    # Animals: the signers do not rest between signs; the span is the caption window
    ("31jCMtMOjZs", "BEE", "arı", 0.08, 2.96),
    ("31jCMtMOjZs", "LION", "aslan", 3.12, 4.80),
    ("31jCMtMOjZs", "HORSE", "at", 4.96, 7.56),
    ("31jCMtMOjZs", "FISH", "balık", 7.76, 10.40),
    ("31jCMtMOjZs", "CHICK", "civciv", 10.56, 12.96),
    ("31jCMtMOjZs", "CAMEL", "deve", 13.12, 15.72),
    ("31jCMtMOjZs", "DONKEY", "eşek", 15.92, 17.92),
    ("31jCMtMOjZs", "MOUSE", "fare", 18.08, 20.24),
    ("31jCMtMOjZs", "DOLPHIN", "yunus", 20.60, 24.56),
    ("31jCMtMOjZs", "COCK", "horoz", 24.76, 26.92),
    ("31jCMtMOjZs", "ANT", "karınca", 28.08, 30.24),
    ("31jCMtMOjZs", "GIRAFE (giraffe)", "zürafa", 31.60, 33.36),
    ("31jCMtMOjZs", "PARROT", "papağan", 33.84, 35.48),
    ("31jCMtMOjZs", "RABBIT", "tavşan", 36.40, 39.04),
    ("31jCMtMOjZs", "SCORPION", "akrep", 40.28, 41.76),
    ("31jCMtMOjZs", "BEAR", "ayı", 42.92, 44.28),
    ("31jCMtMOjZs", "BEETLE", "böcek", 44.84, 47.08, "captioned BEETLE; glossed as the general word for insect"),
    ("31jCMtMOjZs", "PIG", "domuz", 47.28, 48.88),
    ("31jCMtMOjZs", "ELEPHANT", "fil", 49.24, 51.76),
    ("31jCMtMOjZs", "SHARK", "köpek balığı", 52.00, 55.56),
    ("31jCMtMOjZs", "COW", "inek", 56.04, 58.68),
    ("31jCMtMOjZs", "CAT", "kedi", 59.28, 61.68),
    ("31jCMtMOjZs", "BUTTERFLY", "kelebek", 62.60, 66.28, "one hand waved: may not be the standard sign"),
    ("31jCMtMOjZs", "BIRD", "kuş", 67.20, 70.32),
    ("31jCMtMOjZs", "DOG", "köpek", 70.60, 73.44),
    ("31jCMtMOjZs", "MONKEY", "maymun", 73.68, 76.04),
    ("31jCMtMOjZs", "FLY", "sinek", 76.24, 78.56),
    ("31jCMtMOjZs", "CHICKEN", "tavuk", 79.04, 81.40),
    ("31jCMtMOjZs", "CRAB", "yengeç", 81.52, 84.40),
    ("31jCMtMOjZs", "SNAKE", "yılan", 84.56, 86.12),
    ("31jCMtMOjZs", "TREE", "ağaç", 86.28, 88.32),
    ("31jCMtMOjZs", "FLOWER", "çiçek", 88.68, 91.52),
    ("31jCMtMOjZs", "ROSE", "gül", 91.80, 93.76),
    # The Sign Polyglot (MOTHER, MclATWdXRMw, skipped: 33 s of continuous explanation, no separable demonstration)
    ("xJxy4t9GDec", "SIGN LANGUAGE", "işaret dili", 5.80, 8.56,
     "second of two performances (İŞARET then DİL); hands detected in 75% of its frames against 68% in the first"),
    ("6QMph68LDAc", "DEAF", "sağır", 9.70, 11.10,
     "third of three performances: index finger raised at the cheek; the two closest repeats (DTW), all frames tracked"),
    ("Nn2ehWjwtNY", "HOME", "ev", 5.90, 7.64, "second of two performances (roof shape), hands detected in 95%"),
    ("iQ9kEKs2n7I", "Turkey", "türkiye", 7.60, 9.08, "first of two performances, hands detected in 95%"),
]


def fetch(vid):
    video, track = WORK / f"{vid}.mp4", WORK / f"{vid}.track.npz"
    if not video.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f",
                        "bv*[height<=720][ext=mp4]+ba/b[height<=720]", "--merge-output-format", "mp4", "-o", str(video),
                        f"https://www.youtube.com/watch?v={vid}"], check=True, env={**os.environ, "PYTHONPATH": str(YTDLP)})
    if not track.exists():
        subprocess.run([str(MP_PYTHON), str(ROOT / "scripts" / "face" / "track.py"), str(video), str(track)], check=True)
    return track


def existing_sign(lex, gloss):
    """The lexicon's sign for this gloss, if it has one: the gloss itself (or a reviewed synonym), or a gloss that is
    the same word (gelmek is signed by «gel»). A lemma match through another word (karınca read as kar+ınca) and the
    stem fallback do not count."""
    from isharati.text import turkish_morph as morph
    e = lex.lookup(gloss)
    if e is not None:
        return e
    m, route = lex.match(gloss)
    if m is None or route == "prefix":
        return None
    if route == "lemma" and " " not in gloss:
        g, mg = morph.fold(lex._key(gloss)), morph.fold(lex._key(m.gloss))
        return m if g in morph.lemmas(mg) else None  # «gel» lemmatises to gelmek; «kar» never to karınca
    return m


def main():
    from attach import fifty, window_frame  # noqa: E402
    from isharati.lexicon import turkish
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACES.mkdir(parents=True, exist_ok=True)
    lex = turkish.tid_lexicon(extra=False)  # the lexicon without this source: whatever it signs keeps its sign
    rows, firsts, tracks = [], {}, {}
    counts = {}
    for item in SIGNS:
        vid, label, gloss, start, end = item[:5]
        note = item[5] if len(item) > 5 else ""
        dataset, source_name = VIDEOS[vid]
        if vid not in tracks:
            t = dict(np.load(fetch(vid), allow_pickle=True))
            meta = t["meta"].item() if t["meta"].dtype == object else json.loads(str(t["meta"]))
            tracks[vid] = (t, meta, rest_undetected_hands(fifty(t)))
        track, meta, raw50 = tracks[vid]
        s, e = int(round(start * FPS)), int(round(end * FPS))
        clip, neck, scale = window_frame(raw50[s:e])
        counts[vid] = counts.get(vid, 0) + 1
        sid = f"{'oh' if dataset == OURHANDS else 'spg'}_{vid}_{counts[vid]:02d}"
        np.save(OUT / "signs" / f"{sid}.npy", clip.astype(np.float32))
        face = (track["face"][s:e].astype(np.float32) - neck) / scale
        url = f"https://youtu.be/{vid}"
        np.savez_compressed(FACES / f"{sid}.npz", blend=track["blend"][s:e], face=face.astype(np.float16),
                            blend_names=np.array(meta["blend_names"]), aspect=meta["aspect"],
                            ratio=proportions.ratio(clip), match=0.0, start=start, end=end, video=url)
        notes = [n for n in (LEARNER_NOTE if dataset == OURHANDS else "", note) if n]
        row = {"gloss": gloss, "sign_id": sid, "dataset": dataset, "keypoints_path": f"extra/signs/{sid}.npy",
               "review_status": "pending", "is_religious": False, "is_letter": False,
               "source": f"{url}?t={int(start)}", "source_name": source_name, "span_s": [start, end],
               "english_label": label, "note": "; ".join(notes)}
        old = existing_sign(lex, gloss)
        key = lex._key(gloss)
        if old is not None:
            row["variant_of"] = old.sign_id  # the lexicon's own sign stays; this is another recording of it
            row["variant_of_dataset"] = old.dataset
        elif key in firsts:
            row["variant_of"] = firsts[key]  # a second recording of a word this source adds
        else:
            firsts[key] = sid
        rows.append(row)
        hands = float(np.mean(~np.isnan(track["left_hand"][s:e, 0, 0]) | ~np.isnan(track["right_hand"][s:e, 0, 0])))
        print(f"{gloss:16s} {sid:22s} {e - s:4d} fr  hands {hands:4.0%}  "
              f"{'variant of ' + row['variant_of'] if 'variant_of' in row else 'NEW'}")
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    new = [r for r in rows if "variant_of" not in r]
    print(f"{len(rows)} signs: {len(new)} new glosses, {len(rows) - len(new)} variants -> {OUT / 'entries.jsonl'}")


if __name__ == "__main__":
    main()
