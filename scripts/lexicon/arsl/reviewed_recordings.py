"""Recordings chosen by the team to replace a weak sign: an isolated video of one signer, cut to one performance.

Each entry names the gloss, the video and the span of the performance to keep. The video is tracked with the
Holistic task (scripts/face/track.py, the face pass's tracker), so the sign and its face come from the same frames;
the clip is normalize(interpolate_missing()) of that span, as every lexicon clip, and the face is put in the same frame
(attach.save_face's convention). merge.py lists data/lexicon_v2/reviewed/recordings.jsonl right after the
native-signer decisions, and tid_lexicon() loads data/lexicon_tr/reviewed/recordings.jsonl over the dictionary,
so these replace any other sign of the same gloss.

  py scripts/lexicon/arsl/reviewed_recordings.py
Output: reviewed/recordings.jsonl, reviewed/signs/<sign_id>.npy, data/face/ar/<sign_id>.npz (videos and tracks on F:)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "face")]
from isharati.config import DATA  # noqa: E402
from isharati.pose import proportions  # noqa: E402

WORK = Path(r"F:\reviewed_recordings")
MP_PYTHON = Path(r"D:\islam\gathring data\.venv\Scripts\python.exe")  # the environment with MediaPipe
YTDLP = Path(r"D:\islam\yt-dlp")
LEX = {"ar": (DATA / "lexicon_v2", DATA / "face" / "ar"), "tr": (DATA / "lexicon_tr", DATA / "face" / "tr")}  # lexicon root, faces
FPS = 25

RECORDINGS = [
    {"gloss": "لا إله إلا الله", "youtube": "4LVGk_S1o5E", "start_s": 2.0, "end_s": 6.72,
     "note": "isolated recording (Egyptian signs channel, «لا الة الا الله»), first of two performances; replaces the "
             "new_arabic_sources clip, whose body tracking failed (arm thrown across the face)",
     "chosen_by": "project team", "chosen_on": "2026-10-05", "lang": "ar", "review_status": "approved"},
    {"gloss": "إن شاء الله", "youtube": "nHZ80gorFuI", "start_s": 3.4, "end_s": 5.2,
     "note": "Amro Ahmed short «إن شاء الله = باذن الله»: the raised open hand then the index pointing up (first of two "
             "repetitions); the signs before it (1-3 s) are the «إنشاء» he contrasts it with. Chosen by the team as "
             "clearer than the lexicon clip",
     "chosen_by": "project team", "chosen_on": "2026-10-05", "lang": "ar", "review_status": "approved"},
    # TİD: words the dictionary channels lack (red in the gloss list). Picked by matching the repeated performances
    # (DTW), so pending a native signer's check
    {"gloss": "zekât", "youtube": "DPh1CXybQ-g", "start_s": 38.04, "end_s": 40.92,
     "note": "Cüneyt Küçükoğlu, Hareketli Sözlük «Zekat - işaret dili»: the third of three matching performances "
             "(DTW 0.26-0.35 between them, 0.65-0.84 to the other runs), the shortest",
     "chosen_by": "matched performances", "chosen_on": "2026-10-05", "lang": "tr", "review_status": "pending"},
    {"gloss": "şart", "youtube": "MeRAJuSnO5o", "start_s": 0.0, "end_s": 1.52,
     "note": "İşaret Dili Eğitim «Şart -- İşaret Dili Eğitimi» (4 s): the front view, before the same sign in profile",
     "chosen_by": "isolated clip", "chosen_on": "2026-10-05", "lang": "tr", "review_status": "pending"},
]


def rest_undetected_hands(raw50, max_gap=3):
    """A hand the tracker loses for more than max_gap frames is resting out of shot, not frozen where it was last seen:
    interpolating across the gap held the shahada signer's left hand at the chin (detected in 24% of frames, only when
    the hands cross) for the whole first half of the sign. Such frames put the hand's 21 points on its body wrist, which
    keypoints.pin_collapsed_hands and the avatar read as a resting hand; short flickers are still interpolated."""
    out = raw50.copy()
    for hand, wrist in ((slice(8, 29), 7), (slice(29, 50), 4)):     # left hand / left wrist, right hand / right wrist
        missing = np.isnan(out[:, hand, 0]).all(axis=1)
        t = 0
        while t < len(missing):
            if missing[t]:
                u = t
                while u < len(missing) and missing[u]:
                    u += 1
                if u - t > max_gap:
                    out[t:u, hand] = out[t:u, wrist:wrist + 1]
                t = u
            else:
                t += 1
    return out


def fetch(rec):
    video = WORK / f"{rec['youtube']}.mp4"
    if not video.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f",
                        "bv*[height<=720][ext=mp4]+ba/b[height<=720]", "--merge-output-format", "mp4", "-o", str(video),
                        f"https://www.youtube.com/watch?v={rec['youtube']}"],
                       check=True, env={**__import__("os").environ, "PYTHONPATH": str(YTDLP)})
    track = WORK / f"{rec['youtube']}.track.npz"
    if not track.exists():
        subprocess.run([str(MP_PYTHON), str(ROOT / "scripts" / "face" / "track.py"), str(video), str(track)], check=True)
    return video, track


def main():
    from attach import fifty, window_frame  # noqa: E402
    rows = {lang: [] for lang in LEX}
    for rec in RECORDINGS:
        root, faces = LEX[rec["lang"]]
        out = root / "reviewed"
        (out / "signs").mkdir(parents=True, exist_ok=True)
        video, track_path = fetch(rec)
        track = dict(np.load(track_path, allow_pickle=True))
        meta = track["meta"].item() if track["meta"].dtype == object else json.loads(str(track["meta"]))
        s, e = int(round(rec["start_s"] * FPS)), int(round(rec["end_s"] * FPS))
        raw50 = rest_undetected_hands(fifty(track))
        clip, neck, scale = window_frame(raw50[s:e])
        sid = f"rr_{rec['youtube']}"
        np.save(out / "signs" / f"{sid}.npy", clip.astype(np.float32))
        face = (track["face"][s:e].astype(np.float32) - neck) / scale
        faces.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(faces / f"{sid}.npz", blend=track["blend"][s:e], face=face.astype(np.float16),
                            blend_names=np.array(meta["blend_names"]), aspect=meta["aspect"],
                            ratio=proportions.ratio(clip), match=0.0, start=rec["start_s"], end=rec["end_s"],
                            video=f"https://youtu.be/{rec['youtube']}")
        rows[rec["lang"]].append({"gloss": rec["gloss"], "sign_id": sid, "dataset": "reviewed_recording",
                     "keypoints_path": f"reviewed/signs/{sid}.npy", "review_status": rec["review_status"], "is_religious": True,
                     "is_letter": False, "source": f"https://youtu.be/{rec['youtube']}", "span_s": [rec["start_s"], rec["end_s"]],
                     "note": rec["note"], "chosen_by": rec["chosen_by"], "chosen_on": rec["chosen_on"]})
        print(f"{rec['gloss']}: {sid}, {e - s} frames ({rec['start_s']}-{rec['end_s']} s), "
              f"face {float(np.mean(~np.isnan(track['blend'][s:e, 0]))):.0%}")
    for lang, rs in rows.items():
        if rs:
            (LEX[lang][0] / "reviewed" / "recordings.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
