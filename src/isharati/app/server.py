"""The Isharati web application: the React landing page (web/, built to web/dist) with the translator, and the API it
calls. The previous single-file page stays at /classic.

  python -m isharati.app.server          # http://127.0.0.1:8765 (HOST / PORT override; the Space uses 0.0.0.0:7860)
"""
import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from isharati import hub
from isharati.config import ROOT
from isharati.types import FPS

hub.ensure_data()  # on the Space: download the dataset before anything reads it
from isharati.pipeline import LANGS, OUT_DIR as OUT, REPORT_VERSION, Pipeline, question_id, stamp  # noqa: E402

PAGE = Path(__file__).parent / "ui.html"
STATIC = Path(__file__).parent / "static"  # avatar.vrm, avatar.js
STATS = STATIC / "stats.json"  # scripts/eval/app_stats.py
WEB_DIST = Path(os.environ.get("ISHARATI_WEB_DIST", ROOT / "web" / "dist"))  # the React landing page (web/, Vite)
app = FastAPI(title="Isharati ASL demo")
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=2048)  # pose JSON with the face is ~5x smaller compressed


@app.middleware("http")
async def cache_rules(request, call_next):
    """Files with fixed names (the page, avatar.js, the landing clips, VRM models) are revalidated on every load, so a
    deploy is seen at once instead of a browser replaying an old copy (an avatar card kept the pre-fix hands); Vite's
    hashed assets/ never change under their name and are cached for good."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    elif path.endswith(".vrm"):  # ~9 MB models that rarely change: a day, not a full download on every visit
        response.headers.setdefault("Cache-Control", "public, max-age=86400")
    elif path == "/" or path.startswith(("/clips/", "/static/", "/audio/", "/studio")) or path.endswith(".html"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response
_pipelines, _lock = {}, threading.Lock()


def pipeline(lang: str):
    with _lock:  # each language's corpus and lexicon load once, on its first question
        if lang not in _pipelines:
            _pipelines[lang] = Pipeline(lang)
    return _pipelines[lang]


class Ask(BaseModel):
    question: str
    lang: str = "en"     # "en" ASL, "ar" ArSL, "tr" TİD, "ur" ISL
    fresh: bool = False  # True regenerates instead of replaying a question already signed


@app.get("/")
@app.get("/review")
def page():
    index = WEB_DIST / "index.html"
    return FileResponse(index if index.is_file() else PAGE)


@app.get("/classic")
def classic():
    return FileResponse(PAGE)


def _from_dist(sub: str, name: str):
    base = (WEB_DIST / sub).resolve()
    path = (base / name).resolve()
    if not path.is_file() or base not in path.parents:  # no escaping the build folder
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/assets/{name:path}")
def assets(name: str):
    return _from_dist("assets", name)


@app.get("/clips/{name:path}")
def clips(name: str):
    return _from_dist("clips", name)


@app.get("/audio/{name:path}")
def audio(name: str):
    for base in [WEB_DIST / "audio", ROOT / "web" / "public" / "audio"]:
        p = (base / name).resolve()
        if p.is_file() and base.resolve() in p.parents:
            return FileResponse(p)
    raise HTTPException(404)


@app.get("/team/{name:path}")
def team_media(name: str):
    for base in [WEB_DIST / "team", ROOT / "web" / "public" / "team"]:
        p = (base / name).resolve()
        if p.is_file() and base.resolve() in p.parents:
            return FileResponse(p)
    raise HTTPException(404)



@app.get("/favicon.svg")
def favicon():
    return _from_dist(".", "favicon.svg")


@app.get("/record")
def record():
    """The headless clip recorder (scripts/avatars/record_clips.py)."""
    return _from_dist(".", "record.html")


def _replay(saved: Path, lang: str):
    """A saved run made by this pipeline with this lexicon, or None (then it is regenerated)."""
    if not saved.exists():
        return None
    report = json.loads(saved.read_text(encoding="utf-8"))
    if report.get("glosser") == "rule":  # the fallback glosser ran because the LLM quota was out: try the LLM again
        return None
    if report.get("version") == REPORT_VERSION and report.get("stamp") == stamp(lang):
        return {**report, "status": "signed", "id": saved.parent.name, "cached": True,
                "dropped_unsupported": report.get("dropped_unsupported", [])}
    return None


def _fail(e: Exception):
    if "free-models-per-day" in str(e):
        raise HTTPException(503, "The free language-model quota for today is used up. Try again tomorrow, "
                                 "or ask a question that has already been signed.")
    raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/ask")
def ask(body: Ask):  # sync: FastAPI runs it in a worker thread, so the page stays responsive
    q = body.question.strip()
    if not q or len(q) > 300:
        raise HTTPException(400, "Ask a question of 1 to 300 characters.")
    if body.lang not in LANGS:
        raise HTTPException(400, "lang must be one of " + ", ".join(LANGS))
    if not body.fresh and (r := _replay(OUT / question_id(q, body.lang) / "report.json", body.lang)):
        return r
    try:
        return pipeline(body.lang).run(q)
    except ValueError as e:
        if "nothing in the lexicon" in str(e):
            raise HTTPException(422, "None of these words has a recorded sign yet.")
        _fail(e)
    except Exception as e:  # shown on the page rather than as a bare 500
        _fail(e)


class SignText(BaseModel):
    text: str
    lang: str = "en"
    fresh: bool = False


@app.post("/api/sign")
def sign(body: SignText):
    """Studio: the user's own text in sign language, without retrieval or answer generation."""
    text = body.text.strip()
    if not text or len(text) > 1000:
        raise HTTPException(400, "Write a text of 1 to 1,000 characters.")
    if body.lang not in LANGS:
        raise HTTPException(400, "lang must be one of " + ", ".join(LANGS))
    if not body.fresh and (r := _replay(OUT / question_id(text, body.lang, "text") / "report.json", body.lang)):
        return r
    try:
        return pipeline(body.lang).sign_text(text)
    except ValueError as e:
        if "nothing in the lexicon" in str(e):
            raise HTTPException(422, "None of these words has a recorded sign yet.")
        _fail(e)
def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    for p in [Path("D:/islam/.env"), ROOT / ".env", Path(".env")]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return ""


# browser recordings arrive as a nameless blob: their extension from the MIME type
_SUFFIX = {"audio/webm": ".webm", "video/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "video/mp4": ".mp4",
           "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "video/quicktime": ".mov"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe spoken video/audio using Groq Whisper-large-v3, returning text and word/segment timestamps."""
    groq_key = _get_groq_key()
    if not groq_key:
        raise HTTPException(500, "GROQ_API_KEY is not configured on the server.")

    suffix = Path(file.filename or "").suffix or _SUFFIX.get((file.content_type or "").split(";")[0], ".webm")
    content = await file.read()
    if len(content) < 1000:
        raise HTTPException(400, "The recording is empty or too short: check the microphone and record again.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        in_path = tmp_in.name
        tmp_in.write(content)

    tmp_flac = in_path + ".flac"
    try:
        # 16 kHz mono FLAC for Whisper, with the ffmpeg bundled in imageio-ffmpeg (the Space image has no system ffmpeg)
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-vn", tmp_flac]
        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode != 0 or not os.path.exists(tmp_flac) or os.path.getsize(tmp_flac) <= 100:
            tail = res.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] if res.stderr else []
            raise HTTPException(400, f"Could not read audio from the file ({' '.join(tail) or 'no audio stream'}).")
        target_file = tmp_flac

        with open(target_file, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": (Path(target_file).name, f)},
                data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                timeout=120,
            )

        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Groq Whisper API error: {resp.text[:200]}")

        data = resp.json()
        raw_segs = data.get("segments", [])
        segments = [
            {
                "text": s.get("text", "").strip(),
                "start": round(float(s.get("start", 0)), 2),
                "end": round(float(s.get("end", 0)), 2),
            }
            for s in raw_segs
        ]
        return {
            "text": data.get("text", "").strip(),
            "segments": segments,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        for p in [in_path, tmp_flac]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.get("/pose/{vid}.json")
def pose(vid: str):
    """The stitched pose [T,50,3] at 25 fps, for the avatar."""
    path = OUT / vid / "pose.npy"
    if not vid.isalnum() or not path.exists():
        raise HTTPException(404)
    p = np.nan_to_num(np.load(path)).astype(float).round(4)
    out = {"fps": FPS, "frames": p.tolist()}
    if (OUT / vid / "blend.npy").exists():  # the signer's face blendshapes, same frames (isharati.pose.face)
        from isharati.pose import face
        out["blend"] = face.as_json(np.load(OUT / vid / "blend.npy"))
        out["blend_names"] = json.loads((OUT / vid / "blend_names.json").read_text(encoding="utf-8"))
        if (OUT / vid / "face.npy").exists():  # face contour points for the skeleton view, and how to join them
            out["face"] = face.as_json(np.load(OUT / vid / "face.npy")[..., :2])
            out["face_edges"] = face.CONTOURS["edges"]
    return out


@app.get("/api/avatars")
def avatars():
    """The VRM avatar models in src/isharati/app/static (add a .vrm file there to offer another one)."""
    names = sorted(p.name for p in STATIC.glob("*.vrm"))
    return ["avatar.vrm"] + [n for n in names if n != "avatar.vrm"] if "avatar.vrm" in names else names


@app.get("/api/academy/curriculum")
def academy_curriculum(ui: str = "en"):
    """Isharati Academy: learning paths -> lessons -> items, each with the sign languages that have a real recording."""
    from isharati import academy
    return academy.curriculum(ui if ui in ("en", "ar", "tr", "ur") else "en")


@app.get("/api/academy/sign/{lang}/{item_id}.json")
def academy_sign(lang: str, item_id: str):
    """One curriculum item's sign in one sign language, between rest poses, with the signer's face."""
    from isharati import academy
    out = academy.sign_pose(lang, item_id)
    if out is None:
        raise HTTPException(404, "no recorded sign for this item in this sign language")
    return out


@app.get("/api/stats")
def stats():
    """The landing page's figures: per-language signs, passages and coverage, the licences, and the avatar count."""
    from isharati.lexicon.isl import wslp_enabled
    s = json.loads(STATS.read_text(encoding="utf-8"))
    for lang in s["languages"]:
        if lang.get("signs_by_source") and not wslp_enabled():
            lang["signs"] = lang["signs_by_source"]["cislr"]
            lang["glosses"] = lang["glosses_by_source"]["cislr"]
    s["avatars"] = len(avatars())
    return s


@app.get("/api/quran/text")
def quran_text():
    """The Qur'an text by surah and ayah (Uthmani and clean), from the isharati-app-data dataset (Tanzil text)."""
    from isharati import app_data
    p = app_data.path("quran/quran.json")
    if p is None:
        raise HTTPException(503, "Qur'an text unavailable")
    return FileResponse(p, media_type="application/json", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/quran/coverage")
def quran_coverage():
    """Per-source totals and per-surah recitation/tafsir segment counts of the Qur'an sign-language datasets."""
    from isharati import quran_data
    return quran_data.load().coverage()


@app.get("/api/quran/ayah")
def quran_ayah(surah: int, ayah: int):
    """Every segment covering one ayah, plus the nearest covered ayah (for the 'no signing here' case)."""
    from isharati import quran_data
    if not 1 <= surah <= 114 or ayah < 1:
        raise HTTPException(400, "surah must be 1..114 and ayah >= 1")
    q = quran_data.load()
    return {"surah": surah, "ayah": ayah, "segments": q.segments(surah, ayah), "nearest": q.nearest(surah, ayah)}


@app.get("/api/quran/pose/{source}/{segment_id}.json")
def quran_pose(source: str, segment_id: str):
    """The 25 fps Isharati pose of one segment ([frames][50 joints][x, y, z])."""
    from isharati import quran_data
    if source not in quran_data.SOURCES or not re.fullmatch(r"[0-9]{3}[A-Za-z0-9_-]*", segment_id):
        raise HTTPException(404)
    pose = quran_data.load().pose(source, segment_id)
    if pose is None:
        raise HTTPException(404)
    return pose


@app.get("/api/hadith/coverage")
def hadith_coverage():
    """Totals of the signed hadith dataset: samples, distinct hadith and hours per source, sign language and collection."""
    from isharati import hadith_data
    return hadith_data.load().coverage()


@app.get("/api/hadith/list")
def hadith_list(collection: str = "", sign_language: str = ""):
    """The browsable hadith (optionally one collection / one sign language): ref, number, a short matn, sample count."""
    from isharati import hadith_data
    if collection and not re.fullmatch(r"[a-z0-9_]{1,32}", collection):
        raise HTTPException(400, "collection must be a collection key such as nawawi or bukhari")
    if sign_language and sign_language not in hadith_data.SIGN_LANGUAGES:
        raise HTTPException(400, f"sign_language must be one of {', '.join(hadith_data.SIGN_LANGUAGES)}")
    return {"hadith": hadith_data.load().hadith_list(collection or None, sign_language or None)}


@app.get("/api/hadith/item")
def hadith_item(ref: str):
    """One hadith's text and every signed sample of it (signer, original video, gloss, where the reading starts)."""
    from isharati import hadith_data
    if not hadith_data.REF_RE.fullmatch(ref):
        raise HTTPException(400, "ref must look like collection:number, e.g. nawawi:12")
    out = hadith_data.load().item(ref)
    if out is None:
        raise HTTPException(404, "no signed sample of this hadith")
    return out


@app.get("/api/hadith/pose/{sample_id}.json")
def hadith_pose(sample_id: str):
    """The 25 fps Isharati pose of one signed hadith sample ([frames][50 joints][x, y, z])."""
    from isharati import hadith_data
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sample_id):
        raise HTTPException(404)
    pose = hadith_data.load().pose(sample_id)
    if pose is None:
        raise HTTPException(404)
    return pose


# ─── Neon DB Sign Review & Annotation Endpoints ──────────────────────────────
# the connection string is a secret: set DATABASE_URL in the environment (a Space secret on Hugging Face, .env locally)
DB_URI = os.environ.get("DATABASE_URL")

def _get_db():
    if not DB_URI:
        raise HTTPException(503, "review database not configured: set the DATABASE_URL secret")
    import psycopg2
    return psycopg2.connect(DB_URI)


class ReviewSubmission(BaseModel):
    decision: str  # "true" (Approved), "false" (Rejected), "need_modification"
    notes: str = ""
    reviewer: str = "expert_reviewer"


@app.get("/api/reviews/pending")
def get_pending_reviews(limit: int = 50, offset: int = 0, dataset: str = "all", status: str = "all"):
    """Fetch signs queued for expert review and annotation, optionally filtered by dataset and status."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                where_clauses = []
                params = []
                if dataset and dataset != "all":
                    where_clauses.append("dataset = %s")
                    params.append(dataset)
                if status and status != "all":
                    if status == "needs_revision":
                        where_clauses.append("status IN ('pending', 'needs_modification')")
                    else:
                        where_clauses.append("status = %s")
                        params.append(status)

                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                cur.execute(f"""
                    SELECT id, sign_id, gloss, dataset, status, decision, notes, reviewer, keypoints_path, updated_at
                    FROM sign_reviews
                    {where_sql}
                    ORDER BY CASE WHEN status = 'pending' THEN 0 WHEN status = 'needs_modification' THEN 1 ELSE 2 END, id ASC
                    LIMIT %s OFFSET %s;
                """, (*params, limit, offset))
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

                cur.execute(f"SELECT COUNT(*), COUNT(*) FILTER (WHERE status = 'pending') FROM sign_reviews {where_sql};", params)
                total, pending_count = cur.fetchone()

                # Also return available datasets and status counts
                cur.execute("SELECT dataset, COUNT(*) FROM sign_reviews GROUP BY dataset ORDER BY COUNT(*) DESC;")
                ds_breakdown = dict(cur.fetchall())
                cur.execute("SELECT status, COUNT(*) FROM sign_reviews GROUP BY status;")
                status_breakdown = dict(cur.fetchall())

                return {
                    "items": rows,
                    "total": total,
                    "pending": pending_count,
                    "datasets": ds_breakdown,
                    "statuses": status_breakdown,
                }
    except Exception as e:
        raise HTTPException(500, f"Database query failed: {e}")


# Global in-memory / disk cache for review poses from Hugging Face
_REVIEW_POSES_CACHE = {}

def _sanitize_pose_frames(frames: np.ndarray) -> np.ndarray:
    """Sanitize [T, 50, 3] frames by zeroing out collapsed/untracked hand points."""
    for t in range(len(frames)):
        for base in (8, 29):
            hand = frames[t, base:base + 21, :2]
            spread = float(np.ptp(hand, axis=0).max())
            if spread < 0.02:
                frames[t, base:base + 21] = 0.0
    return frames


@app.get("/api/reviews/pose/{sign_id}.json")
def get_review_pose(sign_id: str):
    """Load keypoints trajectory [T, 50, 3] for a review sign (local disk or HF Hub dataset)."""
    global _REVIEW_POSES_CACHE
    if sign_id in _REVIEW_POSES_CACHE:
        return {"fps": FPS, "frames": _REVIEW_POSES_CACHE[sign_id]}

    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT keypoints_path FROM sign_reviews WHERE sign_id = %s;", (sign_id,))
                row = cur.fetchone()
                if not row or not row[0]:
                    raise HTTPException(404, "Sign keypoint path not found")
                kp_rel = row[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")

    # 1. Search in local lexicon directories
    candidates = [
        ROOT / "data" / "lexicon_v2" / kp_rel,
        ROOT / "data" / "lexicon" / kp_rel,
        Path("D:/islam/isharati/data/lexicon_v2") / kp_rel,
        Path("D:/islam/isharati/data/lexicon") / kp_rel,
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p:
        frames = np.nan_to_num(np.load(p)).astype(float).round(4)
        if frames.ndim == 2:
            frames = frames.reshape(-1, 50, 3)
        frames = _sanitize_pose_frames(frames)
        _REVIEW_POSES_CACHE[sign_id] = frames.tolist()
        return {"fps": FPS, "frames": _REVIEW_POSES_CACHE[sign_id]}

    # 2. Fallback: Download / stream from Hugging Face dataset FatimahEmadEldin/isharati-review-signs
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        cache_npz = ROOT / "data" / "review_poses.npz"
        if not cache_npz.exists():
            downloaded = hf_hub_download(
                repo_id="FatimahEmadEldin/isharati-review-signs",
                repo_type="dataset",
                filename="review_poses.npz",
                token=token,
            )
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(downloaded, cache_npz)

        with np.load(cache_npz) as z:
            if sign_id in z:
                frames = np.nan_to_num(z[sign_id]).astype(float).round(4)
                if frames.ndim == 2:
                    frames = frames.reshape(-1, 50, 3)
                frames = _sanitize_pose_frames(frames)
                _REVIEW_POSES_CACHE[sign_id] = frames.tolist()
                return {"fps": FPS, "frames": _REVIEW_POSES_CACHE[sign_id]}
    except Exception as ex:
        print(f"Error fetching from review dataset: {ex}", flush=True)

    raise HTTPException(404, f"Pose trajectory missing for sign: {sign_id}")


@app.post("/api/reviews/{sign_id}")
def submit_review(sign_id: str, body: ReviewSubmission):
    """Record expert annotation (true / false / need_modification) into Neon PostgreSQL."""
    valid_decisions = {"true", "false", "need_modification"}
    if body.decision not in valid_decisions:
        raise HTTPException(400, f"decision must be one of {valid_decisions}")
    status_map = {
        "true": "approved",
        "false": "rejected",
        "need_modification": "needs_modification"
    }
    status = status_map[body.decision]
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sign_reviews
                    SET status = %s, decision = %s, notes = %s, reviewer = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE sign_id = %s;
                """, (status, body.decision, body.notes, body.reviewer, sign_id))
                if cur.rowcount == 0:
                    raise HTTPException(404, "Sign not found in review queue")
                conn.commit()
                return {"ok": True, "sign_id": sign_id, "status": status, "decision": body.decision}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Database update failed: {e}")


@app.post("/api/reviews/record/{sign_id}")
def upload_recorded_sign(sign_id: str, video: UploadFile = File(...)):
    """Upload webcam-recorded sign video for re-capture / replacement."""
    os.makedirs(ROOT / "data" / "recordings", exist_ok=True)
    out_file = ROOT / "data" / "recordings" / f"{sign_id}_recapture.webm"
    try:
        content = video.file.read()
        out_file.write_bytes(content)
        # Record video URL in review row
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sign_reviews
                    SET video_url = %s, notes = COALESCE(notes, '') || ' [Camera Re-recorded]', updated_at = CURRENT_TIMESTAMP
                    WHERE sign_id = %s;
                """, (str(out_file), sign_id))
                conn.commit()
        return {"ok": True, "sign_id": sign_id, "saved_path": str(out_file), "size": len(content)}
    except Exception as e:
        raise HTTPException(500, f"Failed saving recording: {e}")



@app.get("/static/{name}")
def static(name: str):
    path = STATIC / name
    if "/" in name or "\\" in name or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/video/{vid}.mp4")
def video(vid: str):
    path = OUT / vid / "video.mp4"
    if not vid.isalnum() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8765)))
