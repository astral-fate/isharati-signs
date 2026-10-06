"""The end-to-end pipeline: a question -> grounded answer from the Qur'an and hadith -> sign glosses -> assembled pose ->
skeleton video, with a per-word report and a back-translation score.

  python -m isharati.pipeline "What are the pillars of Islam?"          # English -> ASL
  python -m isharati.pipeline --lang ar "ما هو الإسلام؟"                 # Arabic -> ArSL
  python -m isharati.pipeline --lang tr "Ramazan nedir?"   # Turkish -> TİD
  python -m isharati.pipeline --lang ur "رمضان کیا ہے؟"     # Urdu -> ISL

Output per question in ISHARATI_OUT_DIR (default data/asl/out/<id>/): video.mp4, pose.npy and report.json.
"""
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isharati.glossing.asl import ASLLLMGlosser
from isharati.llm import default_client
from isharati.lexicon.asl import asl_lexicon
from isharati.retrieval.engine import Corpus, Referral, answer
from isharati.backtranslate import DTWRecognizer, SampledRecognizer, score
from isharati.pose.poser import MissingAwarePoser
from isharati.types import FPS

from isharati.config import DATA

LEX = DATA / "asl" / "lexicon"
AR_LEX = DATA / "lexicon_v2" / "lexicon_qa.jsonl"  # KArSL + religious + Tawasol (scripts/lexicon/arsl/merge.py)
AR_TEMPLATES = DATA / "lexicon" / "templates.jsonl"
OUT_DIR = Path(os.environ.get("ISHARATI_OUT_DIR", DATA / "asl" / "out"))  # one folder per answered question


LANGS = ("en", "ar", "tr", "ur")


def lexicon_paths(lang: str) -> list[Path]:
    if lang == "ar":
        return [AR_LEX]
    if lang == "tr":
        from isharati.lexicon.turkish import ALPHABET, AUTSL, CUNEYT, EXTRA, NAMES, PAIRS_FIXED, TID_FILE, TID_ROOT, TID_YOUTUBE
        return [TID_ROOT / TID_FILE, TID_ROOT / PAIRS_FIXED, TID_ROOT / ALPHABET, TID_ROOT / NAMES,
                TID_ROOT / TID_YOUTUBE, TID_ROOT / AUTSL, TID_ROOT / CUNEYT, TID_ROOT / EXTRA]
    if lang == "ur":
        from isharati.lexicon.isl import entries_files
        return entries_files()
    from isharati.lexicon.asl import ALPHABET, NAMES
    return [LEX / "lexicon_all.jsonl", LEX / ALPHABET, LEX / NAMES]


def stamp(lang: str = "en") -> str:
    """Changes whenever the lexicon or the answering/glossing code changes; a saved answer is replayed only while its
    stamp still matches (server.py), so fixes and new signs show up instead of an old result."""
    pkg = Path(__file__).parent
    code = [pkg / "retrieval" / "engine.py", pkg / "retrieval" / "arabic.py", pkg / "retrieval" / "turkish_urdu.py",
            pkg / "glossing" / "asl.py", pkg / "glossing" / "llm.py", pkg / "glossing" / "arabic.py",
            pkg / "text" / "arabic_morph.py", pkg / "lexicon" / "arabic.py", pkg / "lexicon" / "turkish.py",
            pkg / "lexicon" / "isl.py", pkg / "pose" / "poser.py", Path(__file__)]
    files = [*lexicon_paths(lang), *code]
    parts = [f"{f.name}:{f.stat().st_mtime_ns}:{f.stat().st_size}" for f in files if f.exists()]
    parts.append(f"files:{len(lexicon_paths(lang))}")  # Urdu: CISLR alone or CISLR + WSLP
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def question_id(question: str, lang: str = "en", kind: str = "answer") -> str:
    """Output folder name; English answers keep the original ids so earlier runs replay. Signed texts (Studio) get
    their own ids, so a text never replays an answer to the same words."""
    h = hashlib.sha1(question.encode()).hexdigest()[:8]
    if kind == "text":
        return "t" + lang + h
    return h if lang == "en" else lang + h


REPORT_VERSION = 8  # 2: complete sentences, rest pose; 3: verses quoted; 4: one body shape; 5: pose.npy for the avatar;
# 6: blend.npy, the signer's face; 7: rigid-skeleton body repair (keypoints.repair_body), relaxed rest hands;
# 8: the body shape in square units (proportions.REF 0.74 -> 0.50)


def load_lexicon(lang: str):
    """The sign lexicon for a language, with one body shape across its sources (pose.proportions)."""
    from isharati.pose import proportions
    if lang == "ar":
        from isharati.lexicon import arabic as ar_lexicon
        lexicon = ar_lexicon.qa_lexicon(AR_LEX)
    elif lang == "tr":
        from isharati.lexicon.turkish import tid_lexicon
        lexicon = tid_lexicon()
    elif lang == "ur":
        from isharati.lexicon.isl import isl_lexicon
        lexicon = isl_lexicon()
    else:
        # Islamic signs first, then dictionaries (scripts/lexicon/asl/merge.py), and the A-Z letters (alphabet/)
        lexicon = asl_lexicon()
    return proportions.apply(lexicon)


class Pipeline:
    """Loaded once (corpus, lexicon, templates) and reused per question, by this script and by src/isharati/app/server.py."""

    def __init__(self, lang: str = "en", client=None):
        if lang not in LANGS:
            raise ValueError(f"unsupported language {lang!r}")
        self.lang = lang
        self.client = client or default_client()
        self.lexicon = load_lexicon(lang)
        templates = None
        if lang == "ar":  # Arabic question -> Arabic sources -> Arabic Sign Language lexicon
            from isharati.glossing import arabic as ar_glossing
            from isharati.retrieval import arabic as ar_retrieval
            self.corpus, templates = ar_retrieval.corpus(), AR_TEMPLATES
            self.glosser = ar_glossing.ArabicQAGlosser(self.lexicon, self.client)  # LLM only, no rule-based glossing
        elif lang in ("tr", "ur"):  # Turkish -> TİD, Urdu -> ISL (English glosses)
            from isharati.glossing.llm import ISL_SPEC, TID_SPEC, LLMGlosser
            from isharati.retrieval import turkish_urdu
            self.corpus = turkish_urdu.corpus(lang)
            self.glosser = LLMGlosser(self.lexicon, self.client, TID_SPEC if lang == "tr" else ISL_SPEC)
        else:
            self.corpus, templates = Corpus(), LEX / "templates.jsonl"
            self.glosser = ASLLLMGlosser(self.lexicon, self.client)
        if templates is not None and templates.exists():
            self.recognizer = DTWRecognizer.from_templates(templates)
        elif lang in ("tr", "ur"):
            self.recognizer = SampledRecognizer(self.lexicon)
        else:
            self.recognizer = DTWRecognizer.from_lexicon(self.lexicon)

    def run(self, question: str) -> dict:
        """The report; {"status": "referred" | "unanswered", "reason": ...} when nothing is signed."""
        try:
            ans = answer(question, self.corpus, self.client)
        except Referral as e:
            return {"question": question, "status": "referred", "reason": str(e)}
        except LookupError as e:
            return {"question": question, "status": "unanswered", "reason": str(e)}
        return _sign(question, ans, self.lexicon, self.glosser, self.client, self.recognizer, self.lang)

    def sign_text(self, text: str) -> dict:
        """The user's own text, signed as written: no retrieval, no answer generation, no sources."""
        text = text.strip()
        glosser = self.glosser
        if self.lang == "en":  # the rule glosser keeps English working when the free LLM quota runs out
            from isharati.glossing.asl import ASLRuleGlosser
            from isharati.glossing.llm import FallbackGlosser
            glosser = FallbackGlosser(self.glosser, ASLRuleGlosser(self.lexicon))
        ans = {"answer": text, "sources": [], "dropped_unsupported": []}
        return _sign(text, ans, self.lexicon, glosser, self.client, self.recognizer, self.lang, kind="text")


def _sign(question, ans, lexicon, glosser, client, recognizer, lang="en", kind="answer"):
    result = glosser.gloss(ans["answer"])
    pose, segments = MissingAwarePoser(lexicon).pose(result)  # no fingerspelling: missing signs are marked
    pose = np.nan_to_num(pose, nan=0.0).astype(np.float32)
    if hasattr(recognizer, "for_glosses"):  # Turkish / Urdu: this answer's signs plus fixed distractors
        recognizer = recognizer.for_glosses(result.glosses)
    bt = score(pose, segments, result.glosses, recognizer)

    out = OUT_DIR / question_id(question, lang, kind)
    from isharati.pose.render import SkeletonRenderer  # OpenCV is loaded only when a video is rendered
    video = SkeletonRenderer().render(pose, out / "video.mp4")
    np.save(out / "pose.npy", pose.astype(np.float32))  # drives the 3D avatar (scripts/lexicon/asl/avatar.js)
    from isharati.pose import face
    signer_face = face.track(segments, len(pose), lang, lexicon)  # the signer's face, frame for frame
    if signer_face is not None:
        np.save(out / "blend.npy", signer_face["blend"])
        np.save(out / "face.npy", signer_face["points"].astype(np.float16))
        (out / "blend_names.json").write_text(json.dumps(signer_face["names"]), encoding="utf-8")
    report = {
        "status": "signed", "mode": kind, "version": REPORT_VERSION, "stamp": stamp(lang), "lang": lang, "question": question, "answer": ans["answer"],
        "dropped_unsupported": ans.get("dropped_unsupported", []),
        "sources": [{"reference": s["reference"], "url": s["url"]} for s in ans["sources"]],
        "glosser": result.backend, "llm": getattr(client, "last", None), "glosses": result.glosses, "missing_signs": result.oov,
        "coverage": len(result.glosses) / max(1, len(result.glosses) + len(result.oov)),
        "pending_review_signs": sorted({r.gloss for r in result.trace if r.review_status == "pending"}),
        "backtranslation": {"gloss_acc": bt.gloss_acc, "wer": bt.wer, "recognized": bt.recognized},
        "segments": [{"label": s.label, "kind": s.kind, "start_s": s.start / FPS, "end_s": s.end / FPS}
                     for s in segments if s.kind != "transition"],
        "trace": [asdict(r) for r in result.trace], "video": str(video),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    report["id"] = out.name
    return report


def main():
    args = sys.argv[1:]
    lang = "en"
    if args[:1] == ["--lang"] and len(args) > 1:
        lang, args = args[1], args[2:]
    question = " ".join(args) or "What are the pillars of Islam?"
    report = Pipeline(lang).run(question)
    if report["status"] != "signed":
        print(f"{report['status']}, nothing signed: {report['reason']}")
        return
    print(f"Q: {question}\nA: {report['answer']}\nSources: {', '.join(s['reference'] for s in report['sources'])}")
    print(f"ASL: {' '.join(report['glosses'])}\nMissing signs (to collect): {', '.join(report['missing_signs']) or 'none'}")
    print(f"Coverage {report['coverage']:.0%}; back-translation {report['backtranslation']['gloss_acc']:.0%}; "
          f"video: {report['video']}")


if __name__ == "__main__":
    main()
