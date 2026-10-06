"""Precompute the answers to the page's suggested questions (web/src/sections/signLanguages.ts, EXAMPLES: three per
language) and publish them to the isharati-app-data dataset under answers/, so the Space shows a suggestion at once
instead of calling the models again after every restart (server.py: _saved, _from_hub).

Each answer is the folder the pipeline writes (report.json, pose.npy, face.npy, blend.npy, blend_names.json,
video.mp4), made with the Space's settings (OpenRouter models, hosted embeddings, Urdu with WSLP). A saved answer is
replayed only while its stamp (pipeline.stamp: the lexicon and the answering code, by content) matches the Space's, so
run this again after changing the lexicon or that code, then deploy; until then the Space answers live.

  env -u HF_TOKEN py scripts/hub/precompute_answers.py [--fresh] [--dry-run]
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ISHARATI_LLM", "openrouter")
os.environ.setdefault("ISHARATI_EMBED", "hf")
os.environ.setdefault("ISHARATI_URDU_WSLP", "1")
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import OUT_DIR, REPORT_VERSION, Pipeline, question_id, stamp  # noqa: E402

EXAMPLES_TS = ROOT / "web" / "src" / "sections" / "signLanguages.ts"
FILES = ("report.json", "pose.npy", "face.npy", "blend.npy", "blend_names.json", "video.mp4")
STAGE = Path(r"D:\islam\hub_build") / "app_answers"


def examples() -> dict[str, list[str]]:
    """EXAMPLES from the page's source, so the precomputed questions are exactly the ones it shows."""
    src = EXAMPLES_TS.read_text(encoding="utf-8")
    block = src[src.index("export const EXAMPLES"):]
    block = block[:block.index("};")]
    return {lang: re.findall(r'"([^"]+)"', body) for lang, body in re.findall(r"(\w\w): \[([^\]]*)\]", block)}


def current(qid: str, lang: str) -> bool:
    p = OUT_DIR / qid / "report.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text(encoding="utf-8"))
    return r.get("version") == REPORT_VERSION and r.get("stamp") == stamp(lang) and r.get("glosser") != "rule"


def main():
    fresh, dry = "--fresh" in sys.argv, "--dry-run" in sys.argv
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    index = {}
    for lang, questions in examples().items():
        pipe = None
        for q in questions:
            qid = question_id(q, lang)
            if fresh or not current(qid, lang):
                pipe = pipe or Pipeline(lang)
                print(f"[{lang}] answering: {q}", flush=True)
                pipe.run(q)
            if not current(qid, lang):  # an old or fallback answer would never replay on the Space: stop here
                raise SystemExit(f"[{lang}] {q}: no current answer in {OUT_DIR / qid} after running; nothing published")
            r = json.loads((OUT_DIR / qid / "report.json").read_text(encoding="utf-8"))
            files = [f for f in FILES if (OUT_DIR / qid / f).exists()]
            (STAGE / qid).mkdir()
            for f in files:
                shutil.copy2(OUT_DIR / qid / f, STAGE / qid / f)
            index[qid] = {"lang": lang, "question": q, "stamp": r.get("stamp"), "files": files,
                          "status": r.get("status", "signed")}
            print(f"[{lang}] {qid} {q}: {len(r.get('glosses', []))} glosses, {len(files)} files", flush=True)
    (STAGE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} answers staged in {STAGE}")
    if dry:
        return
    from huggingface_hub import HfApi
    sys.path.insert(0, str(Path(__file__).parent))
    from publish_app_data import CARD, REPO
    api = HfApi()
    repo = f"{api.whoami()['name']}/{REPO}"
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(STAGE), path_in_repo="answers",
                      commit_message="Precomputed answers to the suggested questions", delete_patterns=["*"])
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="Card: precomputed answers")
    print("published (private):", f"https://huggingface.co/datasets/{repo}/tree/main/answers")


if __name__ == "__main__":
    main()
