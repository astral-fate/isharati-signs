"""The figures the landing page shows, from the data: signs and distinct glosses per lexicon, approved passages, and
coverage (content-word token coverage of the Qur'an + hadith corpus, results/eda.json; Urdu through ISL's English
glosses). Writes src/isharati/app/static/stats.json, which the Space serves (it has no results/ folder).

  py scripts/eval/app_stats.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "hub")]
from isharati.config import DATA  # noqa: E402
from publish import LICENCES  # noqa: E402

OUT = ROOT / "src" / "isharati" / "app" / "static" / "stats.json"
LANGS = [  # code, language, sign language, lexicon files, eda.json key for coverage, eda.json key for passages
    ("en", "English", "ASL", [DATA / "asl" / "lexicon" / "lexicon_all.jsonl"], "en", "en"),
    ("ar", "Arabic", "ArSL", [DATA / "lexicon_v2" / "lexicon_qa.jsonl"], "ar", "ar"),
    ("tr", "Turkish", "TİD", [DATA / "lexicon_tr" / "dictionaries" / "entries_tid_dictionary.jsonl",
                              DATA / "lexicon_tr" / "tid_youtube" / "entries.jsonl", DATA / "lexicon_tr" / "autsl" / "entries.jsonl",
                              DATA / "lexicon_tr" / "names" / "entries.jsonl",
                              DATA / "lexicon_tr" / "cuneyt" / "entries.jsonl",
                              DATA / "lexicon_tr" / "extra" / "entries.jsonl"], "tr", "tr"),
    ("ur", "Urdu", "ISL · PSL", [DATA / "lexicon_isl" / "cislr" / "entries.jsonl",
                           DATA / "lexicon_isl" / "wslp" / "entries.jsonl",
                           DATA / "lexicon_isl" / "islrtc" / "entries.jsonl",
                           DATA / "lexicon_isl" / "psl" / "entries.jsonl"], "isl_on_en", "ur"),
]


def distinct_passages():
    """Distinct passage ids (q2:255, h1234 ...) across the four corpora: the languages share the same verses and
    hadith, so summing the per-language counts would count each one up to four times."""
    from isharati.retrieval import arabic, engine
    paths = [engine.CORPUS, arabic.CORPUS, DATA / "tr" / "corpus.jsonl", DATA / "ur" / "corpus.jsonl"]
    return len({json.loads(l)["pid"] for p in paths for l in p.read_text(encoding="utf-8").splitlines() if l.strip()})


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    eda = json.loads((ROOT / "results" / "eda.json").read_text(encoding="utf-8"))
    langs = []
    for code, name, sl, files, cov_key, pas_key in LANGS:
        by_source = {f.parent.name: [r for r in rows(f) if not r.get("is_letter") and not r.get("variant_of")] for f in files if f.exists()}
        signs = [r for rs in by_source.values() for r in rs]
        def gloss_count(rs):
            return len({r["gloss"].rstrip("0123456789").strip().lower() for r in rs})
        entry = {"code": code, "name": name, "sign_language": sl, "signs": len(signs),
                 "glosses": gloss_count(signs),
                 "passages": eda[pas_key]["passages"],
                 "coverage": round(eda[cov_key]["all"]["content_token_coverage"], 4)}
        if code == "ur":
            entry["signs_by_source"] = {k: len(v) for k, v in by_source.items()}
            entry["glosses_by_source"] = {k: gloss_count(v) for k, v in by_source.items()}
        langs.append(entry)
    out = {"languages": langs, "distinct_passages": distinct_passages(), "licences": [{"source": s, "licence": l, "use": u} for s, l, u in LICENCES],
           "created": date.today().isoformat()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["languages"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
