import csv
import json

import numpy as np
from isharati.lexicon.medoid import build, medoid


def test_medoid_picks_a_central_clip():
    a = np.zeros((10, 50, 3), np.float32)
    assert medoid([a, a + 0.01, a + 5]) in (0, 1)


def test_build_writes_lexicon_and_templates(tmp_path):
    (tmp_path / "npy").mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for sign in ("0100", "0400"):
        for signer in ("01", "02", "03"):
            for k in range(3):
                cid = f"{signer}_{sign}_{k}"
                np.save(tmp_path / "npy" / f"{cid}.npy", rng.normal(size=(15, 50, 3)).astype(np.float32))
                rows.append({"clip_id": cid, "signer": signer, "sign_id": sign, "npy_path": f"npy/{cid}.npy",
                             "frames": 15, "missing_ratio": "0.9" if k == 2 else "0.0", "ok": "True"})
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(tmp_path / "labels.csv", "w", newline="", encoding="utf-8") as f:
        f.write("sign_id,gloss_ar,chapter\n0100,وجه,health\n0400,صلاة,religion\n0999,مفقود,house\n")
    out = tmp_path / "lex"
    stats = build(tmp_path / "manifest.csv", tmp_path / "labels.csv", out, {"01", "02"}, {"03"})
    lex = [json.loads(l) for l in (out / "lexicon.jsonl").read_text(encoding="utf-8").splitlines()]
    tpl = [json.loads(l) for l in (out / "templates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {e["gloss"] for e in lex} == {"وجه", "صلاة"}
    assert {e["gloss"]: e["review_status"] for e in lex}["صلاة"] == "pending"
    assert {e["gloss"]: e["is_religious"] for e in lex}["صلاة"] is True
    assert len(tpl) == 2 and all(e["keypoints_path"].startswith("templates/") for e in tpl)
    assert {k: stats[k] for k in ("signs", "skipped", "religious")} == {"signs": 2, "skipped": 1, "religious": 1}
    assert all((out / e["keypoints_path"]).exists() for e in lex + tpl)


def _mini_manifest(tmp_path, sign_ids):
    (tmp_path / "npy").mkdir(exist_ok=True)
    rows = []
    for sid in sign_ids:
        for signer in ("01", "02"):
            cid = f"{signer}_{sid}"
            np.save(tmp_path / "npy" / f"{cid}.npy", np.zeros((5, 50, 3), np.float32))
            rows.append({"clip_id": cid, "signer": signer, "sign_id": sid, "npy_path": f"npy/{cid}.npy",
                         "frames": 5, "missing_ratio": "0.0", "ok": "True"})
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_build_reports_missing_letter_shapes(tmp_path):
    import pytest as _p
    _mini_manifest(tmp_path, ["0032", "0033"])
    (tmp_path / "labels.csv").write_text("sign_id,gloss_ar,chapter\n0032,ا,letters\n0033,ب,letters\n", encoding="utf-8")
    stats = build(tmp_path / "manifest.csv", tmp_path / "labels.csv", tmp_path / "lex", {"01", "02"}, {"03"})
    assert "ت" in stats["missing_letter_shapes"] and "ا" not in stats["missing_letter_shapes"]


def test_build_rejects_letter_glosses_that_are_names_not_characters(tmp_path):
    import pytest as _p
    _mini_manifest(tmp_path, ["0032"])
    (tmp_path / "labels.csv").write_text("sign_id,gloss_ar,chapter\n0032,ألف,letters\n", encoding="utf-8")
    with _p.raises(ValueError, match="letter"):
        build(tmp_path / "manifest.csv", tmp_path / "labels.csv", tmp_path / "lex", {"01", "02"}, {"03"})


def test_build_does_not_write_empty_templates_file(tmp_path):
    _mini_manifest(tmp_path, ["0100"])
    (tmp_path / "labels.csv").write_text("sign_id,gloss_ar,chapter\n0100,وجه,health\n", encoding="utf-8")
    build(tmp_path / "manifest.csv", tmp_path / "labels.csv", tmp_path / "lex", {"01", "02"}, {"03"})
    assert not (tmp_path / "lex" / "templates.jsonl").exists()


def test_build_keeps_a_sign_using_its_best_clip_when_none_pass_the_quality_threshold(tmp_path):
    (tmp_path / "npy").mkdir()
    rows = []
    for k, ratio in enumerate(("1.0", "0.6", "0.9")):
        cid = f"01_0032_{k}"
        np.save(tmp_path / "npy" / f"{cid}.npy", np.full((5, 50, 3), k, np.float32))
        rows.append({"clip_id": cid, "signer": "01", "sign_id": "0032", "npy_path": f"npy/{cid}.npy",
                     "frames": 5, "missing_ratio": ratio, "ok": "True"})
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "labels.csv").write_text("sign_id,gloss_ar,chapter\n0032,ا,letters\n", encoding="utf-8")
    out = tmp_path / "lex"
    stats = build(tmp_path / "manifest.csv", tmp_path / "labels.csv", out, {"01"}, {"03"})
    assert stats["signs"] == 1 and stats["skipped"] == 0 and stats["low_quality"] == 1
    assert np.load(out / "signs" / "0032.npy")[0, 0, 0] == 1.0     # the 0.6 clip, the least missing
    row = json.loads((out / "lexicon.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["review_status"] == "pending"    # a low-quality clip must be reviewed before it is published
