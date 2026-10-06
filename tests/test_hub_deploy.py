"""Deployment plumbing: model client selection, hub data restore (offline, with a fake snapshot)."""
import json
import shutil

import numpy as np


def test_openrouter_chain_selected(monkeypatch):
    from isharati import llm
    monkeypatch.setenv("ISHARATI_LLM", "openrouter")
    monkeypatch.setenv("ISHARATI_MODELS", "a/model-1:free, b/model-2:free")
    monkeypatch.setattr(llm, "env_key", lambda name: None)  # no NIM / Groq keys: OpenRouter only
    c = llm.default_client()
    assert [x.model for x in c.clients] == ["a/model-1:free", "b/model-2:free"]
    assert all(isinstance(x, llm.OpenRouterClient) for x in c.clients)


def test_openrouter_chain_falls_back_to_nim_and_groq_when_keys_are_set(monkeypatch):
    from isharati import llm
    monkeypatch.setenv("ISHARATI_LLM", "openrouter")
    monkeypatch.setenv("ISHARATI_MODELS", "a/model-1:free")
    monkeypatch.setattr(llm, "env_key", lambda name: "k")
    names = [type(x).__name__ for x in llm.default_client().clients]
    assert names == ["OpenRouterClient", "NvidiaClient", "NvidiaClient", "ReasoningGroqClient"]


def test_quota_exhausted_is_a_short_503(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from isharati.app import server
    monkeypatch.setattr(server, "OUT", tmp_path)
    def boom(lang):
        raise RuntimeError('openrouter 429: {"message":"Rate limit exceeded: free-models-per-day"}')
    monkeypatch.setattr(server, "pipeline", boom)
    r = TestClient(server.app).post("/api/ask", json={"question": "Who was Musa?", "lang": "en"})
    assert r.status_code == 503 and "quota" in r.json()["detail"]


def test_default_chain_unchanged(monkeypatch):
    from isharati import llm
    monkeypatch.delenv("ISHARATI_LLM", raising=False)
    names = [type(x).__name__ for x in llm.default_client().clients]
    assert names[:2] == ["NvidiaClient", "NvidiaClient"] and names[-1] == "ReasoningGroqClient"


def test_openrouter_rejects_leaked_reasoning(monkeypatch):
    from isharati import llm
    from isharati.llm import GlossError

    class R:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "We need to answer the user..."}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    c = llm.OpenRouterClient("x:free", api_key="k")
    try:
        c.chat([{"role": "user", "content": "hi"}])
        assert False, "reasoning should be rejected"
    except GlossError:
        pass


def test_hub_restores_layout(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    (snap / "en").mkdir(parents=True)
    (snap / "en" / "corpus.jsonl").write_text('{"pid": "q1:1"}\n', encoding="utf-8")
    np.savez(snap / "en" / "poses.npz", **{"asl/lexicon/signs/x.npy": np.ones((3, 50, 3), np.float32)})
    (snap / "layout.json").write_text(json.dumps({"en/corpus.jsonl": "asl/corpus.jsonl"}), encoding="utf-8")
    data = tmp_path / "data"
    from isharati import hub
    monkeypatch.setattr(hub, "DATA", data)
    monkeypatch.setenv("ISHARATI_HUB_DATASET", "u/isharati-data")
    monkeypatch.setenv("ISHARATI_LANGS", "en")
    import huggingface_hub

    class Info:
        sha = "abc123"
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda self, repo: Info())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    hub.ensure_data()
    assert (data / "asl" / "corpus.jsonl").read_text(encoding="utf-8").startswith('{"pid"')
    assert np.load(data / "asl" / "lexicon" / "signs" / "x.npy").shape == (3, 50, 3)
    assert any(p.name.startswith(".hub_abc123") for p in data.iterdir())


def test_stage_includes_the_built_web_app():
    """deploy/space/Dockerfile copies the page built here (web/dist), so the stage carries it, without the sources."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("deploy_space", root / "scripts" / "hub" / "deploy_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.stage()
    try:
        assert (out / "web" / "dist" / "index.html").exists() and (out / "web" / "dist" / "record.html").exists()
        assert not (out / "web" / "node_modules").exists() and not (out / "web" / "src").exists()
        assert not list((out / "web" / "dist").rglob("_*probe*.json"))  # retarget/face probes never ship
        for f in ("avatar.js", "outfits.js", "stats.json"):
            assert (out / "src" / "isharati" / "app" / "static" / f).exists(), f
        docker = (out / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY --chown=user web/dist" in docker and "ISHARATI_LANGS=en,ar,tr,ur" in docker
        assert "ISHARATI_URDU_WSLP=1" in docker
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_faces_pack_and_unpack_to_the_files_the_app_reads(tmp_path, monkeypatch):
    """publish.faces_of -> faces.npz -> hub._unpack_faces gives back each sign's blendshapes and contour points."""
    import importlib.util
    import io
    from pathlib import Path
    from isharati import hub
    from isharati.pose import face
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("publish", root / "scripts" / "hub" / "publish.py")
    publish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish)
    src = tmp_path / "src"
    (src / "face" / "ar").mkdir(parents=True)
    blend, mesh = np.random.rand(5, 52).astype(np.float16), np.random.rand(5, 478, 3).astype(np.float16)
    names = np.array([f"b{i}" for i in range(52)])
    from isharati.pose import proportions
    np.savez(src / "face" / "ar" / "s1.npz", blend=blend, face=mesh, blend_names=names, ratio=proportions.REF)
    monkeypatch.setattr(publish, "DATA", src)
    buf = io.BytesIO()
    np.savez_compressed(buf, **publish.faces_of("face/ar"))
    (tmp_path / "faces.npz").write_bytes(buf.getvalue())
    monkeypatch.setattr(hub, "DATA", tmp_path / "data")
    hub._unpack_faces(tmp_path / "faces.npz", "ar")
    monkeypatch.setattr(face, "FACE", tmp_path / "data" / "face")
    face.load.cache_clear()
    b, pts, n = face.load("ar", "s1")
    face.load.cache_clear()
    assert np.array_equal(b, blend.astype(np.float32)) and n == list(names)
    assert np.allclose(pts, mesh[:, face.POINTS].astype(np.float32))  # ratio = REF: heights unchanged
