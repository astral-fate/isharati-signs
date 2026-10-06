"""Dense embeddings of the approved passages, for hybrid retrieval (BM25 + meaning) in isharati.retrieval.engine.

One multilingual model, multilingual-e5-large, for Arabic, English, Turkish and Urdu: a question finds «كسوف» passages when
it says «الخسوف», and paraphrases match without sharing words. Runs in the GPU environment (models/.venv).

  D:\\islam\\models\\.venv\\Scripts\\python scripts/corpora/embeddings.py index ar|en|tr|ur
      -> data/<lang>/embeddings.npy (float16, unit length, one row per corpus line) + embeddings_ids.json
  D:\\islam\\models\\.venv\\Scripts\\python scripts/corpora/embeddings.py serve
      -> POST http://127.0.0.1:8766/embed {"texts": [...]} -> {"vectors": [[...], ...]}
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
MODEL = "intfloat/multilingual-e5-large"  # safetensors weights; bge-m3 needs torch >= 2.6 to load
PORT = 8766
MAX_CHARS = 1500  # the title and the start of a hadith carry its meaning; long explanations are cut


def corpus_path(lang):
    return DATA / ("asl" if lang == "en" else lang) / "corpus.jsonl"


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        m.half()
    m.max_seq_length = 512
    return m


def index(lang):
    rows = [json.loads(line) for line in corpus_path(lang).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = ["passage: " + r["text"][:MAX_CHARS] for r in rows]  # e5 marks passages and queries
    model = load_model()
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch: less padding, faster
    vecs = np.zeros((len(texts), model.get_sentence_embedding_dimension()), np.float16)
    batch = 32
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        vecs[idx] = model.encode([texts[i] for i in idx], normalize_embeddings=True, convert_to_numpy=True)
        if k % (batch * 40) == 0:
            print(f"{lang}: {k}/{len(texts)}", flush=True)
    out = corpus_path(lang).parent
    np.save(out / "embeddings.npy", vecs)
    (out / "embeddings_ids.json").write_text(json.dumps([r["pid"] for r in rows]), encoding="utf-8")
    print(f"{lang}: {len(texts)} passages embedded -> {out / 'embeddings.npy'}", flush=True)


def serve():
    model = load_model()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            texts = ["query: " + str(t)[:MAX_CHARS] for t in body.get("texts", [])][:64]
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True) if texts else np.zeros((0, 0))
            data = json.dumps({"vectors": vecs.astype(float).round(5).tolist()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"embedding service on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    {"index": lambda: index(sys.argv[2]), "serve": serve}[sys.argv[1]]()
