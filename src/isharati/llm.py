"""LLM clients with the same .chat(messages) -> str interface as isharati.llm.GroqClient.

NVIDIA NIM serves Llama-family models over an OpenAI-compatible endpoint. The key is read from NVIDIA_API_KEY
(the environment, or D:/islam/.env)."""
import os
import re
from pathlib import Path

import requests


class GlossError(Exception):
    """A glossing backend failed or returned unusable output."""


def require_text(text) -> str:
    if text is None or not str(text).strip():
        raise ValueError("empty text")
    return str(text)



class GroqClient:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b", timeout: int = 60):
        self.key = api_key or os.environ.get("GROQ_API_KEY")
        self.model, self.timeout = model, timeout

    def chat(self, messages) -> str:
        if not self.key:
            raise GlossError("GROQ_API_KEY not set")
        r = requests.post(self.URL, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
                          json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 1500})
        if r.status_code != 200:
            raise GlossError(f"groq {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"].get("content") or ""



from isharati.config import ROOT  # noqa: E402  (importing config also loads .env)
ENV = ROOT / ".env"


def env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


LEAKED_REASONING = re.compile(r"\s*(we need to|we have to|let's|let me|the user (?:asks|wants)|okay,? so|first,? i)", re.I)


class NvidiaClient:
    URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    name = "nvidia"

    def __init__(self, model: str = "nvidia/nemotron-3-super-120b-a12b", api_key: str | None = None,
                 timeout: int = 120):
        self.key = api_key or env_key("NVIDIA_API_KEY")
        self.model, self.timeout = model, timeout

    def chat(self, messages) -> str:
        if not self.key:
            raise GlossError("NVIDIA_API_KEY not set")
        r = requests.post(self.URL, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
                          json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 1500,
                                # Nemotron 3 reasons by default and can return the reasoning as the answer
                                "chat_template_kwargs": {"enable_thinking": False}})
        if r.status_code != 200:
            raise GlossError(f"nvidia {r.status_code}: {r.text[:200]}")
        content = r.json()["choices"][0]["message"].get("content") or ""
        content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()  # reasoning models may inline their thoughts
        if not content or LEAKED_REASONING.match(content):
            raise GlossError(f"nvidia {self.model}: empty reply or reasoning instead of an answer")
        return content


class FallbackClient:
    """First client that answers; the next one if it fails (e.g. NVIDIA, then Groq)."""

    def __init__(self, clients):
        self.clients, self.last = clients, None

    def chat(self, messages) -> str:
        errors = []
        for c in self.clients:
            try:
                reply = c.chat(messages)
                self.last = getattr(c, "model", type(c).__name__)  # which backend answered, for reports
                return reply
            except Exception as e:
                errors.append(f"{type(c).__name__}: {e}")
        raise GlossError("all LLM backends failed: " + " | ".join(errors))


class ReasoningGroqClient(GroqClient):
    """Groq's gpt-oss is a reasoning model: with a small token budget it can spend it all thinking and return an
    empty reply. More room and low reasoning effort (the Arabic app's GroqClient is left unchanged)."""

    def chat(self, messages) -> str:
        if not self.key:
            raise GlossError("GROQ_API_KEY not set")
        r = requests.post(self.URL, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
                          json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 4000,
                                "reasoning_effort": "low"})
        if r.status_code != 200:
            raise GlossError(f"groq {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"].get("content") or ""


NVIDIA_MODELS = (  # answered in a live check (2026-09-30); many listed NIM models are retired or overloaded
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-20b",
)


class OpenRouterClient:
    """OpenRouter: many models behind one OpenAI-compatible API and one key (OPENROUTER_API_KEY). Model ids ending in
    ':free' cost nothing; a free-tier account (no credits bought) cannot be charged for any model."""
    URL = "https://openrouter.ai/api/v1/chat/completions"
    name = "openrouter"

    def __init__(self, model: str, api_key: str | None = None, timeout: int = 25):
        self.key = api_key or env_key("OPENROUTER_API_KEY")
        self.model, self.timeout = model, timeout

    def chat(self, messages) -> str:
        if not self.key:
            raise GlossError("OPENROUTER_API_KEY not set")
        r = requests.post(self.URL, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}", "X-Title": "Isharati"},
                          json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 1500,
                                "reasoning": {"enabled": False}})   # short structured answers, no thinking
        if r.status_code != 200:
            raise GlossError(f"openrouter {self.model} {r.status_code}: {r.text[:200]}")
        content = (r.json()["choices"][0]["message"].get("content") or "")
        content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
        if not content or LEAKED_REASONING.match(content):
            raise GlossError(f"openrouter {self.model}: empty reply or reasoning instead of an answer")
        return content


OPENROUTER_MODELS = (  # tried in order. Override with ISHARATI_MODELS="id1,id2"
    "qwen/qwen3.8-27b",                       # best Arabic glossing in a check on held-out ArabSign sentences (paid: ~3-6 s;
                                              # its :free variant was withdrawn, a 404 on every request)
    "qwen/qwen3.8-flash",                     # paid, as fast
    "nvidia/nemotron-3-ultra-550b-a55b:free",  # fallbacks when the first is rate-limited
    "nvidia/nemotron-3-super-120b-a12b:free",
)


def default_client():
    """ISHARATI_LLM=openrouter (the Space): OpenRouter models in order. Otherwise NVIDIA NIM, then Groq (development and
    the evaluation runs reported in the paper)."""
    if os.environ.get("ISHARATI_LLM") == "openrouter":
        models = [m.strip() for m in os.environ.get("ISHARATI_MODELS", ",".join(OPENROUTER_MODELS)).split(",") if m.strip()]
        # the free OpenRouter quota is per account per day; NIM and Groq take over when their keys are set
        backup = ([NvidiaClient(model=m, timeout=25) for m in NVIDIA_MODELS[:2]] if env_key("NVIDIA_API_KEY") else []) \
            + ([ReasoningGroqClient(api_key=env_key("GROQ_API_KEY"))] if env_key("GROQ_API_KEY") else [])
        return FallbackClient([OpenRouterClient(m) for m in models] + backup)
    # short timeouts: the free NIM endpoints stall or return 503 under load, and Groq answers in seconds
    return FallbackClient([NvidiaClient(model=m, timeout=25) for m in NVIDIA_MODELS[:2]]
                          + [ReasoningGroqClient(api_key=env_key("GROQ_API_KEY"))])
