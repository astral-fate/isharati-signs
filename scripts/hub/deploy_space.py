"""Deploy the application to a private Hugging Face Docker Space.

Stages deploy/space (Dockerfile, README, requirements) and the isharati package (src/isharati, including the web
app's page, avatar and outfit assets), creates the Space if needed, uploads, and sets its settings: secrets HF_TOKEN (the
logged-in token, read access to the private dataset) and OPENROUTER_API_KEY (from the environment or D:/islam/.env),
variable ISHARATI_HUB_DATASET. Secret values are never printed.

  env -u HF_TOKEN .venv/Scripts/python scripts/hub/deploy_space.py [--space NAME] [--dataset NAME] [--update] [--dry-run]

An existing Space is never overwritten unless --update is given.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def stage() -> Path:
    out = Path(tempfile.mkdtemp(prefix="isharati_space_"))
    shutil.copy2(ROOT / ".env.example", out / ".env.example")  # the settings and where to get each key, for local runs
    for f in (ROOT / "deploy" / "space").iterdir():
        shutil.copy2(f, out / f.name)
    # only files tracked in git: a local copy of data (src/isharati/academy_data, app/static/data) is git-ignored
    # because it belongs in the datasets, and must never reach the Space, which may be public
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "src/isharati"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout.splitlines()
    for rel in tracked:
        if (ROOT / rel).is_file():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, out / rel)
    # the React landing page, built here (cd web && npm run build): deploy/space/Dockerfile copies web/dist as it is
    if not (ROOT / "web" / "dist" / "index.html").exists():
        raise SystemExit("web/dist is missing: run `cd web && npm run build` first")
    shutil.copytree(ROOT / "web" / "dist", out / "web" / "dist", ignore=shutil.ignore_patterns("_*probe*.json"))
    return out


def main():
    import os
    from huggingface_hub import HfApi
    from isharati.llm import env_key
    # Read fresh token directly from .env, bypassing any stale system-level HF_TOKEN
    env_file = ROOT.parent / ".env"
    token = None
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not token:
        token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    os.environ.pop("HF_TOKEN", None)
    api = HfApi(token=token)
    user = api.whoami(token=token)["name"]
    space = f"{user}/{arg('--space', 'isharati-app')}"
    dataset = f"{user}/{arg('--dataset', 'isharati-data')}"
    folder = stage()
    n = sum(1 for f in folder.rglob("*") if f.is_file())
    print(f"staged {n} files in {folder}")
    if "--dry-run" in sys.argv:
        return
    import atexit  # the staged copy (~70 MB) is removed however the upload ends; left behind, they filled C:
    atexit.register(shutil.rmtree, folder, True)
    if api.repo_exists(space, repo_type="space") and "--update" not in sys.argv:
        sdk = api.space_info(space).sdk
        if sdk != "docker" or "--update" not in sys.argv:
            raise SystemExit(f"{space} already exists (sdk: {sdk}); pass --update to redeploy it, or --space <new name>")
    api.create_repo(space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.add_space_secret(space, "HF_TOKEN", token)
    # OpenRouter answers and glosses; Groq transcribes speech (Whisper) in the audio / video mode; NVIDIA is a fallback
    # DATABASE_URL: the Neon (PostgreSQL) database of the review dashboard (/review)
    for name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "DATABASE_URL"):
        key = env_key(name)
        if key:
            api.add_space_secret(space, name, key)
        else:
            print(f"{name} not found: add it in the Space settings")
    api.add_space_variable(space, "ISHARATI_HUB_DATASET", dataset)
    # files the app no longer has are removed from the Space (an upload alone only adds and replaces)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(folder), commit_message="deploy Isharati",
                      delete_patterns=["src/**", "web/**", "isharati/**", "scripts/**"])
    print("deployed (private):", f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
