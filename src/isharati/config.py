"""Where the pipeline reads its data: the project's data/ folder, or ISHARATI_DATA_DIR (the Hugging Face Space downloads
the datasets there at start-up, see src/isharati/hub.py).

Keys and settings come from the environment. A `.env` file at the project root (copy `.env.example`) is read once on
import, without overriding anything already set: the Space sets its secrets as environment variables and has no .env.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(*paths: Path) -> None:
    """KEY=value lines of the first existing file into os.environ, unless already set; # comments and blanks skipped."""
    for path in paths:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip().strip("\"'")  # a trailing comment is not the value
                if key.strip() and value:
                    os.environ.setdefault(key.strip(), value)
            return


load_env(ROOT / ".env", ROOT.parent / ".env")  # the project's own, else one shared with sibling projects
DATA = Path(os.environ.get("ISHARATI_DATA_DIR", ROOT / "data"))
