"""Where the pipeline reads its data: the project's data/ folder, or ISHARATI_DATA_DIR (the Hugging Face Space downloads
the datasets there at start-up, see src/isharati/hub.py)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ISHARATI_DATA_DIR", ROOT / "data"))
