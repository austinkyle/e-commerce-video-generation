from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_config() -> dict[str, str | Path | None]:
    """Load local settings without requiring credentials for local commands."""
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "kie_api_key": os.getenv("KIE_API_KEY"),
        "data_dir": Path(os.getenv("APP_DATA_DIR", "data")).expanduser(),
    }


def require_key(name: str) -> str:
    load_dotenv()
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"{name} is required. Add it to your local .env file.")
    return value.strip()
