"""Load API keys from well-known paths (manual keys file)."""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Env var names per provider
ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "kimi": "KIMI_API_KEY",  # or MOONSHOT_API_KEY
}

# Additional env name for Kimi (Moonshot)
ENV_KEYS_ALT = {"kimi": "MOONSHOT_API_KEY"}


def _keys_file_paths() -> list:
    """Return paths to check for api_keys.json (project root, .question_router, then config dir)."""
    cwd = Path.cwd()
    home = Path.home()
    return [
        cwd / "api_keys.json",
        cwd / ".question_router" / "api_keys.json",
        home / ".config" / "question_router" / "api_keys.json",
    ]


def load_keys_file() -> Optional[Dict[str, str]]:
    """
    Load API keys from the first existing well-known path.
    Path order: ./api_keys.json, ./.question_router/api_keys.json, ~/.config/question_router/api_keys.json.
    Returns None if no file exists; returns dict of provider -> key (only non-empty values).
    """
    for path in _keys_file_paths():
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {k: str(v).strip() for k, v in data.items() if v}
            except (json.JSONDecodeError, OSError):
                pass
    return None


def get_env_key(provider: str) -> Optional[str]:
    """Get API key for provider from environment. Tries primary and alternate env vars."""
    env_name = ENV_KEYS.get(provider)
    if env_name:
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
    alt = ENV_KEYS_ALT.get(provider)
    if alt:
        key = os.environ.get(alt, "").strip()
        if key:
            return key
    return None
