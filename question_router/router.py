"""Router: classify question then route to the appropriate model."""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from .classifier import classify
from .config import (
    DEFAULT_CLASSIFIER_MODEL,
    get_default_routing_map,
)
from .keys import load_keys_file, get_env_key
from .providers import (
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    KimiAdapter,
)


# Built-in adapter registry (provider name -> adapter instance)
_ADAPTERS = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
    "kimi": KimiAdapter(),
}


def _get_default_config() -> dict:
    """Default config for zero-arg Router(): single model for all types."""
    provider, model_id = DEFAULT_CLASSIFIER_MODEL
    return {
        "classifier_model": DEFAULT_CLASSIFIER_MODEL,
        "routing_map": get_default_routing_map(provider, model_id),
    }


def _load_config_file(path: Optional[Path | str] = None) -> Optional[dict]:
    """Load config from path or well-known locations."""
    if path is not None:
        p = Path(path) if isinstance(path, str) else path
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None
    for candidate in [
        Path.cwd() / ".question_router" / "config.json",
        Path.cwd() / "config.json",
    ]:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return None


class Router:
    """
    Route user questions to different models by type (simple, reasoning, coding, creative, other).
    Uses a cheap model to classify, then invokes the model configured for that type.
    """

    def __init__(
        self,
        classifier_model: Optional[Tuple[str, str]] = None,
        routing_map: Optional[Dict[str, Tuple[str, str]]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        key_provider: Optional[Callable[[str], Optional[str]]] = None,
    ):
        """
        :param classifier_model: (provider, model_id) for classification; default (openai, gpt-4o-mini).
        :param routing_map: Map question_type -> (provider, model_id). Default: all types -> classifier model.
        :param api_keys: Optional explicit keys: {"openai": "sk-...", ...}. Highest precedence.
        :param key_provider: Optional callable(provider) -> key; used after keys file and env (e.g. OpenClaw).
        """
        cfg = _get_default_config()
        self._classifier_model = classifier_model or cfg["classifier_model"]
        self._routing_map = routing_map or cfg["routing_map"]
        self._api_keys = dict(api_keys) if api_keys else {}
        self._key_provider = key_provider
        self._keys_file_cache: Optional[Dict[str, str]] = None

    def _get_keys_file(self) -> Dict[str, str]:
        if self._keys_file_cache is None:
            self._keys_file_cache = load_keys_file() or {}
        return self._keys_file_cache

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Resolution: explicit -> manual keys file -> env -> key_provider."""
        if provider in self._api_keys and self._api_keys[provider]:
            return self._api_keys[provider]
        from_file = self._get_keys_file().get(provider)
        if from_file:
            return from_file
        from_env = get_env_key(provider)
        if from_env:
            return from_env
        if self._key_provider:
            return self._key_provider(provider)
        return None

    def _get_adapter(self, provider: str):
        if provider in _ADAPTERS:
            return _ADAPTERS[provider]
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {list(_ADAPTERS.keys())}"
        )

    def ask(self, prompt: str) -> str:
        """
        Classify the prompt, route to the configured model for that type, return the response.
        """
        c_provider, c_model = self._classifier_model
        question_type = classify(
            prompt,
            c_provider,
            c_model,
            get_adapter=self._get_adapter,
            get_api_key=self._get_api_key,
        )
        if question_type not in self._routing_map:
            question_type = "other"
        provider, model_id = self._routing_map[question_type]
        api_key = self._get_api_key(provider)
        if not api_key:
            raise ValueError(
                f"No API key for provider '{provider}' (routed for type '{question_type}'). "
                "Set the corresponding env var, add to api_keys.json, or pass api_keys=..."
            )
        adapter = self._get_adapter(provider)
        messages = [{"role": "user", "content": prompt}]
        return adapter.invoke(messages, model_id, api_key)

    @classmethod
    def from_config(
        cls,
        path: Optional[Path] = None,
        api_keys: Optional[Dict[str, str]] = None,
        key_provider: Optional[Callable[[str], Optional[str]]] = None,
    ) -> "Router":
        """
        Build a Router from a config file. Looks for ./.question_router/config.json
        or ./config.json if path is not given.
        Config JSON can include: classifier_model (list [provider, model_id]),
        routing_map (dict type -> [provider, model_id]), and optionally api_keys.
        """
        data = _load_config_file(path)
        if not data:
            return cls(api_keys=api_keys, key_provider=key_provider)
        classifier_model = data.get("classifier_model")
        if classifier_model:
            classifier_model = tuple(classifier_model)
        routing_map = data.get("routing_map")
        if routing_map:
            routing_map = {k: tuple(v) for k, v in routing_map.items()}
        config_keys = data.get("api_keys")
        if api_keys:
            merged_keys = dict(config_keys or {})
            merged_keys.update(api_keys)
        else:
            merged_keys = config_keys
        return cls(
            classifier_model=classifier_model,
            routing_map=routing_map,
            api_keys=merged_keys,
            key_provider=key_provider,
        )
