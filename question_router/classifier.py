"""Classify user questions into types using a cheap model."""

from typing import Dict, Tuple

from .config import DEFAULT_QUESTION_TYPES, CLASSIFIER_SYSTEM_PROMPT


def normalize_label(raw: str) -> str:
    """Normalize classifier output to a known question type."""
    label = (raw or "").strip().lower()
    # Take first word if model replied with extra text
    if " " in label:
        label = label.split()[0]
    if label in DEFAULT_QUESTION_TYPES:
        return label
    return "other"


def classify(
    question: str,
    provider: str,
    model_id: str,
    get_adapter,
    get_api_key,
) -> str:
    """
    Classify a user question into one of the default types using the given model.

    :param question: The user's question text.
    :param provider: Provider name (e.g. "openai", "anthropic").
    :param model_id: Model to use for classification (e.g. "gpt-4o-mini").
    :param get_adapter: Callable(provider_name) -> BaseAdapter.
    :param get_api_key: Callable(provider_name) -> str or None.
    :return: One of DEFAULT_QUESTION_TYPES (e.g. "simple", "reasoning", "other").
    """
    adapter = get_adapter(provider)
    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(
            f"No API key for provider '{provider}' (used for classification). "
            "Set the corresponding env var, add to api_keys.json, or pass api_keys=..."
        )
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    raw = adapter.invoke(messages, model_id, api_key)
    return normalize_label(raw)
