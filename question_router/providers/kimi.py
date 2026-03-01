"""Kimi / Moonshot AI provider adapter (OpenAI-compatible API)."""

from typing import List, Dict, Any

from .base import BaseAdapter


class KimiAdapter(BaseAdapter):
    """
    Kimi (Moonshot) adapter using OpenAI-compatible API.
    Uses the openai package with base_url for Moonshot endpoint.
    """

    # Default Moonshot/Kimi API base URL (OpenAI-compatible)
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or self.DEFAULT_BASE_URL

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        api_key: str,
    ) -> str:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=4096,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()
