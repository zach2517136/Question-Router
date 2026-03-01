"""OpenAI provider adapter."""

from typing import List, Dict, Any

from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """OpenAI API adapter using the openai package."""

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        api_key: str,
    ) -> str:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=4096,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()
