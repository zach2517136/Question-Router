"""Anthropic provider adapter."""

from typing import List, Dict, Any

from .base import BaseAdapter


def _to_anthropic_messages(messages: List[Dict[str, Any]]) -> tuple:
    """Convert generic messages to (system, anthropic_messages)."""
    system_parts = []
    anthropic_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            anthropic_messages.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_messages


class AnthropicAdapter(BaseAdapter):
    """Anthropic API adapter using the anthropic package."""

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        api_key: str,
    ) -> str:
        import anthropic
        system, anthropic_messages = _to_anthropic_messages(messages)
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": model_id,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        if response.content and len(response.content) > 0:
            block = response.content[0]
            if hasattr(block, "text"):
                return block.text.strip()
        return ""
