"""Google Gemini provider adapter."""

from typing import List, Dict, Any

from .base import BaseAdapter


def _to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert generic messages to Gemini role/content parts."""
    contents = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Gemini uses "user" and "model" (not "assistant")
        gemini_role = "user" if role == "user" else "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    return contents


class GeminiAdapter(BaseAdapter):
    """Google Gemini API adapter using google-generativeai."""

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        api_key: str,
    ) -> str:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        # Build prompt: system as first user message if present, then alternate
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            else:
                parts.append(content)
        prompt = "\n\n".join(parts)
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        return ""
