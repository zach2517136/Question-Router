"""
question_router: Route questions to different LLM models by type to save tokens.

Plug-and-play usage:
    from question_router import Router
    router = Router()  # keys from api_keys.json or env
    response = router.ask("What is 2+2?")

To add API keys manually, create `api_keys.json` in your project root with:
    {"openai": "sk-...", "anthropic": "sk-ant-...", "gemini": "...", "kimi": "..."}
Do not commit this file (add api_keys.json to .gitignore).
"""

from .router import Router

__all__ = ["Router"]
__version__ = "1.0.0"
