# question_router

Route user questions to different LLM models by **question type** to save tokens in agentic AI projects. A cheap model classifies each question; the router then calls the model you configured for that type (e.g. simple → cheap model, reasoning → stronger model).

**Plug-and-play:** one API key and `Router().ask(prompt)` is enough to start.

## Install

```bash
pip install question_router[openai]
# or add more providers:
pip install question_router[openai,anthropic,gemini,kimi]
```

## Quick start

**Minimal (one key):** Put your key in `api_keys.json` or set `OPENAI_API_KEY`, then:

```python
from question_router import Router

router = Router()
response = router.ask("What is the capital of France?")
```

**Manual API keys (recommended place):** Create `api_keys.json` in your project root:

```json
{
  "openai": "sk-...",
  "anthropic": "sk-ant-...",
  "gemini": "your-google-api-key",
  "kimi": "your-moonshot-api-key"
}
```

**Do not commit this file.** Add `api_keys.json` to `.gitignore`. The library looks for it in:

1. Project root: `./api_keys.json`
2. Config dir: `./.question_router/api_keys.json`
3. User config: `~/.config/question_router/api_keys.json`

You can also use environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY` (or `MOONSHOT_API_KEY` for Kimi).

## Custom routing

```python
from question_router import Router

router = Router(
    classifier_model=("openai", "gpt-4o-mini"),
    routing_map={
        "simple": ("openai", "gpt-4o-mini"),
        "reasoning": ("anthropic", "claude-sonnet-4"),
        "coding": ("openai", "gpt-4o"),
        "creative": ("anthropic", "claude-sonnet-4"),
        "other": ("openai", "gpt-4o-mini"),
    },
)
response = router.ask("Explain quicksort in Python.")
```

## Config file

Use `Router.from_config()` to load from `.question_router/config.json` (or `./config.json`):

```json
{
  "classifier_model": ["openai", "gpt-4o-mini"],
  "routing_map": {
    "simple": ["openai", "gpt-4o-mini"],
    "reasoning": ["anthropic", "claude-sonnet-4"],
    "coding": ["openai", "gpt-4o"],
    "creative": ["anthropic", "claude-sonnet-4"],
    "other": ["openai", "gpt-4o-mini"]
  }
}
```

## Question types

The classifier uses: `simple`, `reasoning`, `coding`, `creative`, `other`. You map each to a (provider, model) in `routing_map`.

## API key resolution order

1. Keys passed in constructor: `Router(api_keys={"openai": "sk-..."})`
2. Manual keys file: `api_keys.json` (see paths above)
3. Environment variables
4. Optional `key_provider` callable (e.g. for OpenClaw): `Router(key_provider=lambda p: get_credential(p))`

## License

MIT
