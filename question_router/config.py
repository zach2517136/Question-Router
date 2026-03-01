"""Default question types, classifier prompt, and routing config."""

from typing import Dict, Tuple

# Default question types used by the classifier
DEFAULT_QUESTION_TYPES = (
    "simple",   # factual, short-answer, lookup
    "reasoning",  # multi-step, math, logic
    "coding",   # code generation or explanation
    "creative", # writing, brainstorming, open-ended
    "other",    # fallback
)

CLASSIFIER_SYSTEM_PROMPT = """You are a classifier. Given a user question, reply with exactly one word: the question type.

Question types (reply with only one of these words, lowercase):
- simple: factual, short-answer, lookup, definition
- reasoning: multi-step reasoning, math, logic, analysis
- coding: code generation, code explanation, debugging, programming
- creative: writing, brainstorming, open-ended, creative writing
- other: anything that does not clearly fit above

Reply with only the single word, nothing else."""

# Default classifier model (provider, model_id)
DEFAULT_CLASSIFIER_MODEL: Tuple[str, str] = ("openai", "gpt-4o-mini")

# Default routing: all types go to same model (plug-and-play with one key)
def get_default_routing_map(provider: str = "openai", model_id: str = "gpt-4o-mini") -> Dict[str, Tuple[str, str]]:
    """Return a routing map that sends every type to the same model."""
    return {qtype: (provider, model_id) for qtype in DEFAULT_QUESTION_TYPES}
