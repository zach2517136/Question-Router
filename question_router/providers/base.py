"""Base adapter for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAdapter(ABC):
    """Abstract adapter: invoke a model with messages and return response text."""

    @abstractmethod
    def invoke(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        api_key: str,
    ) -> str:
        """
        Send messages to the model and return the assistant reply as a string.

        :param messages: List of dicts with "role" ("system"|"user"|"assistant") and "content".
        :param model_id: Provider-specific model identifier (e.g. "gpt-4o-mini").
        :param api_key: API key for this provider.
        :return: The assistant's reply text.
        """
        pass
