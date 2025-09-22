import abc
from typing import Any, Dict, List, Optional


class LLMClient(abc.ABC):
    """Abstract Base Class for all LLM clients."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key

    @abc.abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text based on a given prompt."""
        pass

    @abc.abstractmethod
    async def generate_chat_response(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generates a chat response based on a list of messages."""
        pass

    # Potentially other common methods like embedding generation, etc.
