import ollama
from .base import LLMClient
from .exceptions import LLMServiceError, RateLimitExceededError
from .utils.rate_limiter import RateLimiter
from .utils.retry_mechanism import retry
from .config import get_llm_config_value
from typing import Any, Dict, List, Optional
import asyncio


class OllamaClient(LLMClient):
    """Concrete implementation of LLMClient for Ollama."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama2",
        host: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(
            api_key, **kwargs
        )  # API key might not be directly used by Ollama, but kept for consistency
        self._model_name = model_name
        self._host = host
        self._client = ollama.Client(host=self._host)  # Ollama client instance

    @retry(
        attempts=get_llm_config_value("ollama", "retry_attempts", 3),
        delay=get_llm_config_value("ollama", "retry_delay", 1.0),
        backoff_factor=get_llm_config_value("ollama", "retry_backoff_factor", 2.0),
        exceptions=(LLMServiceError, RateLimitExceededError),
    )
    @RateLimiter(
        rate=get_llm_config_value("ollama", "rate_limit_rate", 20),
        period=get_llm_config_value("ollama", "rate_limit_period", 60),
    )
    async def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                self._client.generate, model=self._model_name, prompt=prompt, **kwargs
            )
            return response["response"]
        except Exception as e:
            raise LLMServiceError(f"Ollama text generation failed: {e}")

    @retry(
        attempts=get_llm_config_value("ollama", "retry_attempts", 3),
        delay=get_llm_config_value("ollama", "retry_delay", 1.0),
        backoff_factor=get_llm_config_value("ollama", "retry_backoff_factor", 2.0),
        exceptions=(LLMServiceError, RateLimitExceededError),
    )
    @RateLimiter(
        rate=get_llm_config_value("ollama", "rate_limit_rate", 20),
        period=get_llm_config_value("ollama", "rate_limit_period", 60),
    )
    async def generate_chat_response(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        try:
            response = await asyncio.to_thread(
                self._client.chat, model=self._model_name, messages=messages, **kwargs
            )
            return response["message"]["content"]
        except Exception as e:
            raise LLMServiceError(f"Ollama chat generation failed: {e}")
