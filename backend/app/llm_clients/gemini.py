import google.generativeai as genai
from .base import LLMClient
from .exceptions import APIKeyError, LLMServiceError
from .utils.rate_limiter import RateLimiter
from .utils.retry_mechanism import retry
from .config import get_llm_config_value
from typing import Any, Dict, List, Optional
import asyncio


class GeminiClient(LLMClient):
    """Concrete implementation of LLMClient for Google Gemini."""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-pro", **kwargs
    ):
        super().__init__(api_key, **kwargs)
        if not self._api_key:
            raise APIKeyError("Gemini API key is required.")
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(model_name)
        self._generation_config = kwargs.get("generation_config", {})
        self._safety_settings = kwargs.get("safety_settings", [])

    @retry(
        attempts=get_llm_config_value("gemini", "retry_attempts", 3),
        delay=get_llm_config_value("gemini", "retry_delay", 1.0),
        backoff_factor=get_llm_config_value("gemini", "retry_backoff_factor", 2.0),
        exceptions=(LLMServiceError, RateLimitExceededError),
    )
    @RateLimiter(
        rate=get_llm_config_value("gemini", "rate_limit_rate", 10),
        period=get_llm_config_value("gemini", "rate_limit_period", 60),
    )
    async def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config=self._generation_config,
                safety_settings=self._safety_settings,
                **kwargs,
            )
            return response.text
        except Exception as e:
            raise LLMServiceError(f"Gemini text generation failed: {e}")

    @retry(
        attempts=get_llm_config_value("gemini", "retry_attempts", 3),
        delay=get_llm_config_value("gemini", "retry_delay", 1.0),
        backoff_factor=get_llm_config_value("gemini", "retry_backoff_factor", 2.0),
        exceptions=(LLMServiceError, RateLimitExceededError),
    )
    @RateLimiter(
        rate=get_llm_config_value("gemini", "rate_limit_rate", 10),
        period=get_llm_config_value("gemini", "rate_limit_period", 60),
    )
    async def generate_chat_response(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        try:
            # Gemini's chat expects roles 'user' and 'model'
            formatted_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                formatted_messages.append({"role": role, "parts": [msg["content"]]})

            chat = self._model.start_chat(history=formatted_messages[:-1])
            response = await asyncio.to_thread(
                chat.send_message,
                formatted_messages[-1]["parts"],  # Send only the last message
                generation_config=self._generation_config,
                safety_settings=self._safety_settings,
                **kwargs,
            )
            return response.text
        except Exception as e:
            raise LLMServiceError(f"Gemini chat generation failed: {e}")
