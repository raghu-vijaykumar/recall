from typing import Dict, Type
from .base import LLMClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .exceptions import InvalidLLMProviderError
from .config import llm_config  # Import the global config instance


class LLMClientFactory:
    _instance = None
    _clients: Dict[str, LLMClient] = {}
    _client_map: Dict[str, Type[LLMClient]] = {
        "gemini": GeminiClient,
        "ollama": OllamaClient,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClientFactory, cls).__new__(cls)
        return cls._instance

    def get_client(self, provider: str) -> LLMClient:
        """
        Returns a singleton instance of the LLM client for the given provider.
        Initializes the client if it doesn't already exist.
        """
        if provider not in self._client_map:
            raise InvalidLLMProviderError(f"Unsupported LLM provider: {provider}")

        if provider not in self._clients:
            client_class = self._client_map[provider]
            api_key = llm_config.get_api_key(provider)
            # Pass all relevant config settings to the client
            provider_settings = llm_config._config.get(provider, {})
            # Remove api_key from provider_settings to avoid duplicate argument
            provider_settings = {
                k: v for k, v in provider_settings.items() if k != "api_key"
            }
            self._clients[provider] = client_class(api_key=api_key, **provider_settings)
        return self._clients[provider]


# Global instance for easy access
llm_client_factory = LLMClientFactory()
