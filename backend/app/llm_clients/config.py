import os
from typing import Dict, Any, Optional


class LLMConfig:
    _instance = None
    _config: Dict[str, Any] = {
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model_name": "gemini-pro",
            "rate_limit_rate": 10,
            "rate_limit_period": 60,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "retry_backoff_factor": 2.0,
        },
        "ollama": {
            "api_key": None,  # Ollama typically doesn't use an API key in the same way
            "model_name": "llama2",
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "rate_limit_rate": 20,
            "rate_limit_period": 60,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "retry_backoff_factor": 2.0,
        },
        # Add other LLMs here
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
        return cls._instance

    def set_api_key(self, provider: str, api_key: str):
        """Sets the API key for a specific LLM provider."""
        if provider in self._config:
            self._config[provider]["api_key"] = api_key
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieves the API key for a specific LLM provider."""
        return self._config.get(provider, {}).get("api_key")

    def set_llm_settings(self, provider: str, settings: Dict[str, Any]):
        """Sets multiple settings for a specific LLM provider."""
        if provider in self._config:
            # Only update known settings to prevent arbitrary config injection
            for key, value in settings.items():
                if key in self._config[provider]:  # Ensure key exists in default config
                    self._config[provider][key] = value
                else:
                    print(
                        f"Warning: Attempted to set unknown setting '{key}' for provider '{provider}'. Ignoring."
                    )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def get_llm_config(
        self, provider: str, setting_name: str, default: Any = None
    ) -> Any:
        """Retrieves a specific setting for an LLM provider."""
        return self._config.get(provider, {}).get(setting_name, default)


# Global instance for easy access
llm_config = LLMConfig()


# Helper function for external access
def get_llm_config_value(provider: str, setting_name: str, default: Any = None) -> Any:
    return llm_config.get_llm_config(provider, setting_name, default)


def set_llm_api_key(provider: str, api_key: str):
    llm_config.set_api_key(provider, api_key)


def set_llm_provider_settings(provider: str, settings: Dict[str, Any]):
    llm_config.set_llm_settings(provider, settings)
