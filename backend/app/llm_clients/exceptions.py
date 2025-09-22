class LLMClientsError(Exception):
    """Base exception for LLM Clients package."""

    pass


class APIKeyError(LLMClientsError):
    """Raised when an API key is missing or invalid."""

    pass


class RateLimitExceededError(LLMClientsError):
    """Raised when a rate limit is exceeded."""

    pass


class LLMServiceError(LLMClientsError):
    """Raised for errors originating from the LLM service itself."""

    pass


class InvalidLLMProviderError(LLMClientsError):
    """Raised when an unsupported LLM provider is requested."""

    pass
