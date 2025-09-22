import asyncio
import time
from collections import defaultdict
from functools import wraps
from ..exceptions import RateLimitExceededError
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    A simple asynchronous rate limiter using a token bucket algorithm.
    Allows 'rate' calls per 'period' seconds.
    """

    _tokens = defaultdict(float)
    _last_refill = defaultdict(float)
    _lock = asyncio.Lock()  # For thread-safe token management

    def __init__(self, rate: int, period: int = 60):
        self.rate = rate
        self.period = period
        self.interval = period / rate  # Time between tokens

    async def _refill_tokens(self, key: str):
        async with self._lock:
            now = time.monotonic()
            if now > self._last_refill[key]:
                # Calculate how many tokens should have been added since last refill
                tokens_to_add = (now - self._last_refill[key]) / self.interval
                self._tokens[key] = min(self.rate, self._tokens[key] + tokens_to_add)
                self._last_refill[key] = now

    async def wait_for_token(self, key: str):
        while True:
            await self._refill_tokens(key)
            async with self._lock:
                if self._tokens[key] >= 1:
                    self._tokens[key] -= 1
                    return
            await asyncio.sleep(self.interval / 2)  # Wait a bit before checking again

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # A unique key for this rate limit, e.g., based on LLM provider
            # For simplicity, let's assume a single global rate limit for now,
            # or a key passed in kwargs.
            # A more robust solution would derive the key from the client instance or config.
            rate_limit_key = kwargs.pop("rate_limit_key", "default")
            logger.debug(f"Waiting for token for key: {rate_limit_key}")
            await self.wait_for_token(rate_limit_key)
            logger.debug(f"Token acquired for key: {rate_limit_key}")
            return await func(*args, **kwargs)

        return wrapper
