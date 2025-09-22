import asyncio
import logging
from functools import wraps
from typing import Callable, Tuple, Type

from ..exceptions import LLMServiceError, RateLimitExceededError

logger = logging.getLogger(__name__)


def retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (LLMServiceError, RateLimitExceededError),
):
    """
    An asynchronous retry decorator with exponential backoff.

    Args:
        attempts: Maximum number of retry attempts.
        delay: Initial delay in seconds between retries.
        backoff_factor: Factor by which the delay increases after each retry.
        exceptions: A tuple of exception types to catch and retry on.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Attempt {attempt}/{attempts} failed for {func.__name__}: {e}"
                    )
                    if attempt < attempts:
                        logger.info(f"Retrying in {current_delay:.2f} seconds...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {attempts} attempts failed for {func.__name__}."
                        )
                        raise  # Re-raise the last exception if all attempts fail

        return wrapper

    return decorator
