import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar
import httpx

from app.core.exceptions import (
    CircuitBreakerOpenError,
    ConflictException,
    DocumentIngestionError,
    EntityNotFoundException,
    ForbiddenException,
    QuotaExceededError,
    UnauthorizedException,
    ValidationException,
    VectorServiceError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


def is_transient_error(exc: Exception) -> bool:
    """
    Classifies whether an exception represents a retriable transient error.
    Non-transient client errors (HTTP 400, 401, 403, 404, 409, 422) return False.
    Transient server and rate-limit errors (HTTP 429, 502, 503, 504, timeouts) return True.
    """
    if isinstance(
        exc,
        (
            CircuitBreakerOpenError,
            ValidationException,
            UnauthorizedException,
            ForbiddenException,
            EntityNotFoundException,
            ConflictException,
            DocumentIngestionError,
        ),
    ):
        return False

    if isinstance(exc, (QuotaExceededError, VectorServiceError)):
        return True

    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", getattr(exc, "code", None))
    if status_code is None and hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        status_code = exc.response.status_code

    if status_code is not None:
        try:
            status_code = int(status_code)
            if status_code in (429, 502, 503, 504):
                return True
            if 400 <= status_code < 500:
                return False
        except (ValueError, TypeError):
            pass

    return False


class CircuitBreaker:
    """
    3-State Circuit Breaker (CLOSED, OPEN, HALF_OPEN) preventing cascading failures.
    Includes in-flight trial throttling during HALF_OPEN to prevent thundering herd.
    """

    def __init__(
        self,
        service_name: str = "GeminiService",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: Optional[float] = None
        self._trial_in_flight: bool = False

    def _check_state(self) -> None:
        """
        Inspects and transitions state before execution.
        Fast-fails if OPEN and recovery timeout has not elapsed.
        In HALF_OPEN, throttles concurrent probes to a single trial.
        """
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - (self.last_failure_time or 0.0)
            if elapsed >= self.recovery_timeout:
                logger.info(f"CircuitBreaker[{self.service_name}] recovery timeout elapsed. Entering HALF_OPEN.")
                self.state = CircuitState.HALF_OPEN
                self._trial_in_flight = False
            else:
                remaining = max(0.0, self.recovery_timeout - elapsed)
                raise CircuitBreakerOpenError(self.service_name, recovery_time_remaining=remaining)

        if self.state == CircuitState.HALF_OPEN:
            if self._trial_in_flight:
                raise CircuitBreakerOpenError(self.service_name, recovery_time_remaining=self.recovery_timeout)
            self._trial_in_flight = True

    def record_success(self) -> None:
        """
        Records successful execution. Resets state to CLOSED and failure counters to 0.
        """
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"CircuitBreaker[{self.service_name}] trial probe succeeded. Transitioning to CLOSED.")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._trial_in_flight = False

    def record_failure(self, exc: Exception) -> None:
        """
        Records failed execution. Filters out non-transient errors so client bugs never trip the breaker.
        Transitions to OPEN if threshold reached or if trial probe fails during HALF_OPEN.
        """
        self._trial_in_flight = False
        if not is_transient_error(exc):
            logger.debug(f"CircuitBreaker[{self.service_name}] ignoring non-transient error: {exc}")
            return

        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"CircuitBreaker[{self.service_name}] tripped to OPEN (failures={self.failure_count}, cause={exc})"
            )

    def reset(self) -> None:
        """
        Administrative reset to initial CLOSED state.
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._trial_in_flight = False

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Executes func under circuit breaker protection.
        """
        self._check_state()
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            self.record_failure(exc)
            raise
        except BaseException:
            self._trial_in_flight = False
            raise


async def execute_with_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    **kwargs: Any,
) -> T:
    """
    Executes an async callable with exponential backoff and AWS Full Jitter for transient errors.
    Aborts immediately on non-transient errors without retry.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if not is_transient_error(exc) or attempt == max_retries:
                raise
            backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_duration = random.uniform(0, backoff)
            logger.info(
                f"Transient error ({exc}). Attempt {attempt}/{max_retries}. Retrying in {sleep_duration:.2f}s"
            )
            await sleep_fn(sleep_duration)
