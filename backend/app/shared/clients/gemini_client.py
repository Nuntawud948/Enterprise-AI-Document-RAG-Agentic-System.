import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional
from app.core.config import settings
from app.core.exceptions import (
    CircuitBreakerOpenError,
    EntityNotFoundException,
    ForbiddenException,
    UnauthorizedException,
    ValidationException,
    VectorServiceError,
)
from app.shared.clients.circuit_breaker import (
    CircuitBreaker,
    execute_with_retry,
    is_transient_error,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Centralized Gateway for Google Gemini AI Engine.
    Handles Text Generation and Vector Embeddings with circuit breaker,
    exponential retry with full jitter, token guardrails, and graceful multi-tier fallback.
    """

    MAX_INPUT_TOKENS: int = 8192
    DEFAULT_SECONDARY_MODEL: str = "gemini-1.5-flash"
    LOCAL_FALLBACK_MESSAGE: str = (
        "The AI service is temporarily unavailable due to high demand. Please try again shortly."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_chat_model: Optional[str] = None,
        secondary_chat_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        chat_executor: Optional[Callable[..., Awaitable[str]]] = None,
        embedding_executor: Optional[Callable[..., Awaitable[List[float]]]] = None,
        sleep_fn: Optional[Callable[[float], Awaitable[None]]] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.default_chat_model = default_chat_model or settings.DEFAULT_CHAT_MODEL
        self.secondary_chat_model = secondary_chat_model or self.DEFAULT_SECONDARY_MODEL
        self.default_embedding_model = default_embedding_model or settings.DEFAULT_EMBEDDING_MODEL
        self.circuit_breaker = circuit_breaker or CircuitBreaker(service_name="GeminiService")
        self.chat_executor = chat_executor
        self.embedding_executor = embedding_executor
        self.sleep_fn = sleep_fn or asyncio.sleep

    def _validate_input_tokens(self, text: str, entity_name: str = "Input text") -> None:
        if not text or not text.strip():
            raise ValidationException(f"{entity_name} cannot be empty.")
        estimated_tokens = max(1, len(text) // 4)
        if estimated_tokens > self.MAX_INPUT_TOKENS:
            raise ValidationException(
                f"Input text exceeds maximum allowed token limit of {self.MAX_INPUT_TOKENS} tokens (estimated {estimated_tokens})."
            )

    async def _raw_generate_chat(
        self,
        prompt: str,
        model: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if self.chat_executor is not None:
            try:
                return await self.chat_executor(prompt=prompt, model=model, system_instruction=system_instruction)
            except TypeError:
                try:
                    return await self.chat_executor(prompt=prompt, system_instruction=system_instruction)
                except TypeError:
                    return await self.chat_executor(prompt)
        logger.info(f"Invoking Gemini Model: {model}")
        return f"Gemini response for: {prompt[:50]}..."

    async def _raw_generate_embedding(self, text: str, model: str) -> List[float]:
        if self.embedding_executor is not None:
            try:
                return await self.embedding_executor(text=text, model=model)
            except TypeError:
                try:
                    return await self.embedding_executor(text=text)
                except TypeError:
                    return await self.embedding_executor(text)
        logger.info(f"Generating embedding for text length {len(text)} chars with model {model}")
        return [0.0] * 768

    async def generate_chat_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generates text completion via multi-tier fallback:
        1. Token guardrail check (<= 8192 estimated tokens)
        2. Fast-fail to LOCAL_FALLBACK_MESSAGE if Circuit Breaker is OPEN
        3. Primary model with retry -> if fails, Secondary model with retry
        4. If all remote tiers fail, records breaker failure and returns LOCAL_FALLBACK_MESSAGE
        """
        # 1. Token guardrail & validation
        if not prompt or not prompt.strip():
            raise ValidationException("Prompt cannot be empty.")

        full_text = f"{system_instruction}\n{prompt}" if system_instruction else prompt
        estimated_tokens = max(1, len(full_text) // 4)
        if estimated_tokens > self.MAX_INPUT_TOKENS:
            raise ValidationException(
                f"Input text exceeds maximum allowed token limit of {self.MAX_INPUT_TOKENS} tokens (estimated {estimated_tokens})."
            )

        # 2. Fast-fail if Circuit Breaker is OPEN
        try:
            self.circuit_breaker._check_state()
        except CircuitBreakerOpenError:
            logger.warning(
                f"CircuitBreaker[{self.circuit_breaker.service_name}] is OPEN. Returning local fallback message."
            )
            return self.LOCAL_FALLBACK_MESSAGE

        # 3. Model Fallback Chain: Primary -> Secondary -> Local Fallback
        last_error: Optional[Exception] = None

        # Tier 1: Primary Model
        try:
            result = await execute_with_retry(
                self._raw_generate_chat,
                prompt=prompt,
                model=self.default_chat_model,
                system_instruction=system_instruction,
                max_retries=max_retries,
                sleep_fn=self.sleep_fn,
            )
            self.circuit_breaker.record_success()
            return result
        except Exception as exc:
            if not is_transient_error(exc):
                self.circuit_breaker.record_failure(exc)
                raise
            logger.warning(
                f"Primary model '{self.default_chat_model}' failed with transient error: {exc}. Attempting secondary model fallback."
            )
            last_error = exc

        # Tier 2: Secondary Model (gemini-1.5-flash)
        try:
            result = await execute_with_retry(
                self._raw_generate_chat,
                prompt=prompt,
                model=self.secondary_chat_model,
                system_instruction=system_instruction,
                max_retries=max_retries,
                sleep_fn=self.sleep_fn,
            )
            self.circuit_breaker.record_success()
            return result
        except Exception as exc:
            if not is_transient_error(exc):
                self.circuit_breaker.record_failure(exc)
                raise
            logger.error(
                f"Secondary model '{self.secondary_chat_model}' also failed with transient error: {exc}."
            )
            last_error = exc

        # Tier 3: All remote models failed -> record breaker failure and return local graceful message
        if last_error is not None:
            self.circuit_breaker.record_failure(last_error)

        return self.LOCAL_FALLBACK_MESSAGE

    async def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generates 768-dimensional vector embedding for document chunk.
        Enforces token guardrail, protected by circuit breaker and retry.
        On failure, wraps in VectorServiceError (HTTP 502).
        """
        self._validate_input_tokens(text, entity_name="Text")

        async def _do_embed():
            return await execute_with_retry(
                self._raw_generate_embedding,
                text=text,
                model=self.default_embedding_model,
                max_retries=max_retries,
                sleep_fn=self.sleep_fn,
            )

        try:
            return await self.circuit_breaker.call(_do_embed)
        except (
            CircuitBreakerOpenError,
            ValidationException,
            UnauthorizedException,
            ForbiddenException,
            EntityNotFoundException,
        ):
            raise
        except VectorServiceError:
            raise
        except Exception as exc:
            logger.error(f"Vector embedding generation failed: {exc}")
            raise VectorServiceError(f"Vector embedding service failed: {str(exc)}") from exc
