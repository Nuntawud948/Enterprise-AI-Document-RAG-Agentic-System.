import asyncio
import time
from unittest.mock import AsyncMock
import httpx
import pytest

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
from app.shared.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    execute_with_retry,
    is_transient_error,
)
from app.shared.clients.gemini_client import GeminiClient


# ============================================================================
# 1. Circuit Breaker State Machine & Transition Tests
# ============================================================================

def test_circuit_breaker_initial_state_is_closed():
    """Verify initial state of CircuitBreaker is CLOSED with zero failures."""
    cb = CircuitBreaker(service_name="TestGemini", failure_threshold=3, recovery_timeout=30.0)
    assert cb.service_name == "TestGemini"
    assert cb.failure_threshold == 3
    assert cb.recovery_timeout == 30.0
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.last_failure_time is None
    assert cb._trial_in_flight is False


def test_circuit_breaker_success_resets_failure_count():
    """Verify record_success clears accumulated failure count and resets state to CLOSED."""
    cb = CircuitBreaker(failure_threshold=3)
    cb.failure_count = 2
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED
    assert cb._trial_in_flight is False


def test_circuit_breaker_trips_to_open_after_threshold_failures():
    """Verify circuit trips to OPEN only when consecutive transient failures reach threshold."""
    cb = CircuitBreaker(failure_threshold=3)
    exc = QuotaExceededError("Rate limit 429")

    cb.record_failure(exc)
    assert cb.failure_count == 1
    assert cb.state == CircuitState.CLOSED

    cb.record_failure(exc)
    assert cb.failure_count == 2
    assert cb.state == CircuitState.CLOSED

    cb.record_failure(exc)
    assert cb.failure_count == 3
    assert cb.state == CircuitState.OPEN
    assert cb.last_failure_time is not None


@pytest.mark.asyncio
async def test_circuit_breaker_open_fast_fails_without_calling_func():
    """Verify OPEN state raises CircuitBreakerOpenError immediately without invoking downstream func."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
    cb.record_failure(QuotaExceededError("429"))
    assert cb.state == CircuitState.OPEN

    mock_func = AsyncMock()
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await cb.call(mock_func)

    mock_func.assert_not_called()
    assert exc_info.value.service_name == "GeminiService"
    assert exc_info.value.recovery_time_remaining is not None
    assert 0 < exc_info.value.recovery_time_remaining <= 30.0


def test_circuit_breaker_transitions_to_half_open_after_timeout():
    """Verify OPEN transitions to HALF_OPEN after recovery_timeout elapses."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
    cb.record_failure(QuotaExceededError("429"))
    assert cb.state == CircuitState.OPEN

    # Fast forward past timeout
    cb.last_failure_time = time.time() - 15.0
    cb._check_state()

    assert cb.state == CircuitState.HALF_OPEN
    assert cb._trial_in_flight is True


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_trial_success_resets_to_closed():
    """Verify successful trial during HALF_OPEN resets breaker to CLOSED and clears failure count."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    cb.state = CircuitState.HALF_OPEN
    cb.failure_count = 3

    mock_func = AsyncMock(return_value="trial_success")
    res = await cb.call(mock_func)

    assert res == "trial_success"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.last_failure_time is None
    assert cb._trial_in_flight is False


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_trial_failure_trips_back_to_open():
    """Verify failed trial during HALF_OPEN immediately trips breaker back to OPEN."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    cb.state = CircuitState.HALF_OPEN

    mock_func = AsyncMock(side_effect=QuotaExceededError("Still degraded"))
    with pytest.raises(QuotaExceededError):
        await cb.call(mock_func)

    assert cb.state == CircuitState.OPEN
    assert cb._trial_in_flight is False
    assert cb.last_failure_time is not None


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_throttles_concurrent_trials():
    """Verify HALF_OPEN throttles concurrent requests to a single trial probe and fast-fails others."""
    cb = CircuitBreaker(service_name="ThrottledService", recovery_timeout=10.0)
    cb.state = CircuitState.HALF_OPEN

    in_trial = asyncio.Event()
    release_trial = asyncio.Event()

    async def long_trial():
        in_trial.set()
        await release_trial.wait()
        return "trial_ok"

    task1 = asyncio.create_task(cb.call(long_trial))
    await in_trial.wait()

    # Second concurrent call while task1 is probing must fast-fail
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await cb.call(AsyncMock())
    assert exc_info.value.service_name == "ThrottledService"

    release_trial.set()
    res1 = await task1
    assert res1 == "trial_ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_non_transient_error_does_not_trip_breaker():
    """
    Verify client validation errors (HTTP 400) do NOT increment failure count
    or trip the circuit breaker, preventing DoS vulnerabilities.
    """
    cb = CircuitBreaker(failure_threshold=2)

    async def bad_client_call():
        raise ValidationException("Field 'title' is required.")

    for _ in range(5):
        with pytest.raises(ValidationException):
            await cb.call(bad_client_call)

    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_manual_reset():
    """Verify administrative reset returns breaker to pristine CLOSED state."""
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure(QuotaExceededError("429"))
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.last_failure_time is None
    assert cb._trial_in_flight is False


# ============================================================================
# 2. Error Classification, Exponential Backoff & Retry Tests
# ============================================================================

def test_is_transient_error_classification():
    """Verify is_transient_error classifies retriable vs non-retriable exceptions correctly."""
    # Transient domain and network errors
    assert is_transient_error(QuotaExceededError("Rate limit reached")) is True
    assert is_transient_error(VectorServiceError("Vector timeout")) is True
    assert is_transient_error(httpx.ReadTimeout("Timeout")) is True
    assert is_transient_error(httpx.ConnectError("Connection refused")) is True
    assert is_transient_error(ConnectionResetError("Connection reset by peer")) is True
    assert is_transient_error(TimeoutError("Operation timed out")) is True
    assert is_transient_error(asyncio.TimeoutError()) is True

    # Status code checks on custom / generic exceptions
    class Error429(Exception):
        status_code = 429
    assert is_transient_error(Error429()) is True

    class Error502(Exception):
        status_code = 502
    assert is_transient_error(Error502()) is True

    class Error503(Exception):
        status_code = 503
    assert is_transient_error(Error503()) is True

    class Error504(Exception):
        status_code = 504
    assert is_transient_error(Error504()) is True

    class ResponseWrapperError(Exception):
        class Response:
            status_code = 503
        response = Response()
    assert is_transient_error(ResponseWrapperError()) is True

    # Non-transient errors (Client 4xx, Validation, Auth, CircuitBreaker)
    assert is_transient_error(ValidationException("Invalid input")) is False
    assert is_transient_error(UnauthorizedException("No token")) is False
    assert is_transient_error(ForbiddenException("Insufficient clearance")) is False
    assert is_transient_error(EntityNotFoundException("User", "u-1")) is False
    assert is_transient_error(ConflictException("Resource exists")) is False
    assert is_transient_error(DocumentIngestionError("d-1", "bad pdf")) is False
    assert is_transient_error(CircuitBreakerOpenError("Gemini")) is False

    class Error400(Exception):
        status_code = 400
    assert is_transient_error(Error400()) is False

    class Error422(Exception):
        status_code = 422
    assert is_transient_error(Error422()) is False

    assert is_transient_error(KeyError("missing_field")) is False
    assert is_transient_error(TypeError("unexpected argument")) is False
    assert is_transient_error(ValueError("invalid literal")) is False


@pytest.mark.asyncio
async def test_retry_backoff_succeeds_on_transient_error_after_retries():
    """Verify execute_with_retry recovers after transient failures and tracks sleep calls."""
    attempts = 0
    sleeps = []

    async def mock_sleep(d):
        sleeps.append(d)

    async def flaky_service():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise QuotaExceededError(f"Attempt {attempts} 429")
        return "success_payload"

    result = await execute_with_retry(
        flaky_service,
        max_retries=3,
        base_delay=1.0,
        sleep_fn=mock_sleep,
    )

    assert result == "success_payload"
    assert attempts == 3
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_retry_backoff_full_jitter_bounds():
    """Verify AWS Full Jitter produces delays within [0, min(max_delay, base_delay * 2^(attempt-1))]."""
    sleeps = []

    async def mock_sleep(d):
        sleeps.append(d)

    async def failing_service():
        raise VectorServiceError("Gateway Timeout 504")

    with pytest.raises(VectorServiceError):
        await execute_with_retry(
            failing_service,
            max_retries=4,
            base_delay=1.0,
            max_delay=8.0,
            sleep_fn=mock_sleep,
        )

    # 4 attempts -> 3 retries (sleeps)
    assert len(sleeps) == 3
    assert 0.0 <= sleeps[0] <= 1.0  # min(8, 1 * 2^0) = 1.0
    assert 0.0 <= sleeps[1] <= 2.0  # min(8, 1 * 2^1) = 2.0
    assert 0.0 <= sleeps[2] <= 4.0  # min(8, 1 * 2^2) = 4.0


@pytest.mark.asyncio
async def test_retry_aborts_immediately_on_non_transient_error():
    """Verify execute_with_retry does NOT retry non-transient validation errors."""
    call_count = 0
    sleeps = []

    async def mock_sleep(d):
        sleeps.append(d)

    async def bad_request_func():
        nonlocal call_count
        call_count += 1
        raise ValidationException("Invalid schema format")

    with pytest.raises(ValidationException):
        await execute_with_retry(bad_request_func, max_retries=3, sleep_fn=mock_sleep)

    assert call_count == 1
    assert len(sleeps) == 0


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_original_exception():
    """Verify exhausting all retries on transient errors re-raises the underlying exception."""
    call_count = 0

    async def noop_sleep(d):
        pass

    async def persistent_outage():
        nonlocal call_count
        call_count += 1
        raise QuotaExceededError("Persistent 429 quota exhaustion")

    with pytest.raises(QuotaExceededError) as exc_info:
        await execute_with_retry(persistent_outage, max_retries=3, sleep_fn=noop_sleep)

    assert "Persistent 429 quota exhaustion" in str(exc_info.value)
    assert call_count == 3


# ============================================================================
# 3. Resilient GeminiClient Guardrails, Tiers & Fallback Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gemini_client_token_guardrail_blocks_oversized_prompt():
    """Verify prompt exceeding 8192 estimated tokens raises ValidationException without network dispatch."""
    mock_executor = AsyncMock()
    client = GeminiClient(chat_executor=mock_executor)

    oversized_prompt = "token " * 9000  # estimated ~9000 tokens
    with pytest.raises(ValidationException) as exc_info:
        await client.generate_chat_response(oversized_prompt)

    assert "exceeds maximum allowed token limit" in exc_info.value.message
    mock_executor.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_client_token_guardrail_blocks_empty_prompt():
    """Verify empty or whitespace-only prompts raise ValidationException."""
    client = GeminiClient()

    with pytest.raises(ValidationException) as exc1:
        await client.generate_chat_response("")
    assert "cannot be empty" in exc1.value.message

    with pytest.raises(ValidationException) as exc2:
        await client.generate_chat_response("   \t\n   ")
    assert "cannot be empty" in exc2.value.message


@pytest.mark.asyncio
async def test_gemini_client_token_guardrail_accounts_for_system_instruction():
    """Verify system instruction is included in token estimation guardrail."""
    client = GeminiClient()
    system_inst = "instruction " * 5000
    user_prompt = "user prompt " * 4000

    with pytest.raises(ValidationException) as exc_info:
        await client.generate_chat_response(user_prompt, system_instruction=system_inst)

    assert "exceeds maximum allowed token limit" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_client_primary_model_success():
    """Verify successful primary model execution returns expected output and keeps breaker CLOSED."""
    async def mock_executor(prompt, model, system_instruction=None):
        return f"Primary model output for {model}: {prompt}"

    client = GeminiClient(
        default_chat_model="gemini-1.5-pro",
        chat_executor=mock_executor,
    )
    result = await client.generate_chat_response("Analyze quarterly report")

    assert result == "Primary model output for gemini-1.5-pro: Analyze quarterly report"
    assert client.circuit_breaker.state == CircuitState.CLOSED
    assert client.circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_gemini_client_fallback_to_secondary_model_on_primary_failure():
    """
    Verify that when Primary Model fails with transient error,
    GeminiClient smoothly falls back to Secondary Model (gemini-1.5-flash) without breaker deadlock.
    """
    async def mock_sleep(d):
        pass

    async def mock_executor(prompt, model, system_instruction=None):
        if model == "gemini-1.5-pro":
            raise QuotaExceededError("Rate limit reached on gemini-1.5-pro")
        return f"Secondary fallback response from {model}"

    client = GeminiClient(
        default_chat_model="gemini-1.5-pro",
        secondary_chat_model="gemini-1.5-flash",
        chat_executor=mock_executor,
        sleep_fn=mock_sleep,
    )

    result = await client.generate_chat_response("Summarize policy doc", max_retries=2)

    assert result == "Secondary fallback response from gemini-1.5-flash"
    assert client.circuit_breaker.state == CircuitState.CLOSED
    assert client.circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_gemini_client_fallback_to_local_message_when_all_models_fail():
    """Verify that when both primary and secondary models fail, local fallback message is returned."""
    async def mock_sleep(d):
        pass

    async def mock_executor(prompt, model, system_instruction=None):
        raise QuotaExceededError(f"Transient failure on {model}")

    client = GeminiClient(
        chat_executor=mock_executor,
        sleep_fn=mock_sleep,
    )

    result = await client.generate_chat_response("Generate executive summary", max_retries=2)

    assert result == client.LOCAL_FALLBACK_MESSAGE
    assert client.circuit_breaker.failure_count == 1


@pytest.mark.asyncio
async def test_gemini_client_circuit_breaker_open_suppresses_network_calls():
    """
    Verify that when CircuitBreaker is OPEN:
    - Chat returns LOCAL_FALLBACK_MESSAGE immediately without remote execution.
    - Embedding raises CircuitBreakerOpenError without remote execution.
    """
    mock_chat = AsyncMock()
    mock_embed = AsyncMock()

    cb = CircuitBreaker(service_name="TestGeminiGateway", failure_threshold=1)
    cb.state = CircuitState.OPEN
    cb.last_failure_time = time.time()

    client = GeminiClient(
        circuit_breaker=cb,
        chat_executor=mock_chat,
        embedding_executor=mock_embed,
    )

    # Chat call returns local fallback
    chat_result = await client.generate_chat_response("Are you online?")
    assert chat_result == client.LOCAL_FALLBACK_MESSAGE
    mock_chat.assert_not_called()

    # Embedding call fast-fails with CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await client.generate_embedding("Embed this chunk")

    assert exc_info.value.service_name == "TestGeminiGateway"
    mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_client_embedding_success_returns_768_dim():
    """Verify default embedding generation returns a 768-dimensional list of floats."""
    client = GeminiClient()
    embedding = await client.generate_embedding("Standard document chunk for vectorization.")

    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(v, float) for v in embedding)


@pytest.mark.asyncio
async def test_gemini_client_embedding_blocks_empty_text():
    """Verify empty text for embeddings raises ValidationException."""
    client = GeminiClient()

    with pytest.raises(ValidationException):
        await client.generate_embedding("")

    with pytest.raises(ValidationException):
        await client.generate_embedding("   \t  ")


@pytest.mark.asyncio
async def test_gemini_client_embedding_blocks_oversized_text():
    """Verify embedding input exceeding 8192 tokens raises ValidationException."""
    client = GeminiClient()
    oversized = "chunk " * 9000

    with pytest.raises(ValidationException) as exc_info:
        await client.generate_embedding(oversized)

    assert "exceeds maximum allowed token limit" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_client_embedding_failure_raises_vector_service_error():
    """Verify remote embedding failure wraps cleanly into VectorServiceError (HTTP 502)."""
    async def mock_sleep(d):
        pass

    async def failing_embed(text, model):
        raise QuotaExceededError("Google API quota exhausted")

    client = GeminiClient(embedding_executor=failing_embed, sleep_fn=mock_sleep)

    with pytest.raises(VectorServiceError) as exc_info:
        await client.generate_embedding("Valid chunk text", max_retries=2)

    assert "Vector embedding service failed" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_client_recovers_after_breaker_half_open_trial():
    """Verify GeminiClient automatically recovers to normal execution once HALF_OPEN trial succeeds."""
    cb = CircuitBreaker(service_name="GeminiRecovery", recovery_timeout=1.0)
    cb.state = CircuitState.OPEN
    cb.last_failure_time = time.time() - 2.0  # Timeout elapsed

    async def recovering_executor(prompt, model, system_instruction=None):
        return "Service has fully recovered."

    client = GeminiClient(circuit_breaker=cb, chat_executor=recovering_executor)

    result = await client.generate_chat_response("Health ping")

    assert result == "Service has fully recovered."
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb._trial_in_flight is False
