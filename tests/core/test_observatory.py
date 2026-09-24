"""Context Observatory (Phase 1) instrumentation contracts."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from loguru import logger

from free_claude_code.config.logging_config import configure_logging
from free_claude_code.core.observatory import (
    ObservatoryTap,
    build_observatory_event,
    derive_provider_input_tokens,
    emit_llm_request_event,
    extract_responses_terminal_usage,
    normalize_provider_usage,
)


def _observatory_rows(log_file: str) -> list[dict]:
    """Return only observatory rows (``observatory`` flag set) from the log."""
    logger.complete()
    text = Path(log_file).read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows = [json.loads(line) for line in text.split("\n")]
    return [row for row in rows if row.get("observatory") is True]


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


class _ScriptedStream:
    """Yields chunks; optional exception raised after the scripted chunks."""

    def __init__(
        self,
        chunks: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __aiter__(self) -> "_ScriptedStream":
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                raise self._error from None
            raise StopAsyncIteration from None


_MESSAGES_OK = [
    _sse("message_start", {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 120, "output_tokens": 1}},
    }),
    _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "t1", "name": "Bash"},
    }),
    _sse("content_block_start", {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "t2", "name": "Read"},
    }),
    _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {
            "input_tokens": 120,
            "output_tokens": 57,
            "cache_read_input_tokens": 80,
        },
    }),
    _sse("message_stop", {"type": "message_stop"}),
]


# ---------------------------------------------------------------------------
# 1 + 6. Non-streaming success: event emitted, usage + finish + tool_calls.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nonstreaming_success_records_full_event(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    tap = ObservatoryTap(
        _ScriptedStream(_MESSAGES_OK),
        endpoint="/v1/messages",
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        request_id="req_test1",
        generation_id=3,
        local_input_tokens=135,
        streaming=False,
    )
    chunks = [chunk async for chunk in tap]
    assert chunks == _MESSAGES_OK

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "llm_request"
    assert row["request_id"] == "req_test1"
    assert row["endpoint"] == "/v1/messages"
    assert row["provider"] == "openrouter"
    assert row["model"] == "qwen/qwen3.8-27b"
    assert row["generation_id"] == 3
    assert row["streaming"] is False
    assert row["http_status"] == 200
    assert row["finish_reason"] == "tool_use"
    assert row["tool_calls"] == 2
    assert row["local_input_tokens_estimate"] == 135
    # Provider-reported, source-tagged. ``provider_input_tokens`` is the
    # provider's own reported ``input_tokens`` verbatim (never a synthesized
    # sum of input + cache, which would double-count on the inclusive
    # OpenAI-Chat convention).
    assert row["usage"]["provider_input_tokens"] == 120
    assert row["usage"]["provider_cached_input_tokens"] == 80
    assert row["usage"]["output_tokens"] == 57


# ---------------------------------------------------------------------------
# 2. Streaming success: same metadata, streaming flag true, bytes unchanged.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_streaming_success_records_event_and_passes_bytes(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    # Split the stream into smaller wire fragments to prove the tap reassembles
    # usage across arbitrarily split SSE chunks (the real wire is not framed).
    joined = "".join(_MESSAGES_OK)
    fragments = [joined[i:i + 7] for i in range(0, len(joined), 7)]
    tap = ObservatoryTap(
        _ScriptedStream(fragments),
        endpoint="/v1/messages",
        provider="openrouter",
        model="m",
        request_id="req_stream",
        generation_id=None,
        local_input_tokens=None,
        streaming=True,
    )
    received = [chunk async for chunk in tap]
    # The tap must yield chunks exactly as received (no re-framing).
    assert received == fragments
    assert "".join(received) == joined

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["streaming"] is True
    assert rows[0]["finish_reason"] == "tool_use"
    assert rows[0]["usage"]["output_tokens"] == 57


# ---------------------------------------------------------------------------
# 3. Provider usage extraction: input + output token extraction.
# ---------------------------------------------------------------------------
def test_provider_usage_extraction() -> None:
    normalized = normalize_provider_usage(
        {
            "input_tokens": 120,
            "output_tokens": 57,
            "cache_read_input_tokens": 80,
        }
    )
    assert normalized["input_tokens"] == 120
    assert normalized["output_tokens"] == 57
    assert normalized["cached_input_tokens"] == 80
    # The provider's own reported total, not a sum of input + cache.
    assert derive_provider_input_tokens(normalized) == 120


# ---------------------------------------------------------------------------
# Regression: the OpenAI-compatible Chat transport (the /v1/messages path
# under review) reports ``input_tokens`` as the FULL inclusive prompt total
# (provider ``prompt_tokens``) with ``cache_read_input_tokens`` a subset of
# it. The observatory must NOT add the cached amount back on top, or it
# double-counts. E.g. prompt_tokens=1000, cached=400 -> total is 1000, not 1400.
# ---------------------------------------------------------------------------
def test_openai_chat_inclusive_input_not_double_counted() -> None:
    # Mirrors the exact shape of the OpenAI-Chat message_delta usage frame.
    frame = {
        "input_tokens": 1000,          # provider prompt_tokens: inclusive total
        "output_tokens": 57,
        "cache_read_input_tokens": 400,  # a subset of the 1000 above
    }
    normalized = normalize_provider_usage(frame)
    # provider_input_tokens must be the provider's own reported 1000.
    assert derive_provider_input_tokens(normalized) == 1000

    event = build_observatory_event(
        request_id="r",
        endpoint="/v1/messages",
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        wire_api="messages",
        streaming=True,
        duration_ms=10,
        provider_usage=frame,
    )
    usage = event["usage"]
    assert usage["provider_input_tokens"] == 1000  # not 1400
    assert usage["provider_cached_input_tokens"] == 400
    assert usage["output_tokens"] == 57


# ---------------------------------------------------------------------------
# 4. Cached token extraction (Anthropic + OpenAI nested forms).
# ---------------------------------------------------------------------------
def test_cached_token_extraction_anthropic_and_openai() -> None:
    anthropic = normalize_provider_usage(
        {"input_tokens": 10, "cache_read_input_tokens": 4}
    )
    assert anthropic["cached_input_tokens"] == 4

    openai = normalize_provider_usage(
        {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 7}}
    )
    assert openai["cached_input_tokens"] == 7


# ---------------------------------------------------------------------------
# 5. Output token extraction.
# ---------------------------------------------------------------------------
def test_output_token_extraction() -> None:
    normalized = normalize_provider_usage({"output_tokens": 42})
    assert normalized["output_tokens"] == 42
    event = build_observatory_event(
        request_id="r",
        endpoint="/v1/messages",
        provider="p",
        model="m",
        wire_api="messages",
        streaming=False,
        duration_ms=10,
        provider_usage={"output_tokens": 42},
    )
    assert event["usage"]["output_tokens"] == 42


# ---------------------------------------------------------------------------
# 6 (extra). Cost extraction when the provider reports it (OpenRouter).
# ---------------------------------------------------------------------------
def test_cost_extraction() -> None:
    normalized = normalize_provider_usage(
        {"input_tokens": 5, "output_tokens": 9, "cost": 0.0249}
    )
    assert normalized["cost"] == 0.0249
    event = build_observatory_event(
        request_id="r",
        endpoint="/v1/messages",
        provider="openrouter",
        model="m",
        wire_api="messages",
        streaming=False,
        duration_ms=1,
        provider_usage={"input_tokens": 5, "output_tokens": 9, "cost": 0.0249},
    )
    assert event["usage"]["cost"] == 0.0249


# ---------------------------------------------------------------------------
# 7. Missing usage fields must not crash; event still emitted, no usage key.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_usage_does_not_crash(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    # A stream with no usage at all.
    chunks = [
        _sse("message_start", {"type": "message_start", "message": {}}),
        _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        }),
        _sse("message_stop", {"type": "message_stop"}),
    ]
    tap = ObservatoryTap(
        _ScriptedStream(chunks),
        endpoint="/v1/messages",
        provider="p",
        model="m",
        request_id="req_nousage",
        generation_id=None,
        local_input_tokens=10,
        streaming=False,
    )
    received = [chunk async for chunk in tap]
    assert received == chunks
    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["finish_reason"] == "end_turn"
    # No provider usage reported -> no ``usage`` key, no invented numbers.
    assert "usage" not in rows[0]
    assert rows[0]["local_input_tokens_estimate"] == 10


# ---------------------------------------------------------------------------
# 8. Instrumentation does not change the returned response bytes.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tap_does_not_mutate_response_bytes(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    source = _MESSAGES_OK
    tap = ObservatoryTap(
        _ScriptedStream(list(source)),
        endpoint="/v1/messages",
        provider="p",
        model="m",
        request_id="req_bytes",
        generation_id=None,
        local_input_tokens=None,
        streaming=False,
    )
    received = [chunk async for chunk in tap]
    assert received == source


# ---------------------------------------------------------------------------
# 9. Failed requests are recorded appropriately.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failed_request_is_recorded(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    tap = ObservatoryTap(
        _ScriptedStream(
            [_MESSAGES_OK[0]],
            error=RuntimeError("provider blew up"),
        ),
        endpoint="/v1/messages",
        provider="p",
        model="m",
        request_id="req_fail",
        generation_id=None,
        local_input_tokens=None,
        streaming=False,
    )
    with pytest.raises(RuntimeError, match="provider blew up"):
        async for _ in tap:
            pass

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["http_status"] == 500
    assert rows[0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# 10. Sensitive request contents are not written to the telemetry log.
# ---------------------------------------------------------------------------
def test_sensitive_contents_not_in_event() -> None:
    secret = "sk-ant-super-secret-123"
    event = build_observatory_event(
        request_id="r",
        endpoint="/v1/messages",
        provider="p",
        model="m",
        wire_api="messages",
        streaming=False,
        duration_ms=1,
        local_input_tokens=100,
        provider_usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 2,
            "api_key": secret,  # a credential-shaped key that must be redacted
            "secret": secret,
        },
    )
    serialized = json.dumps(event)
    assert secret not in serialized
    # No request/message/prompt fields are ever present.
    for forbidden in ("messages", "prompt", "request", "content", "system"):
        assert forbidden not in event


def test_emit_llm_request_event_writes_json_row(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")
    event = build_observatory_event(
        request_id="req_emit",
        endpoint="/v1/messages",
        provider="p",
        model="m",
        wire_api="messages",
        streaming=False,
        duration_ms=12,
        provider_usage={"input_tokens": 3, "output_tokens": 4},
    )
    emit_llm_request_event(event)
    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["request_id"] == "req_emit"
    assert rows[0]["level"] == "INFO"


def test_extra_provider_fields_preserved() -> None:
    normalized = normalize_provider_usage(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    )
    # Unknown nested fields preserved under ``extra`` for future analysis.
    assert normalized["extra"]["prompt_tokens_details"] == {"cached_tokens": 4}


# ---------------------------------------------------------------------------
# End-to-end (tap-level) regression for the reviewed /v1/messages path: drive
# the real ObservatoryTap over a stream shaped exactly like the OpenAI-Chat
# transport's terminal frames, where message_delta carries the provider's
# inclusive prompt_tokens plus a cache_read subset. Proves the emitted event's
# provider_input_tokens is the provider's own total (not total + cached).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tap_openai_chat_stream_not_double_counted(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    # message_start carries the LOCAL estimate (135); message_delta overwrites
    # with the provider's inclusive prompt_tokens (1000) + cached subset (400).
    chunks = [
        _sse("message_start", {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 135, "output_tokens": 1}},
        }),
        _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 57,
                "cache_read_input_tokens": 400,
            },
        }),
        _sse("message_stop", {"type": "message_stop"}),
    ]
    tap = ObservatoryTap(
        _ScriptedStream(chunks),
        endpoint="/v1/messages",
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        request_id="req_openai_chat",
        generation_id=None,
        local_input_tokens=135,
        streaming=True,
    )
    received = [chunk async for chunk in tap]
    assert received == chunks  # bytes pass through untouched

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    row = rows[0]
    usage = row["usage"]
    # Provider-reported total = the provider's own 1000 (not 1400).
    assert usage["provider_input_tokens"] == 1000
    assert usage["provider_cached_input_tokens"] == 400
    assert usage["output_tokens"] == 57
    # Local estimate stays separately labelled.
    assert row["local_input_tokens_estimate"] == 135


# ---------------------------------------------------------------------------
# Regression: duration_ms must reflect the actual elapsed stream time, not 0.
# The tap's __aiter__ can be bypassed when a consumer drives it via anext()
# directly (as _ObservatoryForward does); the start time must still be
# captured on the first __anext__.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duration_ms_is_nonzero_for_a_slow_stream(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    import asyncio

    async def _slow_stream() -> AsyncIterator[str]:
        yield _MESSAGES_OK[0]
        await asyncio.sleep(0.05)  # simulate time-to-next-chunk
        yield "".join(_MESSAGES_OK[1:])

    tap = ObservatoryTap(
        _slow_stream(),
        endpoint="/v1/messages",
        provider="openrouter",
        model="m",
        request_id="req_dur",
        generation_id=None,
        local_input_tokens=None,
        streaming=True,
    )
    # Drive via anext() directly, bypassing __aiter__ (mirrors
    # _ObservatoryForward) to prove the start is still captured.
    chunks = []
    it = tap  # not iter()ed; anext() calls __anext__ directly
    while True:
        try:
            chunks.append(await anext(it))
        except StopAsyncIteration:
            break

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["duration_ms"] >= 40  # ~50ms sleep; tolerate scheduler jitter


# ---------------------------------------------------------------------------
# Regression: when the serving provider model differs from the gateway model
# (e.g. fallback, or a gateway alias mapping to a different provider model),
# the event carries a distinct ``provider_model`` field while ``model`` stays
# the gateway model. The final values are read from the live state dict at
# emit time, so a fallback that mutates the dict is reflected.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_model_surfaced_when_it_differs(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    # The live state dict the executor shares. It starts at the primary and is
    # updated to the serving candidate (simulating a fallback) before the tap
    # finishes and emits.
    state = {
        "provider_id": "open_router",
        "provider_model": "z-ai/glm-5.3-flash",
    }
    tap = ObservatoryTap(
        _ScriptedStream(_MESSAGES_OK),
        endpoint="/v1/messages",
        provider="open_router",
        model="claude-fable-5-1",  # the gateway / public model
        request_id="req_pv",
        generation_id=None,
        local_input_tokens=None,
        streaming=True,
        observatory_state=state,
    )
    async for _ in tap:
        pass

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    row = rows[0]
    # ``model`` stays the gateway model (unchanged behaviour).
    assert row["model"] == "claude-fable-5-1"
    # The actual serving provider model is surfaced separately.
    assert row["provider_model"] == "z-ai/glm-5.3-flash"
    assert row["provider"] == "open_router"


def test_provider_model_omitted_when_same_as_model() -> None:
    event = build_observatory_event(
        request_id="r",
        endpoint="/v1/messages",
        provider="p",
        model="m",
        wire_api="messages",
        streaming=False,
        duration_ms=1,
        provider_model="m",  # identical to the gateway model -> omit
    )
    assert "provider_model" not in event


# ---------------------------------------------------------------------------
# /v1/responses: extract_responses_terminal_usage reads the structured usage
# the pipeline already puts in the terminal response.completed /
# response.incomplete event. This is the new coverage for the Responses wire.
# ---------------------------------------------------------------------------
def _responses_sse(event_type: str, response: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps({'type': event_type, 'response': response})}\n\n"


def test_responses_terminal_usage_extracted_from_completed() -> None:
    sse = (
        _responses_sse("response.created", {"id": "r1", "status": "in_progress", "model": "m"})
        + _responses_sse(
            "response.completed",
            {
                "id": "r1",
                "status": "completed",
                "model": "m",
                "usage": {
                    "input_tokens": 900,
                    "output_tokens": 42,
                    "input_tokens_details": {"cached_tokens": 300},
                    "output_tokens_details": {"reasoning_tokens": 10},
                    "total_tokens": 942,
                },
            },
        )
    )
    usage, outcome = extract_responses_terminal_usage(sse)
    assert outcome == "ok"
    assert usage is not None
    assert usage["input_tokens"] == 900
    assert usage["output_tokens"] == 42
    assert usage["input_tokens_details"]["cached_tokens"] == 300
    assert usage["output_tokens_details"]["reasoning_tokens"] == 10


def test_responses_terminal_usage_from_incomplete() -> None:
    sse = _responses_sse(
        "response.incomplete",
        {
            "id": "r2",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 50, "output_tokens": 200},
        },
    )
    usage, outcome = extract_responses_terminal_usage(sse)
    assert outcome == "incomplete"
    assert usage is not None
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] == 200


def test_responses_no_terminal_usage_returns_none() -> None:
    # A stream with no terminal usage event (e.g. truncated) yields nothing.
    sse = _responses_sse("response.created", {"id": "r3", "status": "in_progress"})
    usage, outcome = extract_responses_terminal_usage(sse)
    assert usage is None
    assert outcome is None


def test_responses_extract_ignores_non_terminal_events() -> None:
    # Non-terminal events must not be mistaken for the usage source.
    sse = _responses_sse(
        "response.output_item.added",
        {"id": "r4", "status": "in_progress"},
    )
    usage, outcome = extract_responses_terminal_usage(sse)
    assert usage is None
    assert outcome is None


def test_responses_usage_survives_split_chunks() -> None:
    # The terminal frame can arrive split across chunks; the forwarder joins
    # them before extraction. Simulate by splitting the joined text.
    sse = _responses_sse(
        "response.completed",
        {"id": "r5", "status": "completed",
         "usage": {"input_tokens": 10, "output_tokens": 5}},
    )
    fragments = [sse[i:i + 9] for i in range(0, len(sse), 9)]
    usage, outcome = extract_responses_terminal_usage("".join(fragments))
    assert outcome == "ok"
    assert usage is not None
    assert usage["input_tokens"] == 10


# End-to-end: drive the real forwarder over a Responses stream and confirm the
# observatory row carries the provider usage, while chunks pass through.
@pytest.mark.asyncio
async def test_responses_forwarder_records_provider_usage(tmp_path) -> None:
    log_file = str(tmp_path / "obs.log")
    configure_logging(log_file, force=True, level="INFO")

    from free_claude_code.application.execution import _ResponsesObservatoryForward

    chunks = [
        _responses_sse("response.created", {"id": "r6", "status": "in_progress", "model": "m"}),
        _responses_sse(
            "response.completed",
            {"id": "r6", "status": "completed", "model": "m",
             "usage": {"input_tokens": 700, "output_tokens": 30,
                       "input_tokens_details": {"cached_tokens": 250}}},
        ),
    ]
    forward = _ResponsesObservatoryForward(
        _ScriptedStream(chunks),
        request_id="req_responses",
        endpoint="/v1/responses",
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        generation_id=None,
        local_input_tokens=720,
    )
    received = [chunk async for chunk in forward]
    assert received == chunks  # bytes pass through untouched

    rows = _observatory_rows(log_file)
    assert len(rows) == 1
    row = rows[0]
    assert row["wire_api"] == "responses"
    assert row["endpoint"] == "/v1/responses"
    assert row["http_status"] == 200
    assert row["local_input_tokens_estimate"] == 720
    usage = row["usage"]
    # Provider-reported, source-tagged, from the terminal event.
    assert usage["provider_input_tokens"] == 700
    assert usage["provider_cached_input_tokens"] == 250
    assert usage["output_tokens"] == 30
