"""Provider diagnostics (DIAGNOSTICS task): credential-safe outbound + error logging."""

from __future__ import annotations

import json

import httpx2
import pytest
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
)
from free_claude_code.providers.admission import ProviderAdmissionController

from tests.providers.request_factory import make_messages_request


pytestmark = pytest.mark.asyncio


# --- capture fixture ---------------------------------------------------------


@pytest.fixture
def diagnostics_enabled(monkeypatch):
    """Enable transport diagnostics for the duration of a test and capture events."""
    captured: list[dict] = []

    def sink(message):
        payload = message.record["extra"].get("trace_payload")
        # Only our opt-in diagnostic events. The `diag: true` marker separates
        # them from the pre-existing `provider.*` telemetry (admission
        # `provider.attempt.started`, `provider.request.sent`, ...) which may
        # legitimately carry message content and is out of scope here.
        if (
            isinstance(payload, dict)
            and payload.get("diag") is True
            and payload.get("stage") == "provider"
            and str(payload.get("event", "")).startswith("provider.")
        ):
            captured.append(dict(payload))

    sink_id = logger.add(sink, level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink_id)


def _transport(client: AsyncOpenAI, provider_diagnostics: bool = True) -> OpenAIChatTransport:
    return OpenAIChatTransport(
        client=client,
        admission=ProviderAdmissionController(
            provider_name="open_router",
            rate_limit=100,
            rate_window=1,
            max_concurrency=1,
            max_attempts=1,
        ),
        behavior=OpenAIChatBehavior(
            OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    "open_router", ReasoningReplayMode.DISABLED
                ),
                NO_REASONING,
            )
        ),
        read_timeout_s=2,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
        provider_diagnostics=provider_diagnostics,
    )


def _success_stream_body() -> httpx2.Response:
    chunk = {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}
        ],
    }
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
    )


def _client(reply):
    return AsyncOpenAI(
        api_key="sk-test-secret-do-not-log",
        base_url="https://openrouter.example/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )


# --- 1. successful request logs safe outbound metadata -----------------------


async def test_success_logs_safe_outbound_metadata(diagnostics_enabled):
    calls = 0

    def reply(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        calls += 1
        return _success_stream_body()

    client = _client(reply)
    transport = _transport(client)
    stream = transport.stream_messages(make_messages_request("model", tools=[]))
    chunks = [event async for event in stream]

    assert "hello" in "".join(chunks)
    started = [e for e in diagnostics_enabled if e["event"] == "provider.attempt.started"]
    assert started, "expected a provider.attempt.started event"
    event = started[0]
    assert event["http_method"] == "POST"
    assert event["stream"] is True
    assert event["provider"] == "open_router"
    assert "openrouter.example" in event["base_url"]
    outbound = event["outbound"]
    assert "model" in outbound
    # messages are lengths/shapes only
    assert "messages" in outbound
    flat = json.dumps(event)
    # Never log secrets or content
    assert "sk-test-secret" not in flat
    assert "Hello" not in flat
    assert "authorization" not in flat.lower() or "<redacted>" in flat
    await client.close()


# --- 2. HTTP 400 captures the provider error body -----------------------------


async def test_http_400_captures_error_body(diagnostics_enabled):
    payload = {
        "error": {
            "message": "bad model parameter supplied",
            "type": "invalid_request_error",
            "code": "model_does_not_exist",
        }
    }

    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return httpx2.Response(400, json=payload)

    client = _client(reply)
    transport = _transport(client)
    with pytest.raises(ExecutionFailure):
        async for _ in transport.stream_messages(make_messages_request("model")):
            pass

    failed = [
        e
        for e in diagnostics_enabled
        if e["event"] == "provider.attempt.failed"
    ]
    assert failed
    event = failed[0]
    assert event["http_status"] == 400
    assert event["failure_kind"] == "invalid_request"
    assert "model_does_not_exist" in event.get("provider_error_code", "")
    assert event.get("provider_error_message") == "bad model parameter supplied"
    assert event["response"]["http_status"] == 400
    flat = json.dumps(event)
    assert "sk-test-secret" not in flat
    await client.close()


# --- 3. HTTP 429 captures body + rate-limit headers ----------------------------


async def test_http_429_captures_body_and_ratelimit_headers(diagnostics_enabled):
    payload = {
        "error": {
            "message": "rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_error",
        }
    }

    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return httpx2.Response(
            429,
            json=payload,
            headers={
                "x-ratelimit-limit": "100",
                "x-ratelimit-remaining": "0",
                "retry-after": "3",
                "set-cookie": "session=secret-token",
                "content-type": "application/json",
            },
        )

    client = _client(reply)
    transport = _transport(client)
    with pytest.raises(ExecutionFailure):
        async for _ in transport.stream_messages(make_messages_request("model")):
            pass

    failed = [
        e
        for e in diagnostics_enabled
        if e["event"] == "provider.attempt.failed"
    ]
    assert failed
    event = failed[0]
    assert event["http_status"] == 429
    assert event["failure_kind"] == "rate_limit"
    assert event.get("provider_error_message") == "rate limit exceeded"
    response = event["response"]
    headers = response.get("headers", {})
    assert headers.get("x-ratelimit-limit") == "100"
    assert headers.get("retry-after") == "3"
    flat = json.dumps(event)
    # sensitive header values and credentials never leak
    assert "secret-token" not in flat
    assert "set-cookie" not in flat.lower()
    assert "sk-test-secret" not in flat
    await client.close()


# --- 4. authorization / api key never logged ----------------------------------


async def test_authorization_and_api_key_never_logged(diagnostics_enabled):
    payload = {"error": {"message": "nope", "type": "authentication_error", "code": "401"}}

    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return httpx2.Response(401, json=payload)

    client = _client(reply)
    transport = _transport(client)
    with pytest.raises(ExecutionFailure):
        async for _ in transport.stream_messages(make_messages_request("model")):
            pass

    flat = json.dumps(diagnostics_enabled)
    assert "sk-test-secret" not in flat
    assert "bearer" not in flat.lower()
    await client.close()


# --- 5. tool schema values are not leaked -------------------------------------


async def test_tool_schema_values_not_leaked(diagnostics_enabled):
    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return _success_stream_body()

    tools = [
        {
            "name": "Bash",
            "description": "run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        }
    ]
    client = _client(reply)
    transport = _transport(client)
    req = make_messages_request("model", tools=tools)
    async for _ in transport.stream_messages(req):
        pass

    started = [
        e
        for e in diagnostics_enabled
        if e["event"] == "provider.attempt.started"
    ]
    assert started
    for tool in started[0]["outbound"]["tools"]["tools"]:
        assert "schema_keys" in tool
        # schema VALUES are redacted: input_schema content must not appear
    flat = json.dumps(started[0])
    assert "shell command" not in flat
    assert "command" not in flat  # even parameter names inside schema are stripped
    await client.close()


# --- 6. streaming success does not log generated content ------------------------


async def test_streaming_success_does_not_log_generated_content(diagnostics_enabled):
    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return _success_stream_body()

    client = _client(reply)
    transport = _transport(client)

    async def consume(wire: str) -> str:
        if wire == "messages":
            stream = transport.stream_messages(make_messages_request("model"))
        else:
            stream = transport.stream_responses(
                OpenAIResponsesRequest(model="model", input="hello")
            )
        out: list[str] = []
        async for event in stream:
            out.append(event)
        return "".join(out)

    await consume("messages")
    await consume("responses")
    started = [
        e
        for e in diagnostics_enabled
        if e["event"] == "provider.attempt.started"
    ]
    responses = [
        e for e in diagnostics_enabled if e["event"] == "provider.response"
    ]
    assert started
    assert responses, "expected provider.response events"
    flat = json.dumps(diagnostics_enabled)
    # the provider generated content ("hello") must never be logged
    assert '"hello"' not in flat
    assert '"content": "hello"' not in flat
    await client.close()


# --- 7. off by default: no provider.attempt events -----------------------------


async def test_disabled_by_default_no_diag_events(diagnostics_enabled):
    def reply(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        return _success_stream_body()

    client = _client(reply)
    transport = _transport(client, provider_diagnostics=False)
    async for _ in transport.stream_messages(make_messages_request("model")):
        pass

    events = [e for e in diagnostics_enabled if e["event"].startswith("provider.")]
    assert events == []
    await client.close()