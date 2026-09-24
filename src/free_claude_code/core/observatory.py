"""Read-only Context Observatory: one structured event per completed LLM request.

Phase 1 records what happens to every LLM request that passes through FCC and
measures context / token growth. It is intentionally passive: it never alters
the bytes returned to the client, it never invents token counts the provider
did not report, and it never writes prompt / message / tool / credential
content to the log.

Events are emitted through the existing loguru logger as a single JSON row
(the ``trace_payload`` binding is promoted to top-level keys by
``config.logging_config``), so the observatory rides the same sink, rotation,
and retention as the rest of the server log.
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from loguru import logger

from free_claude_code.core.trace import sanitize_trace_value

# Fixed namespace so observatory rows are greppable distinct from TRACE rows.
OBSERVATORY_EVENT = "llm_request"

# Keys a provider usage dict may use to express cache reads, mapped to the
# normalized cached-input field.
_CACHE_READ_KEYS = ("cache_read_input_tokens",)
_CACHE_WRITE_KEYS = ("cache_creation_input_tokens",)


def _usage_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _usage_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return value
    return None


def normalize_provider_usage(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize provider-reported usage into stable, source-tagged fields.

    Returns a dict of the fields we can read *without* inventing anything.
    Absent provider fields are simply omitted (no zero-padding) so the
    consumer can tell "provider said 0" from "provider said nothing".
    """
    out: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return out

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = _usage_int(raw.get(key))
        if value is not None:
            out[key] = value

    for key in _CACHE_READ_KEYS:
        value = _usage_int(raw.get(key))
        if value is not None:
            out["cached_input_tokens"] = value

    for key in _CACHE_WRITE_KEYS:
        value = _usage_int(raw.get(key))
        if value is not None:
            out["cache_creation_input_tokens"] = value

    # OpenAI-style nested details.
    details = raw.get("input_tokens_details")
    if isinstance(details, Mapping):
        cached = _usage_int(details.get("cached_tokens"))
        if cached is not None:
            out.setdefault("cached_input_tokens", cached)
    details = raw.get("output_tokens_details")
    if isinstance(details, Mapping):
        reasoning = _usage_int(details.get("reasoning_tokens"))
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning

    # OpenRouter / native cost, when present.
    cost = _usage_number(raw.get("cost"))
    if cost is not None:
        out["cost"] = cost

    # Preserve any extra numeric / dict / str fields the provider gave us
    # (e.g. OpenRouter ``prompt_tokens_details``) verbatim under ``extra``.
    handled = {"input_tokens", "output_tokens", "total_tokens",
               *_CACHE_READ_KEYS, *_CACHE_WRITE_KEYS,
               "input_tokens_details", "output_tokens_details", "cost"}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key in handled:
            continue
        if isinstance(value, (int, float, dict, str)) and not isinstance(value, bool):
            extra[key] = value
    if extra:
        out["extra"] = extra
    return out


def derive_provider_input_tokens(normalized: Mapping[str, Any] | None) -> int | None:
    """Provider-reported input-token total, or ``None`` when not reported.

    Returns the provider's own ``input_tokens`` verbatim. FCC serves two wire
    conventions that both decode to the same Anthropic-shaped SSE: the
    OpenAI-compatible Chat transport reports ``input_tokens`` as the *full
    inclusive* prompt total (with ``cache_read``/``cache_creation`` a subset
    of it), while a native Anthropic relay reports the *non-cached remainder*.
    The decoded frame carries no signal that distinguishes the two, so summing
    cache fields on top is only correct for one convention and would
    double-count cached tokens for the other. We therefore never synthesize a
    larger total; the cache fields remain reported separately so a consumer
    can compute the convention-appropriate total itself.
    """
    if not isinstance(normalized, Mapping):
        return None
    return _usage_int(normalized.get("input_tokens"))


# Responses terminal event types that carry the final usage in their payload.
_RESPONSES_TERMINAL_USAGE_EVENTS = frozenset(
    {"response.completed", "response.incomplete"}
)


def extract_responses_terminal_usage(
    sse_text: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Extract the final ``usage`` dict and outcome from a Responses SSE stream.

    The ``/v1/responses`` wire emits a single terminal event (``response.
    completed`` / ``response.incomplete``) whose ``response.usage`` is the
    provider-reported usage the pipeline already produced (native relay passes
    it through; the Chat-to-Responses path builds it via ``_responses_usage``).
    This reads that structured field; it does not re-derive or invent numbers.

    Returns ``(usage, outcome)`` where ``usage`` is the provider ``usage``
    mapping (or ``None`` when absent) and ``outcome`` is ``"ok"`` for a
    completed terminal event or ``"incomplete"`` for an incomplete one.
    Only the *last* terminal event in the text is considered.
    """
    if not isinstance(sse_text, str) or not sse_text:
        return None, None
    # A Responses SSE frame is "event: <type>\ndata: <json>\n\n". Walk the
    # frames and keep the last terminal one's payload.
    last_usage: Mapping[str, Any] | None = None
    last_outcome: str | None = None
    for frame in sse_text.split("\n\n"):
        event_type: str | None = None
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
        if event_type not in _RESPONSES_TERMINAL_USAGE_EVENTS or not data_lines:
            continue
        try:
            payload = json.loads("".join(data_lines))
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        response = payload.get("response")
        if not isinstance(response, Mapping):
            continue
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            last_usage = usage
            last_outcome = (
                "ok"
                if event_type == "response.completed"
                else "incomplete"
            )
    return last_usage, last_outcome


def build_observatory_event(
    *,
    request_id: str,
    endpoint: str,
    provider: str,
    model: str,
    wire_api: str,
    streaming: bool,
    duration_ms: float,
    generation_id: int | None = None,
    finish_reason: str | None = None,
    http_status: int = 200,
    local_input_tokens: int | None = None,
    provider_model: str | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    tool_calls: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Assemble one observatory event dict, layering provider-reported usage.

    ``model`` is the gateway / public model the client asked for.
    ``provider_model`` is the actual provider model that served the request
    (post-fallback), when it differs from ``model``; it is omitted when the
    two are the same to keep the row compact. ``local_input_tokens`` is FCC's
    own estimate and is always labelled as an estimate. Provider-reported
    numbers come from ``provider_usage`` and are kept under ``usage`` with
    their source intact. No request / prompt / message content is included.
    """
    event: dict[str, Any] = {
        "event": OBSERVATORY_EVENT,
        "request_id": request_id,
        "endpoint": endpoint,
        "provider": provider,
        "model": model,
        "wire_api": wire_api,
        "streaming": streaming,
        "duration_ms": int(duration_ms),
        "http_status": http_status,
    }
    if (
        isinstance(provider_model, str)
        and provider_model
        and provider_model != model
    ):
        event["provider_model"] = provider_model
    if generation_id is not None:
        event["generation_id"] = generation_id
    if finish_reason is not None:
        event["finish_reason"] = finish_reason
    if tool_calls is not None:
        event["tool_calls"] = tool_calls

    if isinstance(local_input_tokens, int) and not isinstance(local_input_tokens, bool):
        event["local_input_tokens_estimate"] = local_input_tokens

    normalized = normalize_provider_usage(provider_usage)
    if normalized:
        provider_usage_out = dict(normalized)
        full_total = derive_provider_input_tokens(normalized)
        if full_total is not None:
            provider_usage_out["provider_input_tokens"] = full_total
        cached = provider_usage_out.get("cached_input_tokens")
        if cached is not None:
            provider_usage_out["provider_cached_input_tokens"] = cached
        event["usage"] = provider_usage_out

    if error_type is not None:
        event["error_type"] = error_type
    if error_message is not None:
        # Keep only a short shape of the error, never a raw provider body.
        event["error"] = str(error_message)[:200]

    return sanitize_trace_value(event)


def emit_llm_request_event(event: Mapping[str, Any]) -> None:
    """Write one observatory event as a JSON row on the server logger.

    Emitted at INFO so it is visible at the default log level (the TRACE
    namespace is DEBUG-gated and would otherwise be invisible in production).
    """
    payload = dict(event)
    payload["observatory"] = True
    logger.bind(trace_payload=sanitize_trace_value(payload)).info(
        "OBSERVATORY {}", OBSERVATORY_EVENT
    )


class ObservatoryTap:
    """Passively observe one Anthropic-SSE stream and record usage metadata.

    Wraps an async iterator of Anthropic-style SSE text. It decodes only the
    metadata-bearing events (``message_start`` / ``message_delta`` /
    ``content_block_start``) to read provider-reported usage, finish-reason,
    and tool-call count. It never re-frames, mutates, or re-orders chunks;
    every chunk is yielded exactly as received. On stream end it emits a
    single observatory event via :func:`emit_llm_request_event`.

    The tap is deliberately best-effort: any failure inside the observation
    path is swallowed so it can never change or break the stream.
    """

    def __init__(
        self,
        body: AsyncIterator[str],
        *,
        endpoint: str,
        provider: str,
        model: str,
        request_id: str,
        generation_id: int | None,
        local_input_tokens: int | None,
        streaming: bool,
        observatory_state: Mapping[str, Any] | None = None,
    ) -> None:
        self._body = body
        self._endpoint = endpoint
        self._provider = provider
        self._model = model
        self._request_id = request_id
        self._generation_id = generation_id
        self._local_input_tokens = local_input_tokens
        self._streaming = streaming
        # The live, mutable state dict the executor updates as candidates are
        # committed. Read at emit time so the event reflects the final
        # (post-fallback) provider / provider model, not the primary.
        self._observatory_state = observatory_state

        self._decoder = None
        self._usage: dict[str, Any] = {}
        self._finish_reason: str | None = None
        self._tool_calls = 0
        self._emitted = False
        self._start = None

    def _ensure_decoder(self):
        if self._decoder is None:
            from free_claude_code.core.anthropic.streaming.decoder import (
                AnthropicSSEDecoder,
            )

            self._decoder = AnthropicSSEDecoder()
        return self._decoder

    def _handle_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        ptype = payload.get("type")
        if ptype == "message_start":
            message = payload.get("message")
            if isinstance(message, Mapping):
                usage = message.get("usage")
                if isinstance(usage, Mapping):
                    self._merge_usage(usage)
        elif ptype == "message_delta":
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                self._merge_usage(usage)
            delta = payload.get("delta")
            if isinstance(delta, Mapping):
                stop = delta.get("stop_reason")
                if isinstance(stop, str) and stop:
                    self._finish_reason = stop
        elif ptype == "content_block_start":
            block = payload.get("content_block")
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                self._tool_calls += 1

    def _merge_usage(self, usage: Mapping[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, (int, float, dict)) and not isinstance(value, bool):
                self._usage[key] = value

    def _emit(self, *, http_status: int, error_type: str | None = None,
              error_message: str | None = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        from time import monotonic

        duration_ms = (monotonic() - self._start) * 1000 if self._start else 0.0
        # Resolve the final serving provider / provider model from the live
        # state at emit time (post-fallback), falling back to the constructor
        # values when the executor did not share its state dict.
        provider = self._provider
        provider_model: str | None = None
        if isinstance(self._observatory_state, Mapping):
            final_provider = self._observatory_state.get("provider_id")
            if isinstance(final_provider, str) and final_provider:
                provider = final_provider
            final_model = self._observatory_state.get("provider_model")
            if isinstance(final_model, str) and final_model:
                provider_model = final_model
        event = build_observatory_event(
            request_id=self._request_id,
            endpoint=self._endpoint,
            provider=provider,
            model=self._model,
            wire_api="messages",
            streaming=self._streaming,
            duration_ms=duration_ms,
            generation_id=self._generation_id,
            finish_reason=self._finish_reason,
            http_status=http_status,
            local_input_tokens=self._local_input_tokens,
            provider_model=provider_model,
            provider_usage=self._usage or None,
            tool_calls=self._tool_calls or None,
            error_type=error_type,
            error_message=error_message,
        )
        try:
            emit_llm_request_event(event)
        except Exception:
            # Observability must never take the request down.
            pass

    def __aiter__(self) -> "ObservatoryTap":
        if self._start is None:
            from time import monotonic

            self._start = monotonic()
        self._ensure_decoder()
        return self

    async def __anext__(self) -> str:
        # Capture the start on first iteration. ``__aiter__`` may be bypassed
        # when a consumer drives the tap via ``anext()`` directly (as
        # ``_ObservatoryForward`` does), so this is the guaranteed entry point.
        if self._start is None:
            from time import monotonic

            self._start = monotonic()
        decoder = self._ensure_decoder()
        try:
            chunk = await anext(self._body)
        except StopAsyncIteration:
            for event in decoder.finish():
                self._handle_payload(event.data)
            self._emit(http_status=200)
            raise
        except BaseException as exc:
            failure_type = type(exc).__name__
            self._emit(
                http_status=500,
                error_type=failure_type,
                error_message=failure_type,
            )
            raise
        try:
            for event in decoder.feed(chunk):
                self._handle_payload(event.data)
        except Exception:
            pass
        return chunk

    async def aclose(self) -> None:
        if not self._emitted:
            self._emit(http_status=200, error_type="cancelled")
        close = getattr(self._body, "aclose", None)
        if close is not None:
            await close()

    def athrow(self, *args: object, **kwargs: object) -> str:
        athrow = getattr(self._body, "athrow", None)
        if athrow is None:
            raise RuntimeError("wrapped stream does not support athrow")
        return athrow(*args, **kwargs)
