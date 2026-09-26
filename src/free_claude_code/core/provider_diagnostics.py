"""Credential-safe provider-diagnostic shape builders.

These helpers produce *structural* snapshots of an outbound OpenAI-compatible
chat request and of an error response. They exist so the transport can log what
it actually sent and what a provider rejected, without ever exposing prompts,
messages, tool contents/schemas, credentials, or authorization values.

All functions are pure (no IO, no transport imports) so they are unit-testable
in isolation.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .diagnostics import extract_upstream_error_detail

# Maximum bytes of an error body to retain in diagnostics. Mirrors the existing
# cap used for user-facing error detail.
_DIAGNOSTIC_ERROR_CAP_BYTES = 8_192

# Response-header allowlist: only these keys (lowercased) are captured. Anything
# else -- most importantly authorization, set-cookie, and credential fields -- is
# dropped outright.
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "server",
        "retry-after",
        "x-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-used",
        "x-openrouter-request-id",
        "x-openrouter-model",
        "cf-ray",
    }
)

# Parameter names whose *values* are never logged; only present/type is.
_OBSERVABLE_PARAMETER_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "reasoning_effort",
    "response_format",
)

# Top-level keys that are always omitted (they can carry user content or PII).
_OMITTED_TOP_LEVEL_KEYS = frozenset(
    {
        "user",
        "messages",
        "input",
        "system",
        "prompt",
        "content",
        "metadata",
        "seed",
        "tools",
    }
)

# Credential-shaped header names, shown as "<redacted>" in header-name lists.
_CREDENTIAL_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "cookie",
        "set-cookie",
        "x-auth-token",
    }
)


def redact_base_url(url: str) -> str:
    """Strip userinfo credentials from a URL, preserving scheme+host+path."""
    if not url:
        return ""
    if "://" in url:
        scheme, _, rest = url.partition("://")
        authority, sep, tail = rest.partition("/")
        if "@" in authority:
            authority = authority.rsplit("@", 1)[1]
        return f"{scheme}://{authority}{sep}{tail}"
    # Scheme-less URL: strip a leading userinfo@ from the host portion.
    if "@" in url:
        return url.rsplit("@", 1)[1]
    return url


def _content_type_of(content: Any) -> str:
    if isinstance(content, str):
        return "string"
    if isinstance(content, list):
        return "list"
    if isinstance(content, dict):
        return "object"
    return type(content).__name__


def _content_chars(message: Mapping[str, Any]) -> int:
    content = message.get("content")
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
                total += len(json.dumps(part, ensure_ascii=False))
            else:
                total += len(str(part))
        return total
    if isinstance(content, (Mapping, list)):
        return len(json.dumps(content, ensure_ascii=False))
    return len(str(content))


def message_summary(message: Mapping[str, Any], index: int) -> dict[str, Any]:
    """One message's safe shape: role, content type, and char count only."""
    return {
        "index": index,
        "role": str(message.get("role", "")),
        "content_type": _content_type_of(message.get("content")),
        "content_chars": _content_chars(message),
    }


def tool_summary(tool: Mapping[str, Any], index: int) -> dict[str, Any]:
    """One tool's safe shape: type, name, and function-schema key names only."""
    summary: dict[str, Any] = {
        "index": index,
        "type": str(tool.get("type", "")),
    }
    name = tool.get("name")
    if name is None:
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else None
        name = fn.get("name") if fn is not None else None
    if isinstance(name, str):
        summary["name"] = name
    fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else None
    schema = fn.get("parameters") if fn is not None else None
    if isinstance(schema, Mapping):
        summary["schema_keys"] = sorted(str(k) for k in schema.keys())
    return summary


def _tool_shapes(tools: Any) -> dict[str, Any]:
    if not isinstance(tools, list):
        return {"count": 0, "types": [], "tools": []}
    types: list[str] = []
    summaries: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            continue
        types.append(str(tool.get("type", "")))
        summaries.append(tool_summary(tool, index))
    return {"count": len(tools), "types": types, "tools": summaries}


def _parameter_summaries(body: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _OBSERVABLE_PARAMETER_KEYS:
        value = body.get(key)
        present = value is not None
        out[key] = {
            "present": present,
            "type": None if not present else type(value).__name__,
        }
    # Surface an allowlist of other commonly-significant scalar parameters without
    # logging their values.
    for scalar_key in ("n", "seed", "logprobs", "stream"):
        value = body.get(scalar_key)
        out[scalar_key] = {
            "present": value is not None,
            "type": None if value is None else type(value).__name__,
        }
    return out


def _header_names(headers: Mapping[str, Any]) -> dict[str, Any]:
    """Header key-only list; credential-shaped names are shown as redacted."""
    names: list[str] = []
    for key in headers.keys():
        low = str(key).lower()
        if low in _CREDENTIAL_HEADER_NAMES:
            names.append("<redacted>")
        else:
            names.append(str(key))
    return {"count": len(names), "names": names}


def outbound_request_shape(body: Mapping[str, Any]) -> dict[str, Any]:
    """Redacted structural snapshot of a serialized chat body.

    Messages/tools are reduced to lengths and types; parameter values become
    present/type markers; top-level key *names* are listed so unexpected fields
    are visible; extra_headers become key-only. Nothing user-authored is logged.
    """
    if not isinstance(body, Mapping):
        return {"_redacted_top_level_keys": [], "model": None}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    shape: dict[str, Any] = {
        "model": body.get("model"),
        "stream": body.get("stream"),
        "messages": [message_summary(m, i) for i, m in enumerate(messages)],
        "tools": _tool_shapes(body.get("tools")),
        "tool_choice": _tool_choice_shape(body.get("tool_choice")),
        "parameters": _parameter_summaries(body),
        "top_level_keys": sorted(str(k) for k in body.keys()),
        "_redacted_top_level_keys": sorted(_OMITTED_TOP_LEVEL_KEYS),
    }
    if body.get("n") is not None:
        shape["n"] = body["n"]
    extra_headers = body.get("extra_headers")
    if isinstance(extra_headers, Mapping):
        shape["extra_headers"] = _header_names(extra_headers)
    return shape


def _tool_choice_shape(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"type": "string_value"}
    if isinstance(value, Mapping):
        return {"type": value.get("type")}
    return {"type": type(value).__name__}


def safe_response_headers(headers: Any) -> dict[str, str]:
    """Capture only an allowlisted subset of a response's headers."""
    out: dict[str, str] = {}
    if headers is None:
        return out
    try:
        items = headers.items()
    except AttributeError:
        try:
            items = headers
        except Exception:
            return out
    for key, value in items:
        low = str(key).lower()
        if low in _SAFE_RESPONSE_HEADERS:
            out[str(key)] = str(value)
    return out


def error_response_shape(error: BaseException) -> dict[str, Any] | None:
    """Safe snapshot of an error's HTTP response (status, headers, capped text).

    Reuses ``extract_upstream_error_detail`` which already redacts credentials
    and caps the body size, then adds the allowlisted headers, content-type, and
    the provider's structured ``error`` fields (type / code / message) when the
    body is recognizable JSON. Returns ``None`` when the error carries no HTTP
    response.
    """
    detail = extract_upstream_error_detail(error)
    response = getattr(error, "response", None)
    result: dict[str, Any] = {
        "http_status": detail.status_code,
    }
    if detail.body_text is not None:
        capped_text = _cap(detail.body_text)
        result["body_text"] = capped_text
        result["body_truncated"] = detail.body_truncated
        _apply_provider_error_fields(result, capped_text)
    if response is not None:
        headers = safe_response_headers(getattr(response, "headers", None) or {})
        try:
            content_type = headers.get("content-type")
        except Exception:
            content_type = None
        if content_type:
            result["content_type"] = content_type
        if headers:
            result["headers"] = headers
    return result


def _apply_provider_error_fields(result: dict[str, Any], body_text: str) -> None:
    """Extract provider ``error`` detail when the body is recognizable JSON.

    Providers differ in shape: OpenAI nests ``type``/``code``/``message`` under
    an ``error`` object, while OpenRouter and several gateways put them at the
    top level. Either form is extracted; values flow through the earlier
    redaction/capping before reaching this point.
    """
    try:
        body = json.loads(body_text)
    except (ValueError, TypeError):
        return
    if not isinstance(body, dict):
        return
    error = body.get("error")
    source = error if isinstance(error, dict) else body
    if not isinstance(source, dict):
        return
    for key, out_key in (
        ("type", "provider_error_type"),
        ("code", "provider_error_code"),
        ("message", "provider_error_message"),
    ):
        value = source.get(key)
        if isinstance(value, (str, int, float)):
            result[out_key] = str(value)


def _cap(text: str) -> str:
    if text is None:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _DIAGNOSTIC_ERROR_CAP_BYTES:
        return text
    return encoded[:_DIAGNOSTIC_ERROR_CAP_BYTES].decode("utf-8", errors="replace")