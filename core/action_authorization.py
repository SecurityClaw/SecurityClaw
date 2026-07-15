"""One-time operator authorization for privileged endpoint actions."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_REQUESTS: dict[str, dict[str, Any]] = {}
_TTL_SECONDS = 600


def _action_digest(action: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"action": action, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_authorization(action: str, arguments: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(18)
    with _LOCK:
        _REQUESTS[token] = {
            "digest": _action_digest(action, arguments),
            "expires": time.time() + _TTL_SECONDS,
        }
    return token


def consume_authorization(
    token: str,
    action: str,
    arguments: dict[str, Any],
    operator_message: str,
) -> bool:
    """Consume a token only when the operator echoed it in the current message."""
    if not token or f"AUTHORIZE {token}" not in str(operator_message or ""):
        return False
    with _LOCK:
        request = _REQUESTS.pop(token, None)
    return bool(
        request
        and request["expires"] >= time.time()
        and secrets.compare_digest(request["digest"], _action_digest(action, arguments))
    )
