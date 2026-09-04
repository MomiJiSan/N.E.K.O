from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from plugin.server.local_app_bridge.errors import LocalAppBridgeError, bad_request

PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 64 * 1024
LAUNCH_CODE_TTL_SECONDS = 60.0
ACCESS_TOKEN_TTL_SECONDS = 15 * 60.0
SESSION_IDLE_TTL_SECONDS = 30 * 60.0
SESSION_ABSOLUTE_TTL_SECONDS = 8 * 60 * 60.0
MAX_ACTIVE_SESSIONS = 256
MAX_PENDING_LAUNCH_CODES = 512
SESSION_RATE_LIMIT = 30
SESSION_RATE_WINDOW_SECONDS = 10.0
PAIR_RATE_LIMIT = 5
PAIR_RATE_WINDOW_SECONDS = 60.0
MAX_RATE_LIMIT_BUCKETS = 1024

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")


def require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise bad_request("invalid_identity", f"Invalid {field_name}")
    return value


def require_exact_fields(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise bad_request()
    keys = frozenset(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise bad_request("invalid_fields", "Request fields do not match the contract")
    return value


def json_size(value: object) -> int:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise bad_request(
            "invalid_json_value", "Payload must contain JSON values"
        ) from exc
    return len(encoded)


@dataclass(frozen=True, slots=True)
class LocalAppPolicy:
    app_id: str
    allowed_operations: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        require_identifier(self.app_id, "app_id")
        if (
            not isinstance(self.allowed_operations, Mapping)
            or not self.allowed_operations
        ):
            raise ValueError("allowed_operations must be a non-empty mapping")
        normalized: dict[str, frozenset[str]] = {}
        for scope, operations in self.allowed_operations.items():
            normalized_scope = require_identifier(scope, "scope")
            if not isinstance(operations, frozenset) or not operations:
                raise ValueError(
                    "each scope must have a non-empty frozenset of operations"
                )
            normalized[normalized_scope] = frozenset(
                require_identifier(operation, "operation") for operation in operations
            )
        object.__setattr__(self, "allowed_operations", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class LaunchMaterial:
    bridge_origin: str
    app_id: str
    client_id: str
    launch_code: str = field(repr=False)
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "bridge_origin": self.bridge_origin,
            "app_id": self.app_id,
            "client_id": self.client_id,
            "launch_code": self.launch_code,
        }


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    app_id: str
    client_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class PairResult:
    identity: SessionIdentity
    access_token: str = field(repr=False)
    granted_scopes: frozenset[str] = frozenset()
    access_token_expires_in: int = int(ACCESS_TOKEN_TTL_SECONDS)
    session_idle_timeout: int = int(SESSION_IDLE_TTL_SECONDS)
    session_absolute_timeout: int = int(SESSION_ABSOLUTE_TTL_SECONDS)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "app_id": self.identity.app_id,
            "client_id": self.identity.client_id,
            "session_id": self.identity.session_id,
            "access_token": self.access_token,
            "granted_scopes": sorted(self.granted_scopes),
            "access_token_expires_in": self.access_token_expires_in,
            "session_idle_timeout": self.session_idle_timeout,
            "session_absolute_timeout": self.session_absolute_timeout,
        }


@dataclass(frozen=True, slots=True)
class DispatchEnvelope:
    identity: SessionIdentity
    scope: str
    operation: str
    payload: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: object) -> DispatchEnvelope:
        body = require_exact_fields(
            value,
            required=frozenset(
                {
                    "protocol_version",
                    "app_id",
                    "client_id",
                    "session_id",
                    "scope",
                    "operation",
                    "payload",
                }
            ),
        )
        if body["protocol_version"] != PROTOCOL_VERSION:
            raise bad_request("unsupported_protocol", "Unsupported protocol version")
        payload = body["payload"]
        if not isinstance(payload, Mapping) or any(
            not isinstance(key, str) for key in payload
        ):
            raise bad_request("invalid_payload", "Payload must be an object")
        if json_size(payload) > MAX_BODY_BYTES:
            raise LocalAppBridgeError("payload_too_large", 413, "Payload is too large")
        return cls(
            identity=SessionIdentity(
                app_id=require_identifier(body["app_id"], "app_id"),
                client_id=require_identifier(body["client_id"], "client_id"),
                session_id=require_identifier(body["session_id"], "session_id"),
            ),
            scope=require_identifier(body["scope"], "scope"),
            operation=require_identifier(body["operation"], "operation"),
            payload=dict(payload),
        )
