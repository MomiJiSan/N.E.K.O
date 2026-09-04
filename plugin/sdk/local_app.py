"""Host-only local application invocation contracts for plugin adapters.

These handlers are intentionally separate from plugin entries, custom events,
and LLM tools.  Merely decorating a method does not make it remotely callable:
the host must also register an explicit app/scope/operation policy and target.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

TRUSTED_LOCAL_APP_META_ATTR = "__neko_trusted_local_app_operation__"
_OPERATION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
_CANONICAL_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class TrustedLocalAppPluginContext:
    """Identity asserted by the host after local-app session validation."""

    app_id: str
    client_id: str
    session_id: str
    scope: str
    operation: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TrustedLocalAppPluginContext:
        required = {"app_id", "client_id", "session_id", "scope", "operation"}
        if set(value) != required:
            raise ValueError("trusted local app context fields are invalid")
        fields: dict[str, str] = {}
        for name in required:
            raw = value[name]
            if not isinstance(raw, str) or _OPERATION_RE.fullmatch(raw) is None:
                raise ValueError(f"trusted local app context {name} is invalid")
            fields[name] = raw
        return cls(**fields)


def trusted_local_app_operation(operation: str) -> Callable[[_F], _F]:
    """Mark an async adapter method as a host-only local-app operation."""

    if (
        not isinstance(operation, str)
        or _CANONICAL_OPERATION_RE.fullmatch(operation) is None
    ):
        raise ValueError("operation must be a valid identifier")

    def decorator(func: _F) -> _F:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("trusted local app operation must be async")
        setattr(func, TRUSTED_LOCAL_APP_META_ATTR, operation)
        return func

    return decorator


def get_trusted_local_app_operation(member: object) -> str | None:
    """Return trusted operation metadata through bound/wrapped callables."""

    candidates: list[object] = [member]
    func = getattr(member, "__func__", None)
    if func is not None:
        candidates.append(func)
    current = getattr(member, "__wrapped__", None)
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidates.append(current)
        wrapped_func = getattr(current, "__func__", None)
        if wrapped_func is not None:
            candidates.append(wrapped_func)
        current = getattr(current, "__wrapped__", None)
    for candidate in candidates:
        operation = getattr(candidate, TRUSTED_LOCAL_APP_META_ATTR, None)
        if isinstance(operation, str) and operation:
            return operation
    return None


def collect_trusted_local_app_operations(
    instance: object,
) -> dict[str, Callable[..., Any]]:
    """Collect a plugin's dedicated trusted handlers, rejecting duplicates."""

    operations: dict[str, Callable[..., Any]] = {}
    for _name, member in inspect.getmembers(instance, predicate=callable):
        operation = get_trusted_local_app_operation(member)
        if operation is None:
            continue
        if operation in operations:
            raise ValueError(f"duplicate trusted local app operation: {operation}")
        if not inspect.iscoroutinefunction(member):
            raise TypeError(f"trusted local app operation must be async: {operation}")
        operations[operation] = member
    return operations
