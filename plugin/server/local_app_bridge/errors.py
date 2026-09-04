from __future__ import annotations


class LocalAppBridgeError(Exception):
    """A transport-safe bridge error which never carries secret material."""

    __slots__ = ("code", "status_code", "message", "retry_after")

    def __init__(
        self,
        code: str,
        status_code: int,
        message: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after

    def __str__(self) -> str:
        return self.message


def bad_request(
    code: str = "invalid_request", message: str = "Invalid request"
) -> LocalAppBridgeError:
    return LocalAppBridgeError(code=code, status_code=400, message=message)


def unauthorized(code: str = "unauthorized") -> LocalAppBridgeError:
    return LocalAppBridgeError(
        code=code, status_code=401, message="Authentication failed"
    )


def forbidden(code: str = "forbidden") -> LocalAppBridgeError:
    return LocalAppBridgeError(
        code=code, status_code=403, message="Request is not permitted"
    )
