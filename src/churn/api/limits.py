"""Abuse controls: request-size caps, token auth and rate limiting.

This module exists because a publicly deployed demo API is *addressable by
anyone*, including scanners and runaway browser tabs. It adds three cheap,
dependency-free controls:

1. :class:`BodySizeLimitMiddleware` — reject oversized request bodies.
2. :func:`require_token` — constant-time bearer-token check for endpoints that
   must not be public.
3. :class:`RateLimiter` — an in-process token bucket per client.

Design decisions and their honest limitations
---------------------------------------------
**No new dependency.** ``slowapi``/``limits`` would pull in Redis semantics and a
dependency the free tier does not need; a single-process token bucket is enough
for a one-instance demo, and the repo is a teaching artifact where an explicit
implementation is more useful than a wrapper.

**The limiter is per process.** Render runs one container with one worker, so the
bucket is effectively global there. Behind a multi-instance deployment it would
be *per instance* — the limit would be N× looser. This is a documented
limitation, not a claim of distributed rate limiting.

**Client identity — do not trust arbitrary ``X-Forwarded-For``.** Any client can
send that header, so keying a bucket on it lets an attacker mint unlimited
identities and defeat the limiter entirely. We therefore only read the header
when :attr:`~churn.config.APIConfig.trusted_proxy_count` says a known proxy is in
front of us, and then we take the entry *added by that proxy* (counted from the
right of the list), never the leftmost client-supplied value.

**Fail-closed on unknown identity.** If no trustworthy identity can be derived,
all such requests share one bucket. An attacker cannot escape the limit by
removing headers.

**Time is monotonic.** ``time.monotonic`` cannot jump backwards, so a wall-clock
adjustment (NTP, daylight saving) cannot refill every bucket at once.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp

from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Bucket key used when a trustworthy client identity cannot be established.
UNKNOWN_CLIENT = "unknown"


def _is_multipart(request: Request) -> bool:
    """Whether the body is ``multipart/form-data``.

    Multipart bodies stream to disk beyond a threshold in python-multipart, so the
    in-memory cap does not apply to them. They are size-capped separately.
    """
    return request.headers.get("content-type", "").lower().startswith("multipart/")


def body_size_guard(request: Request, max_bytes: int) -> Response | None:
    """Return a 413 response when the declared body is too large, else ``None``.

    Two cases are handled, because checking ``Content-Length`` alone is
    trivial to bypass:

    * **``Content-Length`` present** — reject early, before buffering anything.
      A transfer-encoded body cannot be measured up front, so an oversized
      ``Content-Length`` is the only signal available and is checked first.
    * **``Content-Length`` absent** (``Transfer-Encoding: chunked``, or a client
      that simply omits it) — the header check cannot fire, so the *actual*
      buffered body is measured here. The middleware below reads the body, so a
      chunked request cannot slip past by omitting the header.

    Non-HTTP scopes are ignored so the same guard can be reused elsewhere.
    """
    if request.scope["type"] != "http":
        return None

    if _is_multipart(request):
        # Streamed to disk by the form parser; only the declared length is
        # meaningful. A missing declaration is accepted and bounded by the
        # server's own limits rather than by a number we cannot verify.
        _reject_if_declared_too_large(request, max_bytes)
        return None

    rejected = _reject_if_declared_too_large(request, max_bytes)
    if rejected is not None:
        return rejected

    # No (trustworthy) declaration: measure what was actually buffered.
    buffered = request._body if isinstance(getattr(request, "_body", None), bytes) else b""
    if len(buffered) > max_bytes:
        return _too_large(max_bytes)
    return None


def _reject_if_declared_too_large(request: Request, max_bytes: int) -> Response | None:
    """Reject on a declared ``Content-Length`` that exceeds the cap."""
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    try:
        length = int(declared)
    except ValueError:
        # A non-numeric Content-Length is malformed; the server would reject it,
        # but being explicit keeps the failure indistinguishable from "too big".
        return _too_large(max_bytes)
    if length > max_bytes:
        return _too_large(max_bytes)
    return None


def _too_large(max_bytes: int) -> Response:
    """The single 413 response shape, so probes cannot distinguish the cause."""
    return Response(
        content=f'{{"detail":"request body exceeds the {max_bytes} byte limit"}}',
        # Numeric on purpose: the named constant was renamed
        # (REQUEST_ENTITY_TOO_LARGE -> CONTENT_TOO_LARGE) and older starlette
        # releases only have the old name. 413 is 413.
        status_code=413,
        media_type="application/json",
    )


class _BodyTooLarge(Exception):
    """Internal signal: the streamed body exceeded the cap mid-read."""


def _contains_body_too_large(exc: BaseException, *, depth: int = 3) -> bool:
    """Whether ``exc`` is, or wraps, a :class:`_BodyTooLarge`.

    ``BaseHTTPMiddleware`` dispatches through an anyio TaskGroup, so an exception
    raised in a downstream ``receive`` call arrives nested inside one or more
    ``ExceptionGroup``s. ``isinstance`` alone would miss it and the intended 413
    would surface as an unhandled 500.

    Depth is bounded: a pathological nesting should not turn an error path into an
    unbounded walk.
    """
    if isinstance(exc, _BodyTooLarge):
        return True
    if depth <= 0:
        return False
    nested = getattr(exc, "exceptions", ()) or ()
    return any(_contains_body_too_large(item, depth=depth - 1) for item in nested)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes``.

    Implemented as **pure ASGI middleware**, not ``BaseHTTPMiddleware``, and that
    distinction is load-bearing. ``BaseHTTPMiddleware`` exposes the body only by
    reading ``receive``, which *consumes* it: the downstream app then sees an empty
    stream. Re-attaching ``request._body`` on the middleware's own ``Request``
    object does not help, because the downstream app builds a fresh ``Request``
    from the same already-drained channel — the result is a wall of 422s for
    perfectly valid requests.

    Working at the ASGI level gives direct access to ``receive``, so a body with no
    ``Content-Length`` can be counted as it streams and still be delivered intact
    when it is within the limit.

    Order of checks:

    1. **Declared ``Content-Length`` over the cap** — reject immediately, before a
       single byte is read.
    2. **Declared length within the cap** — pass through untouched (fast path; the
       common case costs one integer comparison).
    3. **No declared length** (``Transfer-Encoding: chunked``, or a client that
       simply omits the header) — wrap ``receive`` and count the bytes actually
       read. A cap that trusts ``Content-Length`` alone is trivially bypassed by
       omitting the header, which is exactly the case this branch exists for.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    # -- helpers ---------------------------------------------------------- #
    async def _send_too_large(self, send: Any) -> None:
        payload = json.dumps(
            {"detail": f"request body exceeds the {self.max_bytes} byte limit"}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        declared = headers.get("content-length")
        content_type = headers.get("content-type", "")

        # Multipart bodies are spooled to disk by the form parser beyond a
        # threshold; only the declared length is meaningful, and a missing
        # declaration is bounded by the server's own limits.
        if content_type.lower().startswith("multipart/"):
            if declared is not None:
                try:
                    if int(declared) > self.max_bytes:
                        await self._send_too_large(send)
                        return
                except ValueError:
                    pass
            await self.app(scope, receive, send)
            return

        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                # A non-numeric Content-Length is malformed. Reject it without
                # distinguishing the cause from "too large".
                await self._send_too_large(send)
                return
            if length > self.max_bytes:
                await self._send_too_large(send)
                return
            # Within the cap: forward unchanged, no buffering, no extra copy.
            await self.app(scope, receive, send)
            return

        # --- No Content-Length: count the streamed bytes. ----------------- #
        received = 0
        response_started = False

        async def limited_send(message: MutableMapping[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, limited_send)
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless ours
            # Any downstream `@app.middleware("http")` layer is a
            # BaseHTTPMiddleware, which runs the app inside an anyio TaskGroup and
            # re-raises whatever escapes wrapped in an ExceptionGroup. Checking
            # only for a bare `_BodyTooLarge` therefore turns this 413 into a 500
            # as soon as one such middleware exists — which is always, in the real
            # application. Unwrap first, then decide.
            if not _contains_body_too_large(exc) or response_started:
                raise
            # Only replace the response if nothing has been sent yet; once the
            # status line is on the wire it cannot be changed.
            await self._send_too_large(send)


# --------------------------------------------------------------------------- #
# Token authentication
# --------------------------------------------------------------------------- #
def require_token(request: Request) -> None:
    """FastAPI dependency: require a matching ``Authorization: Bearer`` token.

    Behaviour:

    * ``metrics_token`` unset (``None`` or empty) — the endpoint behaves as if it
      does not exist (**404**). Opt-in protection: local development and the
      existing test suite are unaffected, and a deployment that forgets to set a
      token does not silently expose an unauthenticated monitoring interface.
    * Token set, header missing or wrong — **401** with ``WWW-Authenticate``.

    Comparisons use :func:`hmac.compare_digest`, which is constant-time for
    equal-length inputs, so response timing does not reveal how much of a guessed
    token was correct.
    """
    expected = request.app.state.settings.api.metrics_token
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # compare_digest requires both operands to be bytes or both str of ASCII-only.
    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
@dataclass
class _Bucket:
    """One client's token bucket.

    ``last_seen`` is separate from ``updated`` on purpose: ``updated`` moves when
    tokens are *added*, while ``last_seen`` only moves when the client actually
    makes a request. Pruning on ``last_seen`` is what bounds memory — an attacker
    cycling identities would otherwise leave a permanently-refilled bucket behind
    for every identity it invented.
    """

    tokens: float
    updated: float
    last_seen: float = 0.0


@dataclass
class RateLimiter:
    """In-process token bucket, keyed by client identity.

    ``capacity`` requests are allowed as a burst; the bucket then refills at
    ``capacity / window_seconds`` tokens per second. Memory is bounded by pruning
    full buckets: a bucket that has refilled completely is indistinguishable from
    a new one, so dropping it is free — which keeps a rotating-identity attacker
    from growing the dict without limit.
    """

    capacity: int
    window_seconds: float
    buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Prune at most this often, to keep the hot path cheap.
    _last_prune: float = field(default=0.0, repr=False)

    @property
    def rate_per_second(self) -> float:
        """Tokens added per second.

        Returns ``0.0`` for a zero capacity rather than dividing by zero: a bucket
        that can never hold a token is fail-closed, and the caller needs a finite
        retry interval, not an exception on the request path.
        """
        if self.window_seconds <= 0 or self.capacity <= 0:
            return 0.0
        return self.capacity / self.window_seconds

    def _prune(self, now: float) -> None:
        """Drop buckets untouched for a full window.

        A client that has not been seen for a whole window would be indistinguishable
        from a new one anyway (its bucket refills completely in that time), so
        forgetting it costs nothing and bounds memory against identity cycling.
        """
        if now - self._last_prune < self.window_seconds:
            return
        self._last_prune = now
        cutoff = now - self.window_seconds
        stale = [key for key, bucket in self.buckets.items() if bucket.last_seen < cutoff]
        for key in stale:
            # Re-check under the lock: another call may have touched it meanwhile.
            bucket = self.buckets.get(key)
            if bucket is not None and bucket.last_seen < cutoff:
                self.buckets.pop(key, None)

    def consume(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Take one token. Returns ``(allowed, retry_after_seconds)``."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)

            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated=moment, last_seen=moment)
                self.buckets[key] = bucket

            bucket.last_seen = moment
            elapsed = max(0.0, moment - bucket.updated)
            bucket.tokens = min(
                float(self.capacity), bucket.tokens + elapsed * self.rate_per_second
            )
            bucket.updated = moment

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            deficit = 1.0 - bucket.tokens
            rate = self.rate_per_second
            if rate <= 0.0:
                # Never refills (capacity 0 or a non-positive window). Report one
                # window as the retry hint instead of dividing by zero.
                return False, float(self.window_seconds) or 60.0
            return False, deficit / rate


def client_identity(request: Request, trusted_proxy_count: int) -> str:
    """Derive a rate-limit key that a client cannot forge.

    ``X-Forwarded-For`` is a comma-separated list appended to by each proxy. Only
    the entries added by *our* proxies are trustworthy, and they are the ones on
    the right. With ``trusted_proxy_count=1`` (one Render proxy in front of us) the
    real client is the last entry. With ``0`` the header is ignored entirely and
    the socket peer address is used.

    The header is never parsed when ``trusted_proxy_count <= 0``: trusting it
    unconditionally is the bug this function exists to avoid.
    """
    if trusted_proxy_count > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            # The rightmost entry was appended by the proxy nearest to us, so it
            # is the one that proxy actually saw. Missing hops means the header
            # was shorter than expected; fall back to the socket peer.
            index = len(hops) - trusted_proxy_count
            if 0 <= index < len(hops):
                return hops[index]

    peer = request.client.host if request.client else None
    return peer or UNKNOWN_CLIENT


def enforce_rate_limit(
    request: Request,
    *,
    limiter: RateLimiter,
    trusted_proxy_count: int,
    rule: str,
) -> Response | None:
    """Return a 429 response when this client has exhausted its bucket, else ``None``.

    **Returns** rather than raises, deliberately. This is called from
    ``BaseHTTPMiddleware.dispatch``, where a raised ``HTTPException`` is *not*
    converted into a response by FastAPI's exception handlers — it propagates out
    of the middleware stack as an unhandled server error. Returning the response
    makes the behaviour identical whether the caller is a middleware or a
    dependency.

    Different rules get different keys, so exhausting the expensive explanation
    budget cannot lock a client out of cheap predictions.
    """
    identity = client_identity(request, trusted_proxy_count)
    allowed, retry_after = limiter.consume(f"{rule}:{identity}")
    if allowed:
        return None

    # Log the identity so an operator can spot abuse, but never log headers or
    # any part of the body.
    logger.warning("rate limit exceeded", extra={"rule": rule, "client": identity})
    seconds = max(1, int(retry_after + 0.999))
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"rate limit exceeded for {rule}; retry in {seconds}s"},
        headers={"Retry-After": str(seconds)},
    )


def new_token(n_bytes: int = 32) -> str:
    """Generate a URL-safe random token (helper for the ops docs and tests)."""
    return secrets.token_urlsafe(n_bytes)
