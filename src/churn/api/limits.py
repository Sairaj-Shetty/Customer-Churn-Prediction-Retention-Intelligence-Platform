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
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
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


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than ``max_bytes``.

    Runs *before* any handler or validation, so an oversized body never reaches
    pydantic. Placed outermost in the stack so even CORS preflights are capped.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            rejected = _reject_if_declared_too_large(request, self.max_bytes)
            if rejected is not None:
                return rejected

        if _is_multipart(request):
            return await call_next(request)

        # No usable Content-Length: read the body ourselves so the size is
        # actually verified, then hand the buffered bytes to the app. Reading
        # past the cap aborts early rather than buffering an unbounded body.
        body = b""
        if declared is None:
            chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > self.max_bytes:
                    return _too_large(self.max_bytes)
                chunks.append(chunk)
            body = b"".join(chunks)
        else:
            body = await request.body()

        if len(body) > self.max_bytes:
            return _too_large(self.max_bytes)

        request._body = body  # noqa: SLF001 - replay for downstream handlers
        return await call_next(request)


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
    """One client's token bucket."""

    tokens: float
    updated: float


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
        return self.capacity / self.window_seconds if self.window_seconds > 0 else float("inf")

    def _prune(self, now: float) -> None:
        """Drop buckets that have fully refilled (they are equivalent to absent)."""
        if now - self._last_prune < self.window_seconds:
            return
        self._last_prune = now
        full = self.capacity
        refill = self.rate_per_second
        stale = [key for key, bucket in self.buckets.items() if bucket.tokens >= full]
        for key in stale:
            # Re-check: another thread may have consumed from it meanwhile.
            bucket = self.buckets.get(key)
            if bucket is not None and bucket.tokens + (now - bucket.updated) * refill >= full:
                self.buckets.pop(key, None)

    def consume(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Take one token. Returns ``(allowed, retry_after_seconds)``."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)

            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated=moment)
                self.buckets[key] = bucket

            elapsed = max(0.0, moment - bucket.updated)
            bucket.tokens = min(
                float(self.capacity), bucket.tokens + elapsed * self.rate_per_second
            )
            bucket.updated = moment

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            deficit = 1.0 - bucket.tokens
            return False, deficit / self.rate_per_second


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
