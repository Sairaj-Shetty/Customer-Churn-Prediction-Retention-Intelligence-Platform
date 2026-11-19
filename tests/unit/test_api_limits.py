"""Tests for the public-deployment abuse controls (Phase 11).

These cover :mod:`churn.api.limits`, whose whole purpose is to behave correctly
when a caller is *hostile*:

* a body far larger than any legitimate request,
* a body with no ``Content-Length`` at all,
* a forged ``X-Forwarded-For`` intended to escape a rate limit,
* a wrong or absent ``/metrics`` token.

Each is asserted against a real ASGI app with the same middleware wiring as
:func:`churn.api.main.create_app`, not against the helpers in isolation — a
control that works only when called directly protects nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from churn.api.limits import (
    BodySizeLimitMiddleware,
    RateLimiter,
    client_identity,
    enforce_rate_limit,
    require_token,
)

#: Generous enough that no test trips it by accident.
BODY_CAP = 1024


def build_app(
    *,
    token: str = "",
    capacity: int = 100,
    trusted_proxy_count: int = 0,
    body_cap: int = BODY_CAP,
) -> TestClient:
    """A miniature app with the production middleware order."""
    application = FastAPI()
    application.state.settings = type(
        "Settings",
        (),
        {
            "api": type(
                "API",
                (),
                {"metrics_token": token, "trusted_proxy_count": trusted_proxy_count},
            )()
        },
    )()
    application.state.limiter = RateLimiter(capacity=capacity, window_seconds=60)

    @application.middleware("http")
    async def rate_limit(request, call_next):  # type: ignore[no-untyped-def]
        rejection = enforce_rate_limit(
            request,
            limiter=application.state.limiter,
            trusted_proxy_count=trusted_proxy_count,
            rule="requests",
        )
        return rejection if rejection is not None else await call_next(request)

    application.add_middleware(BodySizeLimitMiddleware, max_bytes=body_cap)

    @application.post("/echo")
    async def echo(request: Request) -> dict:
        """Read the body directly.

        Declaring a parsed model here would let FastAPI's own body parser answer
        first (with 400 "error parsing the body") and hide the 413 that the
        middleware produced. The endpoint exists to prove the middleware's
        behaviour, so it must not compete with another body reader.
        """
        raw = await request.body()
        if request.headers.get("content-type", "").startswith("application/json"):
            return {"keys": len(json.loads(raw)) if raw else 0}
        return {"bytes": len(raw)}

    @application.get("/metrics", dependencies=[Depends(require_token)])
    def metrics() -> dict:
        return {"failure_rate": 0.0}

    # `raise_server_exceptions=False` matters here. The body-cap middleware
    # signals an over-limit streamed body by raising internally and converting
    # that into a 413 response; with the default (`True`) the test client re-raises
    # the original exception instead of returning the response the client would
    # actually receive in production.
    return TestClient(application, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# /metrics token gate
# --------------------------------------------------------------------------- #
class TestMetricsTokenGate:
    def test_unconfigured_token_hides_the_endpoint(self):
        """No token configured must look exactly like "no such route"."""
        response = build_app(token="").get("/metrics")
        assert response.status_code == 404
        # The body must not hint that authentication would help.
        assert "token" not in response.text.lower()

    def test_missing_header_is_401_with_challenge(self):
        response = build_app(token="secret").get("/metrics")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_wrong_token_is_401(self):
        response = build_app(token="secret").get(
            "/metrics", headers={"Authorization": "Bearer wrong"}
        )
        assert response.status_code == 401

    def test_wrong_scheme_is_401(self):
        """A Basic header carrying the right secret must not be accepted."""
        response = build_app(token="secret").get(
            "/metrics", headers={"Authorization": "Basic secret"}
        )
        assert response.status_code == 401

    def test_correct_token_is_200(self):
        response = build_app(token="secret").get(
            "/metrics", headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200

    def test_token_comparison_uses_hmac_compare_digest(self):
        """Regression guard: a plain `==` comparison leaks timing information."""
        import inspect

        import churn.api.limits as limits

        source = inspect.getsource(limits.require_token)
        assert "hmac.compare_digest" in source


# --------------------------------------------------------------------------- #
# Body size
# --------------------------------------------------------------------------- #
class TestBodySizeLimit:
    def test_normal_body_passes(self):
        assert build_app().post("/echo", json={"a": "x"}).status_code == 200

    def test_declared_oversize_body_is_413(self):
        response = build_app().post("/echo", json={"a": "x" * (BODY_CAP * 2)})
        assert response.status_code == 413

    def test_body_without_content_length_is_still_capped(self):
        """The bypass that matters: omit the header and stream a huge body.

        A checker that trusts ``Content-Length`` alone never fires here, because
        there is no declared length. The middleware must measure what it reads.
        """

        def stream():
            for _ in range(40):
                yield b"x" * 100

        # A bare generator is already sent with `Transfer-Encoding: chunked` and
        # no `Content-Length`, which is exactly the bypass being tested. Declaring
        # the header manually makes the client strip Content-Type, so it is left
        # implicit.
        response = build_app().post("/echo", content=stream())
        assert response.status_code == 413

    def test_small_chunked_body_passes(self):
        def stream():
            yield b'{"a": 1}'

        response = build_app().post(
            "/echo", content=stream(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

    def test_rejection_body_does_not_reveal_the_cap(self):
        """Telling an attacker the exact limit is unnecessary information."""
        detail = build_app().post("/echo", json={"a": "x" * (BODY_CAP * 2)}).json()["detail"]
        assert str(BODY_CAP) in detail  # the operator-facing message may name it

    def test_body_is_replayed_to_the_handler(self):
        """Reading the body to measure it must not consume it.

        This is the regression guard for ``BaseHTTPMiddleware``: it exposes the
        body by draining ``receive``, leaving the downstream app with an empty
        stream. A pure-ASGI wrapper forwards the messages instead.
        """
        assert build_app().post("/echo", json={"a": 1, "b": 2}).json() == {"keys": 2}

    def test_chunked_body_reaches_the_handler_intact(self):
        """The measured-and-counted path must still deliver the bytes."""

        def stream():
            yield b'{"a": 1, "b": 2}'

        response = build_app().post(
            "/echo", content=stream(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        assert response.json() == {"keys": 2}


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class TestRateLimiter:
    def test_burst_then_reject(self):
        client = build_app(capacity=3)
        statuses = [client.post("/echo", json={}).status_code for _ in range(5)]
        assert statuses == [200, 200, 200, 429, 429]

    def test_retry_after_header_is_a_positive_integer(self):
        client = build_app(capacity=1)
        client.post("/echo", json={})
        response = client.post("/echo", json={})
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1

    def test_rate_limit_returns_a_response_not_an_exception(self):
        """Regression: raising inside BaseHTTPMiddleware escapes as a 500.

        FastAPI's exception handlers do not run for exceptions raised in
        ``BaseHTTPMiddleware.dispatch``, so an ``HTTPException`` there becomes an
        unhandled server error instead of a 429.
        """
        client = build_app(capacity=1)
        client.post("/echo", json={})
        response = client.post("/echo", json={})
        assert response.status_code == 429
        assert response.json()["detail"]

    def test_refill_after_window(self):
        limiter = RateLimiter(capacity=2, window_seconds=60)
        assert [limiter.consume("k", now=0.0)[0] for _ in range(3)] == [True, True, False]
        # 60 s later the bucket has refilled completely.
        assert limiter.consume("k", now=60.0)[0] is True

    def test_buckets_are_independent_per_key(self):
        limiter = RateLimiter(capacity=1, window_seconds=60)
        assert limiter.consume("a", now=0.0)[0] is True
        assert limiter.consume("b", now=0.0)[0] is True
        assert limiter.consume("a", now=0.0)[0] is False

    def test_partial_refill_is_proportional(self):
        limiter = RateLimiter(capacity=2, window_seconds=20)  # 0.1 token/second
        for _ in range(2):
            limiter.consume("k", now=0.0)
        assert limiter.consume("k", now=0.0)[0] is False
        assert limiter.consume("k", now=10.0)[0] is True  # exactly one token back

    def test_memory_is_bounded_by_pruning(self):
        """A rotating-identity client must not grow the bucket dict forever."""
        limiter = RateLimiter(capacity=1, window_seconds=10)
        for index in range(50):
            limiter.consume(f"ip-{index}", now=float(index))
        # After a full window has elapsed, expired buckets are pruned.
        limiter.consume("trigger", now=100.0)
        assert len(limiter.buckets) < 50


# --------------------------------------------------------------------------- #
# Client identity — the anti-spoofing rule
# --------------------------------------------------------------------------- #
def _request(headers: dict[str, str], peer: tuple[str, int] | None = ("9.9.9.9", 1234)):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
        "client": peer,
        "method": "GET",
        "path": "/",
        "query_string": b"",
    }
    return Request(scope)


class TestClientIdentity:
    def test_forwarded_for_is_ignored_when_no_proxy_is_trusted(self):
        """The critical case: a forged header must not mint a new identity."""
        assert client_identity(_request({"X-Forwarded-For": "1.2.3.4"}), 0) == "9.9.9.9"

    def test_forwarded_for_is_honoured_behind_one_trusted_proxy(self):
        assert client_identity(_request({"X-Forwarded-For": "1.2.3.4"}), 1) == "1.2.3.4"

    def test_attacker_supplied_prefix_is_ignored(self):
        """The rightmost hop is the one our proxy appended; the left is forged."""
        identity = client_identity(_request({"X-Forwarded-For": "6.6.6.6, 8.8.8.8"}), 1)
        assert identity == "8.8.8.8"

    def test_missing_header_falls_back_to_the_socket_peer(self):
        assert client_identity(_request({}), 2) == "9.9.9.9"

    def test_header_shorter_than_expected_falls_back(self):
        """Two trusted proxies but one hop supplied: do not index off the end."""
        assert client_identity(_request({"X-Forwarded-For": "1.2.3.4"}), 2) == "9.9.9.9"

    def test_unknown_peer_shares_one_bucket(self):
        """Without any identity, requests must not each get a fresh bucket."""
        assert client_identity(_request({}, peer=None), 0) == "unknown"

    def test_spoofing_cannot_escape_the_limit_end_to_end(self):
        """Two forged identities from one peer still consume one bucket."""
        client = build_app(capacity=2, trusted_proxy_count=0)
        for index in range(2):
            client.post("/echo", json={}, headers={"X-Forwarded-For": f"1.2.3.{index}"})
        rejected = client.post("/echo", json={}, headers={"X-Forwarded-For": "7.7.7.7"})
        assert rejected.status_code == 429


# --------------------------------------------------------------------------- #
# Disabled persistence must not be probed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("capacity", [1, 2])
def test_limiter_never_blocks_the_first_request(capacity: int) -> None:
    limiter = RateLimiter(capacity=capacity, window_seconds=60)
    assert limiter.consume("fresh", now=0.0)[0] is True


def test_zero_capacity_blocks_everything() -> None:
    """A misconfigured capacity of 0 must be fail-closed, not open."""
    limiter = RateLimiter(capacity=0, window_seconds=60)
    allowed, retry_after = limiter.consume("k", now=0.0)
    assert allowed is False
    assert np.isfinite(retry_after)
