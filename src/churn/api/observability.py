"""Operational metrics — kept strictly separate from ML evaluation metrics.

The distinction matters and is easy to blur:

* **Operational metrics** (this module) describe *the service*: how many requests,
  how fast, how many failed. They are counters and timings that change every
  second and mean nothing about model quality.
* **ML evaluation metrics** (``/model/info``) describe *the model*: PR-AUC, Brier
  score, recall, business cost. They are fixed at training time and never change
  while the service runs.

Mixing them into one response would invite the classic reporting error of quoting
a live request count as if it measured model performance. They are served by
separate endpoints from separate data structures.

Why not Prometheus
------------------
``prometheus-client`` is not a dependency of this project, and adding one to serve
four counters on a 512 MB free tier is not justified. This module keeps a small,
**bounded** in-process registry instead:

* latency is recorded as an online mean, plus min/max/total — no unbounded list
  that would leak memory over a long-lived process;
* per-endpoint counters are capped so a stream of unique paths cannot grow the
  dict without limit;
* all access is guarded by a single lock, since FastAPI runs sync handlers in a
  thread pool and these counters are mutated concurrently.

The trade-off is explicit: these numbers are **per-process** and reset on
restart. With a single worker (required on the free tier) that is the whole truth.
With multiple workers the endpoint would report only the worker that answered.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Refuse to track more than this many distinct endpoints. Guards against
#: unbounded dict growth if a client probes random paths.
MAX_TRACKED_ENDPOINTS = 50


@dataclass
class EndpointStats:
    """Counters for one endpoint."""

    requests: int = 0
    errors: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0
    last_status: int = 0

    def observe(self, duration: float, status_code: int) -> None:
        self.requests += 1
        self.total_seconds += duration
        self.min_seconds = min(self.min_seconds, duration)
        self.max_seconds = max(self.max_seconds, duration)
        self.last_status = status_code
        if status_code >= 400:
            self.errors += 1

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": round(self.errors / self.requests, 4) if self.requests else 0.0,
            "mean_latency_ms": round(self.mean_seconds * 1000, 2),
            "min_latency_ms": round(self.min_seconds * 1000, 2)
            if self.min_seconds != float("inf")
            else 0.0,
            "max_latency_ms": round(self.max_seconds * 1000, 2),
            "last_status": self.last_status,
        }


@dataclass
class MetricsRegistry:
    """Thread-safe, bounded, in-process operational counters."""

    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _endpoints: dict[str, EndpointStats] = field(default_factory=dict, repr=False)
    _predictions: dict[str, int] = field(
        default_factory=lambda: {"requests": 0, "customers": 0, "churn_flagged": 0},
        repr=False,
    )
    _explanations: dict[str, int] = field(
        default_factory=lambda: {"requests": 0, "customers": 0}, repr=False
    )
    _validation_rejections: dict[str, int] = field(default_factory=dict, repr=False)
    _persistence: dict[str, float] = field(
        default_factory=lambda: {
            "attempted": 0.0,
            "succeeded": 0.0,
            "duplicates": 0.0,
            "failed": 0.0,
            "retries": 0.0,
            "total_seconds": 0.0,
            "last_error_epoch": 0.0,
        },
        repr=False,
    )

    # ------------------------------------------------------------------ #
    def record_request(self, endpoint: str, duration: float, status_code: int) -> None:
        """Record one HTTP request against an endpoint."""
        with self._lock:
            key = endpoint if endpoint in self._endpoints else endpoint
            if key not in self._endpoints and len(self._endpoints) >= MAX_TRACKED_ENDPOINTS:
                key = "__other__"
            self._endpoints.setdefault(key, EndpointStats()).observe(duration, status_code)

    def record_predictions(self, count: int, flagged: int) -> None:
        """Record a prediction batch (``count`` customers, ``flagged`` positives)."""
        with self._lock:
            self._predictions["requests"] += 1
            self._predictions["customers"] += count
            self._predictions["churn_flagged"] += flagged

    def record_explanations(self, count: int) -> None:
        """Record an explanation batch."""
        with self._lock:
            self._explanations["requests"] += 1
            self._explanations["customers"] += count

    def record_persistence(
        self,
        *,
        attempted: int,
        succeeded: int,
        duplicates: int,
        failed: int,
        retries: int = 0,
        error: str | None = None,
        duration: float = 0.0,
    ) -> None:
        """Record one persistence attempt (background write)."""
        with self._lock:
            self._persistence["attempted"] += attempted
            self._persistence["succeeded"] += succeeded
            self._persistence["duplicates"] += duplicates
            self._persistence["failed"] += failed
            self._persistence["retries"] += retries
            self._persistence["total_seconds"] += duration
            if error:
                self._persistence["last_error_epoch"] = time.time()

    def record_validation_rejection(self, field_name: str) -> None:
        """Record a 422, keyed by the field that failed.

        Field names are attacker-controlled input, so the keys are capped and
        anything beyond the first 50 distinct names is bucketed.
        """
        with self._lock:
            key = field_name if len(self._validation_rejections) < 50 else "__other__"
            self._validation_rejections[key] = self._validation_rejections.get(key, 0) + 1

    def reset(self) -> None:
        """Clear all counters **in place**.

        Rebinding the module-level ``METRICS`` would be a trap: every module that
        did ``from ... import METRICS`` keeps a reference to the original object,
        so a rebinding reset is invisible to them and counters silently keep
        accumulating across tests (and, in principle, across a reload). Mutating
        in place means reset works no matter how the registry was imported.
        """
        with self._lock:
            self.started_at = time.time()
            self._endpoints.clear()
            self._predictions = {"requests": 0, "customers": 0, "churn_flagged": 0}
            self._explanations = {"requests": 0, "customers": 0}
            self._validation_rejections.clear()
            for key in self._persistence:
                self._persistence[key] = 0.0

    def snapshot(
        self, *, model_loaded: bool, uptime_seconds: float | None = None
    ) -> dict[str, Any]:
        """Immutable view for the ``/metrics`` endpoint."""
        with self._lock:
            endpoints = {name: stats.to_dict() for name, stats in self._endpoints.items()}
            predictions = dict(self._predictions)
            explanations = dict(self._explanations)
            rejections = dict(self._validation_rejections)
            persistence = dict(self._persistence)
            started = self.started_at

        uptime = uptime_seconds if uptime_seconds is not None else time.time() - started
        total_requests = sum(item["requests"] for item in endpoints.values())
        total_errors = sum(item["errors"] for item in endpoints.values())

        return {
            "scope": "process",
            "note": (
                "Operational metrics only. These describe service health and reset "
                "on restart. They are NOT model quality metrics — see /model/info "
                "for PR-AUC, Brier score and business cost."
            ),
            "uptime_seconds": round(uptime, 1),
            "model_loaded": model_loaded,
            "requests": {
                "total": total_requests,
                "errors": total_errors,
                "error_rate": round(total_errors / total_requests, 4) if total_requests else 0.0,
                "by_endpoint": endpoints,
            },
            "predictions": {
                **predictions,
                "mean_customers_per_request": round(
                    predictions["customers"] / predictions["requests"], 2
                )
                if predictions["requests"]
                else 0.0,
            },
            "explanations": explanations,
            "validation_rejections": {
                "total": sum(rejections.values()),
                "by_field": rejections,
            },
            "persistence": {
                "mode": "best_effort_async",
                "attempted": int(persistence["attempted"]),
                "succeeded": int(persistence["succeeded"]),
                "duplicates_suppressed": int(persistence["duplicates"]),
                "failed": int(persistence["failed"]),
                "retries": int(persistence["retries"]),
                "failure_rate": round(persistence["failed"] / persistence["attempted"], 4)
                if persistence["attempted"]
                else 0.0,
                "mean_write_ms": round(
                    persistence["total_seconds"] / max(persistence["attempted"], 1) * 1000, 2
                ),
                "last_error_seconds_ago": round(time.time() - persistence["last_error_epoch"], 1)
                if persistence["last_error_epoch"]
                else None,
                "note": (
                    "Best-effort writes issued after the response. Not guaranteed; "
                    "pending writes are lost on process exit."
                ),
            },
        }


#: Process-wide registry. Tests reset it between cases via ``reset_metrics``.
METRICS = MetricsRegistry()


def reset_metrics() -> None:
    """Clear all counters (test helper, and harmless in production).

    Delegates to :meth:`MetricsRegistry.reset`, which mutates the existing object
    so that modules holding a direct reference still see the cleared state.
    """
    METRICS.reset()
