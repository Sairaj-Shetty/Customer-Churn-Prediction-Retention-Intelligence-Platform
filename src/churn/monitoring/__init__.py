"""Production drift monitoring (Phase 10).

Aggregate-only by design: no customer feature vectors are stored. See
``docs/08_DRIFT.md`` for the collection architecture and its limits.
"""

from churn.monitoring.schema import MonitoringSchema

__all__ = ["MonitoringSchema"]
