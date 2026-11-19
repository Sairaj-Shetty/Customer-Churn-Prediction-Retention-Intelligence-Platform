"""Tests for /health (liveness) and /ready (readiness).

The distinction under test is the one that matters operationally: a platform
liveness probe must not fail because a *dependency* is down, or the container will
be restarted in a loop that cannot possibly help.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from churn.api.readiness import CACHE_TTL_SECONDS, reset_cache
from churn.db import session as session_module
from churn.db.base import Base

pytestmark = pytest.mark.slow


@pytest.fixture
def client_healthy(tmp_path: Path):
    """App whose database is reachable."""
    engine = create_engine(f"sqlite:///{tmp_path / 'ok.db'}")
    Base.metadata.create_all(engine)
    session_module.reset_for_tests(engine)
    reset_cache()

    from churn.api.main import create_app

    with TestClient(create_app()) as client:
        yield client

    engine.dispose()
    session_module.reset_for_tests(None)


@pytest.fixture
def client_db_down(monkeypatch):
    """App whose database cannot be reached."""
    from churn.config import get_settings

    monkeypatch.setenv("CHURN_DATABASE__URL", "sqlite:////nonexistent-readiness/x.db")
    get_settings.cache_clear()
    session_module.reset_for_tests(None)
    reset_cache()

    from churn.api.main import create_app

    with TestClient(create_app()) as client:
        yield client

    get_settings.cache_clear()
    session_module.reset_for_tests(None)
    reset_cache()


@pytest.fixture
def client_no_persistence(monkeypatch, tmp_path: Path):
    """Persistence switched off: the database is not a dependency at all."""
    from churn.config import get_settings

    engine = create_engine(f"sqlite:///{tmp_path / 'off.db'}")
    Base.metadata.create_all(engine)
    session_module.reset_for_tests(engine)
    reset_cache()
    monkeypatch.setenv("CHURN_DATABASE__PERSIST_PREDICTIONS", "false")
    get_settings.cache_clear()

    from churn.api.main import create_app

    with TestClient(create_app()) as client:
        yield client

    get_settings.cache_clear()
    engine.dispose()
    session_module.reset_for_tests(None)


class TestLiveness:
    def test_healthy(self, client_healthy):
        response = client_healthy.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["model_loaded"] is True
        assert payload["version"].startswith("phase4-calibrated-")

    def test_does_not_depend_on_database(self, client_db_down):
        """A dependency outage must not make the process look dead."""
        response = client_db_down.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_is_cheap(self, client_healthy):
        """Liveness must not import shap or touch the DB (cold-start cost)."""
        body = client_healthy.get("/health").text
        assert "database" not in body.lower()


class TestReadiness:
    def test_ready_when_everything_up(self, client_healthy):
        response = client_healthy.get("/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ready"
        assert payload["components"]["model"]["ok"] is True
        assert payload["components"]["database"]["ok"] is True
        assert payload["prediction_available"] is True

    def test_reports_database_detail(self, client_healthy):
        payload = client_healthy.get("/ready").json()
        assert payload["database"]["dialect"] == "sqlite"
        assert payload["database"]["reachable"] is True

    def test_degraded_when_database_down(self, client_db_down):
        response = client_db_down.get("/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["components"]["database"]["ok"] is False
        assert payload["components"]["model"]["ok"] is True

    def test_error_is_a_class_name_not_a_dsn(self, client_db_down):
        """Never leak host, user or password into a probe response."""
        detail = client_db_down.get("/ready").json()["components"]["database"]["detail"]
        assert detail == "OperationalError"
        assert "@" not in detail

    def test_predictions_still_available_when_database_down(self, client_db_down):
        """The whole point of best-effort persistence."""
        assert client_db_down.get("/ready").json()["prediction_available"] is True

    def test_persistence_block_declares_best_effort(self, client_healthy):
        persistence = client_healthy.get("/ready").json()["persistence"]
        assert persistence["mode"] == "best_effort_async"
        assert persistence["guaranteed"] is False

    def test_disabled_persistence_is_not_a_dependency(self, client_no_persistence):
        response = client_no_persistence.get("/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["components"]["database"]["ok"] is True
        assert payload["persistence"]["enabled"] is False


class TestReadinessCaching:
    def test_second_call_is_cached(self, client_healthy):
        reset_cache()
        first = client_healthy.get("/ready").json()
        second = client_healthy.get("/ready").json()
        assert first["check_cached"] is False
        assert second["check_cached"] is True

    def test_cache_expires(self, client_healthy, monkeypatch):
        from churn.api import readiness

        reset_cache()
        client_healthy.get("/ready")
        # Pretend the TTL elapsed.
        monkeypatch.setattr(readiness, "CACHE_TTL_SECONDS", -1.0)
        assert client_healthy.get("/ready").json()["check_cached"] is False

    def test_cache_avoids_keeping_neon_awake(self, client_healthy):
        """Caching exists so health probes do not burn free-tier CU-hours."""
        assert CACHE_TTL_SECONDS >= 5


class TestProbeWiring:
    def test_render_healthcheck_uses_liveness_not_readiness(self, project_root: Path):
        """Binding the platform probe to /ready would restart on every Neon idle.

        A suspended database is not a broken container, and restarting cannot fix
        it — so liveness must stay database-free.
        """
        blueprint = (project_root / "render.yaml").read_text(encoding="utf-8")
        assert "healthCheckPath: /health" in blueprint
        assert "healthCheckPath: /ready" not in blueprint

    def test_openapi_documents_both_probes(self, client_healthy):
        paths = set(client_healthy.get("/openapi.json").json()["paths"])
        assert {"/health", "/ready"} <= paths
