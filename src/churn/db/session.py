"""Engine and session management.

Design notes that matter on Neon's free tier
--------------------------------------------
**The engine is created once per process**, not per request. Creating an engine
opens no connections (the pool fills lazily), so construction is cheap, but
repeating it would defeat pooling entirely.

**Pooling is deliberately small** (2 + 2 overflow). Neon suspends compute after
~5 minutes idle and bills in CU-hours; a large warm pool either keeps the compute
awake or holds sockets the server has already closed. Since writes are
best-effort, waiting briefly in the pool queue is the cheaper failure mode.

**``pool_pre_ping`` and ``pool_recycle`` are not optional here.** Together they
are what make scale-to-zero survivable: ``pre_ping`` discards a dead socket
before a request uses it, and ``recycle`` retires connections older than the
suspend window (300 s) instead of trusting them.

**Sessions are short-lived.** A session is opened inside the persistence task and
closed in a ``finally``, so an exception cannot leak a connection — which on a
small pool would eventually starve the service without any obvious error.

Why ``expire_on_commit=False``: after a commit the ORM would otherwise mark
attribute access as requiring a fresh round-trip. The persistence code reads
``prediction_id`` afterwards for logging, and that must not trigger a query
against a database that may already be gone.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from churn.config import DatabaseSettings, get_settings
from churn.logging_utils import get_logger

logger = get_logger(__name__)

#: Process-wide engine. ``None`` until first use so importing this module has no
#: side effects (important for tests and for Alembic).
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def build_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create a new engine from settings (does **not** open a connection)."""
    resolved = settings or get_settings().database
    engine = create_engine(resolved.url, **resolved.engine_kwargs())
    logger.info(
        "database engine created",
        extra={
            "url": resolved.redacted_url,
            "pool_size": None if resolved.is_sqlite else resolved.pool_size,
        },
    )
    return engine


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope that always closes.

    Usage::

        with session_scope() as session:
            session.add(record)

    Commits on clean exit, rolls back on exception, and **closes in every case**.
    The close is what protects the small pool from leaks.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(*, timeout_seconds: int | None = None) -> tuple[bool, str | None]:
    """Run ``SELECT 1`` to test reachability.

    Args:
        timeout_seconds: Statement timeout. Ignored for SQLite, which has no
            server-side ``statement_timeout``.

    Returns:
        ``(True, None)`` when reachable, ``(False, error_class)`` otherwise. The
        error is returned as a **class name only** — never a message, which on
        psycopg2 can embed host and user details, and never the DSN.
    """
    settings = get_settings().database
    timeout = timeout_seconds if timeout_seconds is not None else settings.readiness_timeout_seconds

    try:
        with get_engine().connect() as connection:
            if not settings.is_sqlite:
                # Server-side guard so a hung probe cannot pin a connection.
                connection.execute(text("SET statement_timeout = :ms"), {"ms": timeout * 1000})
            connection.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return False, type(exc).__name__


def dispose_engine() -> None:
    """Close pooled connections (called at application shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        logger.info("database engine disposed")
    _engine = None
    _session_factory = None


def reset_for_tests(engine: Engine | None = None) -> None:
    """Point the module at a specific engine (test helper)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = None


def engine_info() -> dict[str, Any]:
    """Non-sensitive description of the configured database."""
    settings = get_settings().database
    return {
        "dialect": "sqlite" if settings.is_sqlite else "postgresql",
        "url": settings.redacted_url,
        "pool_size": None if settings.is_sqlite else settings.pool_size,
        "max_overflow": None if settings.is_sqlite else settings.max_overflow,
        "pool_pre_ping": settings.pool_pre_ping,
        "pool_recycle_seconds": None if settings.is_sqlite else settings.pool_recycle,
        "connect_timeout_seconds": None if settings.is_sqlite else settings.connect_timeout,
    }
