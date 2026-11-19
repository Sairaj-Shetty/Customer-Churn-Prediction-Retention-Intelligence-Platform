"""Declarative base and shared column types.

Kept separate from the models so Alembic's ``env.py`` can import ``Base.metadata``
without importing the application, the model artifact, or shap.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass
