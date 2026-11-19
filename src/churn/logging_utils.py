"""Structured JSON logging with PII redaction.

Two things this module gets right that ad-hoc ``print()``/``logging.info``
setups usually get wrong:

1. **Machine-readable output.** Every record is one JSON object, so logs can be
   shipped to CloudWatch and queried by field.
2. **PII never leaves the process.** ``customerID`` (and anything that looks
   like a sensitive key) is masked by a filter attached to the *handler*, so it
   applies to every logger without each developer remembering to redact — even
   third-party libraries that log a raw record get scrubbed.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

#: Keys (lower-cased) whose values are always masked when present on a record.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "customerid",
        "customer_id",
        "password",
        "secret",
        "token",
        "api_key",
        "authorization",
    }
)

#: Placeholder substituted for redacted values.
REDACTED = "***REDACTED***"

#: Matches ``customerID: "7590-VHVEG"`` style values inside free-form text.
_INLINE_ID_PATTERN = re.compile(
    r'((?:customer[_-]?id)["\']?\s*[:=]\s*)["\']?[\w-]{4,}["\']?', re.IGNORECASE
)

#: Attributes that exist on every ``LogRecord``. Computed once from a single
#: real record at import time; ``LogRecord.__dict__`` is the authoritative
#: source, so this cannot drift from the standard library.
_RESERVED_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=None, exc_info=None
    ).__dict__
) | {"message", "asctime", "taskName"}


class PIIRedactionFilter(logging.Filter):
    """Mask sensitive values on a log record before it is formatted.

    Redaction happens *before* formatting and rewrites ``record.msg`` plus
    ``record.extra`` attributes, so the formatter (or any other handler) only
    ever sees safe values.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 1. Scrub sensitive keys supplied via ``extra=``.
        #    ``vars(record)`` covers both extra attributes and standard ones;
        #    no standard attribute name collides with SENSITIVE_KEYS.
        for key in list(vars(record)):
            if key.lower() in SENSITIVE_KEYS:
                setattr(record, key, REDACTED)

        # 2. Scrub inline occurrences in the resolved message. The message is
        #    rendered once here and stored back on the record, so later calls to
        #    getMessage() return the already-redacted text (important: without
        #    clearing ``args`` a second render would re-insert the raw value).
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - logging must never break a request
            return True

        if _INLINE_ID_PATTERN.search(message):
            record.msg = _INLINE_ID_PATTERN.sub(rf"\1{REDACTED}", message)
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote any non-standard attributes (i.e. ``extra=`` fields) to
        # top-level JSON keys so they are queryable.
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Root log level name, e.g. ``"INFO"``.
        json_output: Emit JSON lines (default, production) or human-readable
            text (local debugging). The PII filter is attached in both modes.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(PIIRedactionFilter())

    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    # Replace handlers so repeated calls (tests, reloads) don't duplicate lines.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn installs its own handlers; route them through ours instead.
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(noisy)
        logger.handlers.clear()
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
