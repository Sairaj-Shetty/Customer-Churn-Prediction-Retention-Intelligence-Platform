"""Tests for structured logging and PII redaction.

These tests exercise the filter and formatter directly rather than relying on
``capsys`` (which intercepts at the file-descriptor/stdout level and is
unreliable once other code has reconfigured the root logger). Testing the
components in isolation keeps the assertions about *behaviour* — "no PII in the
output" — instead of about test-harness plumbing.
"""

from __future__ import annotations

import json
import logging

import pytest

from churn.logging_utils import (
    JsonFormatter,
    PIIRedactionFilter,
    configure_logging,
    get_logger,
)


def make_record(msg: str, level: int = logging.INFO, **extra: object) -> logging.LogRecord:
    """Build a LogRecord as ``logging.Logger`` would, including ``extra``."""
    record = logging.LogRecord(
        name="test", level=level, pathname="", lineno=1, msg=msg, args=None, exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def render(record: logging.LogRecord) -> str:
    """Pass a record through redaction then JSON formatting."""
    PIIRedactionFilter().filter(record)
    return JsonFormatter().format(record)


# --------------------------------------------------------------------------- #
# JSON structure
# --------------------------------------------------------------------------- #
def test_json_output_is_parseable_with_expected_fields():
    payload = json.loads(render(make_record("hello structured world", run_id="abc123")))
    assert payload["message"] == "hello structured world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["timestamp"]
    assert payload["run_id"] == "abc123", "extra fields must be promoted to top level"


def test_reserved_attributes_are_not_duplicated_in_payload():
    payload = json.loads(render(make_record("x")))
    # Standard LogRecord internals must not leak into the JSON body.
    for leaked in ("msg", "args", "lineno", "levelno", "pathname", "exc_info"):
        assert leaked not in payload


def test_exception_is_serialised():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record("failed", level=logging.ERROR)
        record.exc_info = sys.exc_info()
    payload = json.loads(render(record))
    assert "boom" in payload["exception"]


# --------------------------------------------------------------------------- #
# PII redaction
# --------------------------------------------------------------------------- #
def test_extra_customer_id_is_redacted():
    out = render(make_record("predicting", customerID="7590-VHVEG"))
    assert "7590-VHVEG" not in out
    assert "REDACTED" in out


def test_snake_case_customer_id_is_redacted():
    out = render(make_record("predicting", customer_id="3186-AJIEK"))
    assert "3186-AJIEK" not in out
    assert "REDACTED" in out


def test_inline_customer_id_in_message_is_redacted():
    out = render(make_record('stored record for customerID: "3186-AJIEK"'))
    assert "3186-AJIEK" not in out
    assert "REDACTED" in out


def test_inline_redaction_survives_repeated_get_message():
    """The redacted message must not be re-rendered back to the raw value."""
    record = make_record('customerID="7590-VHVEG"')
    PIIRedactionFilter().filter(record)
    assert "7590-VHVEG" not in record.getMessage()
    assert "7590-VHVEG" not in record.getMessage()


@pytest.mark.parametrize("key", ["password", "secret", "token", "api_key", "authorization"])
def test_sensitive_keys_are_redacted(key):
    out = render(make_record("auth", **{key: "super-secret-value"}))
    assert "super-secret-value" not in out
    assert "REDACTED" in out


def test_benign_fields_are_not_redacted():
    """Redaction must not destroy useful observability data."""
    payload = json.loads(render(make_record("pred", churn_probability=0.83, model_version="3")))
    assert payload["churn_probability"] == 0.83
    assert payload["model_version"] == "3"


def test_filter_returns_true_so_records_are_not_dropped():
    assert PIIRedactionFilter().filter(make_record("ok")) is True


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_attaches_pii_filter():
    configure_logging("INFO")
    handler = logging.getLogger().handlers[0]
    assert any(isinstance(f, PIIRedactionFilter) for f in handler.filters)


def test_text_mode_does_not_emit_json():
    configure_logging("INFO", json_output=False)
    assert isinstance(get_logger("x"), logging.Logger)
