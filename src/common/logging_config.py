"""Structured, correlation-aware logging.

Every log line carries the request id from the current context so that a single
HTTP request can be reconstructed from a log aggregator. JSON is the default;
plain text is available for local development (`LOG_JSON=false`).
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Optional

from src.common.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Fields copied from LogRecord.extra into the structured payload. An explicit
# allowlist keeps arbitrary objects out of the JSON line.
_EXTRA_FIELDS = (
    "event",
    "route",
    "method",
    "status_code",
    "duration_ms",
    "principal",
    "client",
    "query_chars",
    "doc_id",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "attempt",
    "provider",
    "model",
)


class JsonFormatter(logging.Formatter):
    """Minimal dependency-free JSON lines formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local runs."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"[{request_id_var.get()}] {record.name}: {record.getMessage()}"
        )
        extras = {
            field: getattr(record, field)
            for field in _EXTRA_FIELDS
            if getattr(record, field, None) is not None
        }
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: Optional[str] = None, json_output: Optional[bool] = None) -> None:
    """Idempotently configure the root logger."""
    resolved_level = (level or settings.LOG_LEVEL or "INFO").upper()
    use_json = settings.LOG_JSON if json_output is None else json_output

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Uvicorn installs its own handlers; route them through ours for consistency.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
