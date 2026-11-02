"""Cross-cutting HTTP middleware.

Two concerns, kept in one place so ordering is explicit in server.py:
  * RequestContextMiddleware — correlation id, access log, metrics, security headers
  * RateLimitMiddleware      — fixed-window limiter per credential or client IP
"""

import hashlib
import time
import uuid
from collections import defaultdict, deque
from typing import DefaultDict, Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.common.metrics import metrics
from src.common.config import settings
from src.common.logging_config import get_logger, request_id_var

logger = get_logger("api.access")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}

# Endpoints excluded from throttling: probe and scrape traffic must never be limited.
_RATE_LIMIT_EXEMPT_PREFIXES = ("/health", "/ready", "/metrics", "/static", "/favicon.ico")


def _caller_identity(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return "key_" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    host = request.client.host if request.client else "unknown"
    return f"ip_{host}"


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, emit one structured access log, record metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            metrics.incr(
                "http_requests_total",
                {"method": request.method, "route": _route_label(request), "status": "500"},
            )
            logger.exception(
                "Unhandled request error",
                extra={
                    "method": request.method,
                    "route": _route_label(request),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        finally:
            duration = time.perf_counter() - started
            metrics.incr(
                "http_requests_total",
                {"method": request.method, "route": _route_label(request), "status": str(status_code)},
            )
            metrics.observe(
                "http_request_duration_seconds",
                duration,
                {"route": _route_label(request)},
            )
            if not request.url.path.startswith(_RATE_LIMIT_EXEMPT_PREFIXES):
                logger.info(
                    "request",
                    extra={
                        "method": request.method,
                        "route": _route_label(request),
                        "status_code": status_code,
                        "duration_ms": round(duration * 1000, 2),
                        "client": _caller_identity(request),
                    },
                )
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window limiter: RATE_LIMIT_PER_MINUTE requests per caller per minute."""

    def __init__(self, app, limit_per_minute: int = None) -> None:
        super().__init__(app)
        self.limit = limit_per_minute if limit_per_minute is not None else settings.RATE_LIMIT_PER_MINUTE
        self.window_seconds = 60.0
        self._hits: DefaultDict[str, Deque[float]] = defaultdict(deque)

    def _allow(self, identity: str, now: float) -> bool:
        if self.limit <= 0:
            return True
        hits = self._hits[identity]
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        if not hits:
            self._hits.pop(identity, None)
        return True

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_RATE_LIMIT_EXEMPT_PREFIXES) or self.limit <= 0:
            return await call_next(request)

        identity = _caller_identity(request)
        if not self._allow(identity, time.monotonic()):
            metrics.incr("http_requests_rejected_total", {"reason": "rate_limited"})
            logger.warning("Rate limit exceeded", extra={"client": identity})
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": (
                            f"Rate limit of {self.limit} requests per minute exceeded. "
                            "Retry after the window resets."
                        ),
                    },
                    "request_id": request_id_var.get(),
                },
                headers={"Retry-After": str(int(self.window_seconds))},
            )
        return await call_next(request)
