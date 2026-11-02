"""In-process metrics registry with Prometheus text exposition.

Deliberately dependency-free: the exposition format is small and stable, and
adding a client library plus a scrape endpoint creates more surface area than it
removes for a service this size. Swap for `prometheus_client` if you need
native histograms or multiprocess aggregation.

Lives in `common` rather than `api` because the LLM provider and the agent core
record metrics too; keeping it here avoids an api -> agent -> api import cycle.
"""

import threading
import time
from typing import Dict, Iterable, Optional, Tuple

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

Labels = Optional[Dict[str, str]]


def _key(name: str, labels: Labels) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    items = tuple(sorted((labels or {}).items()))
    return name, items


def _format_labels(items: Iterable[Tuple[str, str]]) -> str:
    pairs = [f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
             for k, v in items]
    return "{" + ",".join(pairs) + "}" if pairs else ""


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, tuple], float] = {}
        self._histograms: Dict[Tuple[str, tuple], Dict[str, float]] = {}
        self._help: Dict[str, str] = {}

    # ------------------------------------------------------------- recording
    def describe(self, name: str, help_text: str) -> None:
        self._help.setdefault(name, help_text)

    def incr(self, name: str, labels: Labels = None, value: float = 1.0) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, labels: Labels = None) -> None:
        key = _key(name, labels)
        with self._lock:
            bucket = self._histograms.get(key)
            if bucket is None:
                bucket = {"count": 0.0, "sum": 0.0}
                for le in _LATENCY_BUCKETS:
                    bucket[f"le_{le}"] = 0.0
                self._histograms[key] = bucket
            bucket["count"] += 1
            bucket["sum"] += value
            for le in _LATENCY_BUCKETS:
                if value <= le:
                    bucket[f"le_{le}"] += 1

    def gauge(self, name: str, value: float, labels: Labels = None) -> None:
        """A gauge reuses the counter store but is rendered without the _total suffix."""
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = value

    # ---------------------------------------------------------------- export
    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {name: value for (name, _), value in self._counters.items()}

    def render(self) -> str:
        lines = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: dict(v) for k, v in self._histograms.items()}

        for (name, items), value in sorted(counters.items()):
            lines.append(f"# HELP {name} {self._help.get(name, name)}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{_format_labels(items)} {value}")
            lines.append("")

        for (name, items), bucket in sorted(histograms.items()):
            lines.append(f"# HELP {name} {self._help.get(name, name)}")
            lines.append(f"# TYPE {name} histogram")
            for le in _LATENCY_BUCKETS:
                labels = dict(items)
                labels["le"] = le
                lines.append(
                    f"{name}_bucket{_format_labels(tuple(sorted(labels.items())))} "
                    f"{bucket[f'le_{le}']}"
                )
            labels = dict(items)
            labels["le"] = "+Inf"
            lines.append(
                f"{name}_bucket{_format_labels(tuple(sorted(labels.items())))} {bucket['count']}"
            )
            lines.append(f"{name}_sum{_format_labels(items)} {bucket['sum']}")
            lines.append(f"{name}_count{_format_labels(items)} {bucket['count']}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"


metrics = MetricsRegistry()

metrics.describe("http_requests_total", "Total HTTP requests processed.")
metrics.describe("http_request_duration_seconds", "HTTP request latency in seconds.")
metrics.describe("http_requests_rejected_total", "Requests rejected before reaching a handler.")
metrics.describe("llm_requests_total", "Outbound LLM API calls by outcome.")
metrics.describe("llm_request_duration_seconds", "LLM API call latency in seconds.")
metrics.describe("llm_tokens_total", "LLM tokens consumed (direction=input|output).")
metrics.describe("llm_cost_usd_total", "Estimated LLM spend in USD.")
metrics.describe("rag_queries_total", "RAG queries by verification outcome.")
metrics.describe("rag_query_duration_seconds", "End-to-end RAG query latency in seconds.")
metrics.describe("rag_documents_total", "Documents ingested by result.")
metrics.describe("rag_chunks_indexed", "Child chunks currently indexed.")


class Timer:
    """Context manager that records a histogram observation in seconds."""

    def __init__(self, name: str, labels: Labels = None) -> None:
        self.name = name
        self.labels = labels
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        metrics.observe(self.name, time.perf_counter() - self._start, self.labels)
