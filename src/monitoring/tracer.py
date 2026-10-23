import time
from typing import Dict, Any, List

class PipelineTracer:
    """
    Lightweight, structured pipeline tracer compatible with OpenTelemetry spans.
    Records per-stage latency budgets, candidate counts, and verification outcomes.
    """

    def __init__(self, query_id: str):
        self.query_id = query_id
        self.start_time = time.perf_counter()
        self.spans: List[Dict[str, Any]] = []

    def record_span(self, stage_name: str, duration_ms: float, metadata: Dict[str, Any] = None):
        self.spans.append({
            "stage": stage_name,
            "duration_ms": round(duration_ms, 2),
            "timestamp": time.time(),
            "metadata": metadata or {}
        })

    def get_total_latency_ms(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def summary(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "total_latency_ms": self.get_total_latency_ms(),
            "spans_count": len(self.spans),
            "trace_events": self.spans
        }
