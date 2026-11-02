"""Append-only audit trail.

In-process and bounded by design: this is the minimum needed to answer "who
queried what" during an incident. In production, ship these events to your log
pipeline or a durable store — the ring buffer only survives as long as the
process does.
"""

import time
from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, List

_MAX_EVENTS = 500
_events: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_lock = Lock()


def record(action: str, principal: str, **fields: Any) -> None:
    with _lock:
        _events.append(
            {
                "ts": time.time(),
                "action": action,
                "principal": principal,
                **fields,
            }
        )


def recent(limit: int = 100) -> List[Dict[str, Any]]:
    with _lock:
        return list(_events)[-limit:][::-1]
