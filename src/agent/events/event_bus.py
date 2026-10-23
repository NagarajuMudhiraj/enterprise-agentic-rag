import asyncio
from typing import Callable, List, Dict, Any, Optional

class AgentEvent:
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type,
            **self.data
        }

class AgentEventBus:
    """
    Asynchronous event bus for real-time thought streaming, tool execution observation,
    and Server-Sent Events (SSE) telemetry.
    """

    def __init__(self):
        self._subscribers: List[Callable[[AgentEvent], Any]] = []

    def subscribe(self, callback: Callable[[AgentEvent], Any]):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[AgentEvent], Any]):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def emit(self, event_type: str, **kwargs):
        event = AgentEvent(event_type=event_type, data=kwargs)
        for sub in self._subscribers:
            try:
                res = sub(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
