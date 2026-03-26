import time
from contextvars import ContextVar
from typing import Any

_trace_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar("trace_events", default=None)
_trace_started_at_var: ContextVar[float | None] = ContextVar("trace_started_at", default=None)


def add_trace(stage: str, message: str, **details: Any) -> None:
    events = _trace_events_var.get()
    started_at = _trace_started_at_var.get()
    if events is None or started_at is None:
        return

    events.append(
        {
            "stage": stage,
            "message": message,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "details": details,
        }
    )


def begin_trace() -> tuple[list[dict[str, Any]], object, object, float]:
    started_at = time.perf_counter()
    trace_events: list[dict[str, Any]] = []
    trace_token = _trace_events_var.set(trace_events)
    started_token = _trace_started_at_var.set(started_at)
    return trace_events, trace_token, started_token, started_at


def end_trace(trace_token: object, started_token: object) -> None:
    _trace_events_var.reset(trace_token)
    _trace_started_at_var.reset(started_token)
