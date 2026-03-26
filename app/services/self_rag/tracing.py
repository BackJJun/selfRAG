import time
from contextvars import ContextVar
from typing import Any

_trace_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar("trace_events", default=None)
_trace_started_at_var: ContextVar[float | None] = ContextVar("trace_started_at", default=None)


# 현재 수행 중인 프로세스의 특정 단계 정보를 기록한다.
# 'ContextVar'를 통해 호출한 스레드/태스크별로 독립적인 트레이스 정보를 유지하며, 
# 시작 시간 대비 경과 시간(elapsed_ms)과 상세 내용(details)을 함께 저장한다.
def add_trace(stage: str, message: str, **details: Any) -> None:
    """수행 중인 단계의 이벤트 정보를 현재 컨텍스트의 트레이스 리스트에 추가합니다."""
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


# 새로운 트레이싱 세션을 초기화한다.
# 비동기 환경이나 멀티 스레드 환경에서 안전하게 정보를 공유하기 위해 'ContextVar' 토큰을 생성하고, 
# 이벤트 리스트와 시작 시각을 설정하여 반환한다.
def begin_trace() -> tuple[list[dict[str, Any]], object, object, float]:
    """새로운 트레이스 컨텍스트를 생성하고 시작 정보를 반환합니다."""
    started_at = time.perf_counter()
    trace_events: list[dict[str, Any]] = []
    trace_token = _trace_events_var.set(trace_events)
    started_token = _trace_started_at_var.set(started_at)
    return trace_events, trace_token, started_token, started_at


# 활성화된 트레이싱 세션을 정상적으로 종료한다.
# 'begin_trace'에서 반환받은 토큰들을 사용하여 'ContextVar'를 이전 상태(None)로 복구(Reset)한다.
def end_trace(trace_token: object, started_token: object) -> None:
    """트레이스 세션을 종료하고 컨텍스트 변수를 초기 상태로 되돌립니다."""
    _trace_events_var.reset(trace_token)
    _trace_started_at_var.reset(started_token)
