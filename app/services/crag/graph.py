from typing import Any


_graph_app: Any | None = None


def build_graph_app():
    """CRAG graph placeholder for future implementation."""
    raise NotImplementedError("CRAG graph is not implemented yet.")


def get_graph_app():
    """Return the CRAG graph singleton when implemented."""
    global _graph_app
    if _graph_app is None:
        raise NotImplementedError("CRAG graph is not implemented yet.")
    return _graph_app


def warmup_graph_dependencies() -> None:
    """Warm-up placeholder for future CRAG dependencies."""
    return None
