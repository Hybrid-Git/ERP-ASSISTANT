def _build_langsmith_config(run_name: str, request_id: str, query: str, session_id: str, tags: list | None = None) -> dict:
    return {
        "run_name": run_name,
        "tags": tags or ["fastapi", "langgraph", "erp-assistant"],
        "metadata": {
            "request_id": request_id,
            "query": query,
            "session_id": session_id,
        },
    }