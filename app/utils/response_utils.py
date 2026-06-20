from typing import Any


def make_success_response(
    query: str,
    data: dict[str, Any] | None = None,
    summary: str = "",
    tools_used: list[str] | None = None,
    timings: list[dict[str, Any]] | None = None,
    total_time_sec: float | None = None,
) -> dict[str, Any]:
    response = {
        "success": True,
        "status": "success",
        "query": query,
        "tools_used": tools_used or [],
        "data": data or {},
        "summary": summary,
        "errors": [],
    }

    if timings is not None:
        response["timings"] = timings

    if total_time_sec is not None:
        response["total_time_sec"] = total_time_sec

    return response


def make_error_response(
    query: str = "",
    status: str = "internal_error",
    summary: str = "An unexpected error occurred.",
    errors: list[str] | None = None,
    tools_used: list[str] | None = None,
    data: dict[str, Any] | None = None,
    timings: list[dict[str, Any]] | None = None,
    total_time_sec: float | None = None,
) -> dict[str, Any]:
    response = {
        "success": False,
        "status": status,
        "query": query,
        "tools_used": tools_used or [],
        "data": data or {},
        "summary": summary,
        "errors": errors or [summary],
    }

    if timings is not None:
        response["timings"] = timings

    if total_time_sec is not None:
        response["total_time_sec"] = total_time_sec

    return response