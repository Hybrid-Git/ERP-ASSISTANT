import copy
from collections import OrderedDict
import asyncio

import os

COMPANY_ID = os.getenv("COMPANY_ID")


# ==============================
# FINAL RESPONSE CACHE
# ==============================
FINAL_RESPONSE_CACHE = OrderedDict()
FINAL_RESPONSE_CACHE_MAXSIZE = 500
FINAL_RESPONSE_CACHE_TTL_SECONDS = 300
CACHE_LOCK = asyncio.Lock()


def normalize_query_for_cache(query: str) -> str:
    return f"{COMPANY_ID}::{' '.join((query or '').lower().strip().split())}"


def should_cache_final_response(result: dict) -> bool:
    if not isinstance(result, dict):
        return False

    response = result.get("response")
    if not isinstance(response, dict):
        return False

    success = response.get("success")
    status = response.get("status")
    tools_used = response.get("tools_used", [])

    if not tools_used:
        return False

    return success is True and status == "success"


async def get_cached_final_response(query: str):
    async with CACHE_LOCK:
        key = normalize_query_for_cache(query)
        cached = FINAL_RESPONSE_CACHE.get(key)

        if not cached:
            print(f"[FINAL CACHE MISS] {key}")
            return None

        age = time.monotonic() - cached.get("cached_at", 0)
        if age > FINAL_RESPONSE_CACHE_TTL_SECONDS:
            print(f"[FINAL CACHE EXPIRED] {key}")
            FINAL_RESPONSE_CACHE.pop(key, None)
            return None

        result = cached.get("result")
        if not isinstance(result, dict) or "response" not in result:
            print(f"[FINAL CACHE INVALID] {key}")
            FINAL_RESPONSE_CACHE.pop(key, None)
            return None

        print(f"[FINAL CACHE HIT] {key}")

        # Use deepcopy instead of JSON serialization because LangChain objects may not serialize cleanly.
        result = copy.deepcopy(result)
        result["timings"] = [{"node": "final_response_cache", "duration_sec": 0.001}]
        result["total_time_sec"] = 0.001

        return result


async def set_cached_final_response(query: str, result: dict):
    if not should_cache_final_response(result):
        return

    key = normalize_query_for_cache(query)

    # Cache only API output payload, never session-specific LangChain messages.
    cacheable_result = {
        "response": result.get("response"),
        "timings": result.get("timings", []),
        "total_time_sec": result.get("total_time_sec", 0.0),
    }

    async with CACHE_LOCK:
        FINAL_RESPONSE_CACHE[key] = {
            "cached_at": time.monotonic(),
            "result": copy.deepcopy(cacheable_result),
        }
        while len(FINAL_RESPONSE_CACHE) > FINAL_RESPONSE_CACHE_MAXSIZE:
            FINAL_RESPONSE_CACHE.popitem(last=False)
    print(f"[FINAL CACHE SET] {key}")