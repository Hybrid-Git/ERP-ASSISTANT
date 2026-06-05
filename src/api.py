from langsmith import traceable
import requests
from typing import Any, Optional
import time
from src.config import CHP1_API_BASE_URL, CHP1_API_TIMEOUT,CHP1_API_TOKEN


def build_url(endpoint: str) -> str:
    base_url = CHP1_API_BASE_URL.rstrip("/")
    endpoint = endpoint.strip("/")
    return f"{base_url}/{endpoint}"

def parse_response(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {
            "raw_text": response.text[:1000]
        }

    if not response.ok:
        return {
            "success": False,
            "status_code": response.status_code,
            "data": [],
            "count": 0,
            "error": payload,
            "raw_response": payload,
        }

    if isinstance(payload, dict) and payload.get("st") is False:
        return {
            "success": False,
            "status_code": response.status_code,
            "data": [],
            "count": 0,
            "error": payload.get("msg", "API returned st=false"),
            "raw_response": payload,
        }

    data = payload.get("data", payload) if isinstance(payload, dict) else payload

    if data is None:
        data = []

    return {
        "success": True,
        "status_code": response.status_code,
        "data": data,
        "count": len(data) if isinstance(data, list) else None,
        "error": None,
        "raw_response": payload,
    }
@traceable(name="chapter1_api_post",run_type="tool")
def api_post(endpoint: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = build_url(endpoint)
    final_body = body or {}

    try:
        print(f"[API POST] URL-----------: {url}")
        print(f"[API POST] BODY-----------: {final_body}")

        request_start = time.perf_counter()

        response = requests.post(
                                url,    
                                json=final_body,
                                headers={"Authorization": f"{CHP1_API_TOKEN}"},
                                timeout=CHP1_API_TIMEOUT,
                            )

        request_duration = time.perf_counter() - request_start

        print(f"[API POST] REQUEST TIME-----: {request_duration:.3f}s")

        print(f"[API POST] STATUS----------: {response.status_code}")
        print(f"[API POST] RESPONSE------------: {response.text[:500]}")

        result = parse_response(response)
        # result["endpoint"] = endpoint.strip("/")
        # result["url"] = url
        # result["body"] = final_body

        return result

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": None,
            "endpoint": endpoint.strip("/"),
            "url": url,
            "body": final_body,
            "data": [],
            "count": 0,
            "error": "API request timed out",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": None,
            "endpoint": endpoint.strip("/"),
            "url": url,
            "body": final_body,
            "data": [],
            "count": 0,
            "error": "Could not connect to Chapter1 API",
        }

    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "endpoint": endpoint.strip("/"),
            "url": url,
            "body": final_body,
            "data": [],
            "count": 0,
            "error": str(e),
        }