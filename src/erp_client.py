import logging
import time
from typing import Any, Optional

import httpx
from langsmith import traceable

from src.config import CHP1_API_BASE_URL, CHP1_API_TIMEOUT, CHP1_API_TOKEN
from src.exceptions import ERPAPIError

logger = logging.getLogger("erp_assistant.erp_client")


class ERPClient:
    def __init__(self):
        self.base_url = CHP1_API_BASE_URL.rstrip("/")
        self.timeout = CHP1_API_TIMEOUT
        self.token = CHP1_API_TOKEN

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self.token,
            },
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
            ),
        )

    def build_url(self, endpoint: str) -> str:
        endpoint = endpoint.strip("/")
        return f"{self.base_url}/{endpoint}"

    def parse_response(self, response: httpx.Response, endpoint: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            logger.error(
                "ERP API returned non-JSON response",
                extra={
                    "status_code": response.status_code,
                    "tool_endpoint": endpoint,
                },
            )
            raise ERPAPIError(
                status_code=response.status_code,
                message=response.text[:1000],
                user_message="ERP API returned an unexpected response format.",
            )

        if not response.is_success:
            logger.error(
                "ERP API returned HTTP error",
                extra={
                    "status_code": response.status_code,
                    "tool_endpoint": endpoint,
                },
            )
            raise ERPAPIError(
                status_code=response.status_code,
                message=str(payload),
                user_message="The ERP system returned an error. Please try again later.",
            )

        if isinstance(payload, dict) and payload.get("st") is False:
            logger.error(
                "ERP API returned st=false",
                extra={
                    "status_code": response.status_code,
                    "tool_endpoint": endpoint,
                },
            )
            raise ERPAPIError(
                status_code=response.status_code,
                message=str(payload.get("msg", "API returned st=false")),
                user_message="The ERP system returned an error. Please try again later.",
            )

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

    @traceable(name="chapter1_api_post", run_type="tool")
    async def post(self, endpoint: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = self.build_url(endpoint)
        final_body = body or {}
        start = time.perf_counter()

        try:
            logger.info(
                "ERP API request started",
                extra={"tool_endpoint": endpoint},
            )

            response = await self.client.post(url, json=final_body)

            duration = time.perf_counter() - start

            logger.info(
                "ERP API response received",
                extra={
                    "tool_endpoint": endpoint,
                    "duration_sec": round(duration, 3),
                    "status_code": response.status_code,
                },
            )

            return self.parse_response(response, endpoint)

        except httpx.TimeoutException:
            duration = time.perf_counter() - start
            logger.exception(
                "ERP API request timed out",
                extra={
                    "tool_endpoint": endpoint,
                    "duration_sec": round(duration, 3),
                },
            )
            raise ERPAPIError(
                message="API request timed out",
                user_message="The ERP system took too long to respond. Please try again.",
            )

        except httpx.ConnectError:
            duration = time.perf_counter() - start
            logger.exception(
                "Could not connect to ERP API",
                extra={
                    "tool_endpoint": endpoint,
                    "duration_sec": round(duration, 3),
                },
            )
            raise ERPAPIError(
                message="Could not connect to Chapter1 API",
                user_message="Could not reach the ERP system. Please try again later.",
            )

        except httpx.RequestError as e:
            duration = time.perf_counter() - start
            logger.exception(
                "ERP API request failed",
                extra={
                    "tool_endpoint": endpoint,
                    "duration_sec": round(duration, 3),
                },
            )
            raise ERPAPIError(
                message=str(e),
                user_message="Could not reach the ERP system. Please try again later.",
            )

        except ERPAPIError:
            raise

        except Exception as e:
            duration = time.perf_counter() - start
            logger.exception(
                "Unexpected ERP API error",
                extra={
                    "tool_endpoint": endpoint,
                    "duration_sec": round(duration, 3),
                },
            )
            raise ERPAPIError(
                message=str(e),
                user_message="Unexpected ERP API error. Please try again later.",
            )

    async def close(self):
        await self.client.aclose()


erp_client = ERPClient()