from typing import Any, Optional
from app.services.erp_client import erp_client

async def api_post(endpoint: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return await erp_client.post(endpoint, body)





