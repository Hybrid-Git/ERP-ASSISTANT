import logging

from app.core.config import settings

logger = logging.getLogger("erp_assistant.config")


def validate_settings():
    if settings.company_id <= 0:
        raise RuntimeError("COMPANY_ID must be greater than 0")

    required_urls = {
        "LLM_BASE_URL": settings.llm_base_url,
        "SUMMARY_BASE_URL": settings.summary_base_url,
        "EMB_BASE_URL": settings.emb_base_url,
        "CHP1_API_BASE_URL": settings.chp1_api_base_url,
    }

    for key, value in required_urls.items():
        if not value.startswith(("http://", "https://")):
            raise RuntimeError(f"{key} must start with http:// or https://")

    logger.info("Environment validation successful")