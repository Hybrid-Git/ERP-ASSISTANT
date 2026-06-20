from fastapi.middleware.cors import CORSMiddleware
from app.core.config import CORS_ORIGINS, llm
import time
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import  SystemMessage, HumanMessage
from app.core.settings_validator import validate_settings
from app.core.logging_config import setup_logging
from fastapi.responses import JSONResponse
from app.services.erp_client import erp_client
from app.utils.response_utils import make_error_response
from app.core.exceptions import ERPAssistantError
from app.api.routes.session_routes import router as session_router
from app.api.routes.chat_routes import router as chat_router    


logger = setup_logging()
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Validating configuration")
    validate_settings()
    logger.info("Configuration validation completed")
    logger.info("FastAPI started. Warming up worker LLM...")
    start = time.perf_counter()
    try:
        await llm.ainvoke([SystemMessage(content="Return only: OK\n/no_think"),
                           HumanMessage(content="ping")])
        elapsed_time = time.perf_counter() - start
        logger.info(
                    "Worker LLM warmup completed",
                    extra={"duration_sec": round(elapsed_time, 3)}
                    )
    except Exception:
        logger.exception("LLM warmup failed; will load on first query")
    yield
    await erp_client.close()
    logger.info("ERP client connection pool closed")
def get_cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in CORS_ORIGINS.split(",")
        if origin.strip()
    ]

app = FastAPI(
    title="CHAPTER-1-ASSIST",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
app.include_router(session_router)
app.include_router(chat_router)
#Exception handlers
@app.exception_handler(ERPAssistantError)
async def erp_error_handler(request, exc: ERPAssistantError):
    logger.error(
        "ERP assistant error",
        extra={"error_code": exc.error_code},
    )

    return JSONResponse(
        status_code=400,
        content=make_error_response(
            status=exc.error_code,
            summary=exc.user_message,
            errors=[exc.user_message],
        ),
    )
@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.exception("Unhandled server error")

    return JSONResponse(
        status_code=500,
        content=make_error_response(
            status="internal_error",
            summary="An unexpected error occurred. Please try again later.",
            errors=["internal_error"],
        ),
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "chapter1-erp-assistant",
    }
@app.get("/ready")
async def readiness_check():
    try:
        await llm.ainvoke([
            HumanMessage(content="ping")
        ])

        return {
            "status": "ready",
            "service": "chapter1-erp-assistant",
        }

    except Exception:
        logger.exception("Readiness check failed")

        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                # "error": str(e),
            },
        )

# existing routes below
@app.get("/")
async def root():
    return {"message": "ERP Assistant API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)