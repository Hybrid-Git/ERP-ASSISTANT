from dotenv import load_dotenv
from pydantic import Field, ConfigDict, SecretStr
import os  
import yaml  
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
load_dotenv(override=True)

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "chapter1-erp-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # LLM Models
    llm_model: str = Field(...)
    trans_llm_model: str | None = None
    summary_llm_model: str = Field(...)
    emb_model: str = Field(...)

    # Endpoints
    llm_base_url: str = Field(...)
    trans_base_url: str | None = None
    summary_base_url: str = Field(...)
    emb_base_url: str = Field(...)

    # API Keys
    llm_model_api_key: SecretStr = Field(...)
    trans_model_api_key: SecretStr | None = None
    summary_model_api_key: SecretStr = Field(...)
    emb_model_api_key: SecretStr = Field(...)

    # Chapter1 ERP API
    chp1_api_base_url: str = Field(...)
    chp1_api_token: SecretStr = Field(...)
    chp1_api_timeout: int = 30
    company_id: int = Field(...)

    # App Settings
    output_format: str = "text"
    backend_url: str = "http://127.0.0.1:8000"
    summary_limit: int = 3

    # Security / CORS
    app_api_key: str = ""
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000"


settings = Settings()
# Compatibility exports — so existing code does not break
LLM_MODEL = settings.llm_model
TRANS_LLM_MODEL = settings.trans_llm_model or settings.llm_model
SUMMARY_LLM_MODEL = settings.summary_llm_model
EMB_MODEL = settings.emb_model

LLM_BASE_URL = settings.llm_base_url
TRANS_BASE_URL = settings.trans_base_url or settings.llm_base_url
SUMMARY_BASE_URL = settings.summary_base_url
EMB_BASE_URL = settings.emb_base_url

LLM_MODEL_API_KEY = settings.llm_model_api_key.get_secret_value()
TRANS_MODEL_API_KEY = (
    settings.trans_model_api_key.get_secret_value()
    if settings.trans_model_api_key
    else settings.llm_model_api_key.get_secret_value()
)
SUMMARY_MODEL_API_KEY = settings.summary_model_api_key.get_secret_value()
EMB_MODEL_API_KEY = settings.emb_model_api_key.get_secret_value()

CHP1_API_BASE_URL = settings.chp1_api_base_url
CHP1_API_TOKEN = settings.chp1_api_token.get_secret_value()
CHP1_API_TIMEOUT = settings.chp1_api_timeout
COMPANY_ID = settings.company_id

OUTPUT_FORMAT = settings.output_format
BACKEND_URL = settings.backend_url
SUMMARY_LIMIT = settings.summary_limit

APP_API_KEY = settings.app_api_key
CORS_ORIGINS = settings.cors_origins
embedding_model = OpenAIEmbeddings(
    model=os.getenv("EMB_MODEL"),
    base_url= os.getenv("EMB_BASE_URL"),
    api_key= os.getenv("EMB_MODEL_API_KEY"),
    timeout=180,
    check_embedding_ctx_length=False,
)

normalizer_llm = ChatOpenAI(
    model=os.getenv("SUMMARY_LLM_MODEL"),
    base_url= os.getenv("SUMMARY_BASE_URL"),
    api_key= os.getenv("SUMMARY_MODEL_API_KEY"),
    temperature=0.0,
    max_tokens=256,
    timeout=60,
    # response_format={"type": "json_object"},
    extra_body={
        "keep_alive": "5m",
        "reasoning_effort": None
    },
)
llm = ChatOpenAI(
    model= os.getenv("LLM_MODEL") ,
    base_url= os.getenv("LLM_BASE_URL"),
    api_key= os.getenv("LLM_MODEL_API_KEY"),
    temperature=0.0,
    max_tokens=4096,
    timeout=120,
    disable_streaming="tool_calling",

    extra_body={
        "keep_alive": "5m",
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
summary_llm = ChatOpenAI(
    model= os.getenv("SUMMARY_LLM_MODEL") ,
    base_url= os.getenv("SUMMARY_BASE_URL"),
    api_key= os.getenv("SUMMARY_MODEL_API_KEY"),
    temperature=0.7,
    max_tokens=4096,
    timeout=120,

    extra_body={
        "keep_alive": "5m",
        "reasoning_effort": None
    },
)


# ── Pipeline config (YAML — edit config.yaml, not Python) ──
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def _load_pipeline_config():
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: could not load {_CONFIG_PATH}: {e}")
        return {}


PIPELINE_CONFIG = _load_pipeline_config()


def get_cfg(*keys, default=None):
    """Safely traverse PIPELINE_CONFIG with dotted keys."""
    val = PIPELINE_CONFIG
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default
