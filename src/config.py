from dotenv import load_dotenv
load_dotenv(override=True)

import os  # noqa: E402
import yaml  # noqa: E402
# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

embedding_model = OpenAIEmbeddings(
    model=os.getenv("EMB_MODEL"),
    base_url= os.getenv("EMB_BASE_URL"),
    api_key= os.getenv("EMB_MODEL_API_KEY"),
    timeout=180,
    check_embedding_ctx_length=False,
)

normalizer_llm = ChatOpenAI(
    model=os.getenv("TRANS_LLM_MODEL"),
    base_url= os.getenv("TRANS_BASE_URL"),
    api_key= os.getenv("TRANS_MODEL_API_KEY"),
    temperature=0.0,
    max_tokens=256,
    timeout=60,

    extra_body={
        "keep_alive": "5m",
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
    },
)
summary_llm = ChatOpenAI(
    model= os.getenv("SUMMARY_LLM_MODEL") ,
    base_url= os.getenv("SUMMARY_BASE_URL"),
    api_key= os.getenv("SUMMARY_MODEL_API_KEY"),
    temperature=0.9,
    max_tokens=4096,
    timeout=120,

    extra_body={
        "keep_alive": "5m",
    },
)

print(f"LLM model loaded: {llm.model}")
print(f"Embedding model loaded: {embedding_model.model}")
print("LLM and embedding model initialised!")

# ── API config (env vars) ──
CHP1_API_BASE_URL = os.getenv("CHP1_API_BASE_URL", "")
CHP1_API_TOKEN = os.getenv("CHP1_API_TOKEN", "")
CHP1_API_TIMEOUT = int(os.getenv("CHP1_API_TIMEOUT", ""))
COMPANY_ID = int(os.getenv("COMPANY_ID", ""))

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
