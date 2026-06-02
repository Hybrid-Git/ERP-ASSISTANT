from dotenv import load_dotenv
load_dotenv()

import os
import yaml
from langchain_ollama import OllamaEmbeddings, ChatOllama

embedding_model = OllamaEmbeddings(model="bge-m3")

normalizer_llm = ChatOllama(
    model="qwen3:latest",
    temperature=0.0,
    keep_alive="30m",
    num_ctx=1024,
    num_predict=512,
    reasoning=False,
)
llm = ChatOllama(
    model="qwen3:latest",
    temperature=0.0,
    keep_alive="30m",
    num_ctx=4096,
    num_predict=2048,
    reasoning=False,
)

print("LLM and embedding model initialised!")

# ── API config (env vars) ──
CHP1_API_BASE_URL = os.getenv("CHP1_API_BASE_URL", "https://dev.chapter1.finance/aiAnalytics/")
CHP1_API_TOKEN = os.getenv("CHP1_API_TOKEN", "")
CHP1_API_TIMEOUT = int(os.getenv("CHP1_API_TIMEOUT", "10"))
COMPANY_ID = int(os.getenv("COMPANY_ID", "355"))

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
