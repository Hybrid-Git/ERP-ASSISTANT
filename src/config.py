from dotenv import load_dotenv
load_dotenv()

import os
import yaml
# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="bge-m3:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=180,
    check_embedding_ctx_length=False,
)

normalizer_llm = ChatOpenAI(
    model="qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.0,
    max_tokens=1024,   # OpenAI-style replacement for num_predict
    timeout=180,

    # Ollama/OpenAI-compatible extra parameters
    extra_body={
        "keep_alive": "30m",
        "think": False,   # closest equivalent of reasoning=False for Qwen thinking models
    },
)
llm = ChatOpenAI(
    model="qwen3:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.0,
    max_tokens=2048,   # OpenAI-style replacement for num_predict
    timeout=180,

    # Ollama/OpenAI-compatible extra parameters
    extra_body={
        "keep_alive": "30m",
        "think": False,   # closest equivalent of reasoning=False for Qwen thinking models
    },
)
summary_llm = ChatOpenAI(
        model="qwen3:latest",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.0,
        max_tokens=8192,   # OpenAI-style replacement for num_predict
        timeout=180,

        # Ollama/OpenAI-compatible extra parameters
        extra_body={
            "keep_alive": "30m",
            "think": False,   # closest equivalent of reasoning=False for Qwen thinking models
        },
)

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
