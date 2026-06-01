from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq # type: ignore
# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
# embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

token = os.environ.get("GROQ_API_KEY")
# token = os.environ.get("HUGGINGFACE_API_KEY")

embedding_model = OllamaEmbeddings(model = "bge-m3")


normalizer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=token,
)

# model = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",huggingfacehub_api_token=token,temperature=0.0)
# llm = ChatHuggingFace(llm=model)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=token,
)
# llm = ChatOllama(
#     model="qwen3:8b-q4_K_M",
#     base_url="http://localhost:11434",
#     temperature=0.0,
#     keep_alive="30m",
#     num_ctx=2048,
# )
# llm = ChatOllama(
#     model="Phi4-mini:latest",
#     temperature=0.0,
#     keep_alive="30m",
#     num_ctx=2048,
# )

# router_llm = ChatOllama(
#     model="qwen3:14b-q4_K_M ",
#     temperature=0.0,
#     keep_alive="30m",
#     num_ctx=2048,
# )

print("LLM and embedding model initialised!")


CHP1_API_BASE_URL = os.getenv("CHP1_API_BASE_URL","https://dev.chapter1.finance/aiAnalytics/")
CHP1_API_TOKEN = os.getenv("CHP1_API_TOKEN", "")
# CHP1_API_TIMEOUT = int(os.getenv("CHP1_API_TIMEOUT", "30"))
CHP1_API_TIMEOUT = int(os.getenv("CHP1_API_TIMEOUT", "10"))

COMPANY_ID = int(os.getenv("COMPANY_ID", "355"))

