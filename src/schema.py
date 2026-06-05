from typing import Annotated, List, TypedDict, Dict, Any
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    user_query : str


class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    retrieved_tools: List[str]
    loop_count: int
    final_response: str
    tools_utilized: List[str]       
    step_timings: Annotated[List[Dict[str, Any]], add]

    router_decision: Dict[str, Any]
    selected_tools: List[str]
    query_parts: List[str]
    canonical_query: str
    translator_used: bool
    translator_confidence: str
    document_type: str # to map to our sales/purchase/product tools
    detected_language: str
    skip_router: bool
    unsupported_parts: list[str]
    summary:str #Summary of the conversation so far, to be prepended to the prompt in each loop iteration. Updated after each iteration with the latest summary from the LLM.
    response_text: str
    last_tool_call: dict  # persists last tool call per tool name across summarization
    conversation_context: dict  # persists entity references (customers, products, etc.) across summarization
    memory_answer: str  # stores memory-only query answer so routing can continue to response_generation
    original_query: str  # stores the original user query before canonicalization
    unsupported_reason: str  # stores reason when query is out of domain
    
class OutputState(TypedDict):
    final_response: str
    tools_utilized: List[str]
    step_timings: List[Dict[str, Any]]