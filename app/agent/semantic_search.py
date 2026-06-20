import re
from collections import Counter
from langsmith import traceable
from app.schemas.state import MainState
from app.core.config import embedding_model
from app.prompts.tool_doc import TOOL_INTENT_REGISTRY
from app.utils.utils import (
    _cosine_sim, _build_tool_embeddings, _tool_embeddings,
    normalize_text, add_unique, TH_EMBEDDING_RECALL_MIN, TH_RERANKER_TOP_K,
)

from app.services.tools_api import tools_dict
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger("erp_assistant.semantic_search")




def score_tools_via_reranker(query_part: str, registry: dict) -> list[str]:
    if not _tool_embeddings:
        _build_tool_embeddings()
    if not _tool_embeddings:
        return []
    try:
        query_emb = embedding_model.embed_query(query_part)
    except Exception:
        return []
    scores = []
    for tool_name in registry:
        if tool_name not in _tool_embeddings:
            continue
        tool_emb = _tool_embeddings[tool_name]
        emb_sim = _cosine_sim(query_emb, tool_emb)
        scores.append((tool_name, emb_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return []
    top_k = scores[:TH_RERANKER_TOP_K]
    result = []
    for tool_name, emb_sim in top_k:
        if tool_name not in result and emb_sim >= TH_EMBEDDING_RECALL_MIN:
            result.append(tool_name)
    return result





@traceable(name="semantic_search_node", run_type="retriever")
async def semantic_search(state: MainState) -> MainState:
    try:
        logger.info("Semantic search started", extra={"node": "semantic_search"})
        original_query = state.get("original_query") or state.get("user_query", "") or ""
        canonical_query = state.get("canonical_query", "") or ""
        user_query = canonical_query or original_query

        if not user_query:
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True}

        query_type = (state.get("query_type") or "").strip()

        # Translator-classified non-ERP queries: verify via embedding reranker before skipping
        if query_type in ("greeting", "capability", "conversational", "ood", "ambiguous") and state.get("translator_used"):
            if not _tool_embeddings:
                _build_tool_embeddings()
            if _tool_embeddings:
                try:
                    query_emb = embedding_model.embed_query(user_query)
                    scores = [
                        _cosine_sim(query_emb, _tool_embeddings[t])
                        for t in TOOL_INTENT_REGISTRY if t in _tool_embeddings
                    ]
                    max_score = max(scores) if scores else 0.0
                except Exception:
                    max_score = 0.0
            else:
                max_score = 0.0

            if max_score >= 0.65:
                tools_for_part = score_tools_via_reranker(user_query, TOOL_INTENT_REGISTRY)
                if tools_for_part:
                    print(f"Reranker found tools (max_score={max_score:.3f}) for {query_type} query — overriding to erp_query: {tools_for_part}")
                    query_type = "erp_query"
                else:
                    print(f"Translator classified as {query_type} (max_score={max_score:.3f}) — no tools via reranker: {user_query}")
                    return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True, "query_type": query_type}
            else:
                print(f"Translator classified as {query_type} (max_score={max_score:.3f}) — no tools needed: {user_query}")
                return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True, "query_type": query_type}

        query_parts = state.get("query_parts") or [user_query]
        # print(f"Query parts for tool selection: {query_parts}")
        logger.info(
                        "Semantic search query prepared",
                        extra={
                            "node": "semantic_search",
                            "original_query": original_query,
                            "canonical_query": canonical_query,
                            "query_parts": query_parts,
                        },
                    )

        selected_tool_groups: list[list[str]] = []

        for part in query_parts:
            # Pure embedding-based reranker — no regex, no word-lists
            tools_for_part = score_tools_via_reranker(part, TOOL_INTENT_REGISTRY)
            if tools_for_part:
                logger.info(
                                "Reranker tools selected for query part",
                                extra={
                                    "node": "semantic_search",
                                    "query_part": part,
                                    "selected_tools": tools_for_part,
                                },
                            )
                selected_tool_groups.append(tools_for_part)
            else:
                logger.info(
                                "No tools selected for part",
                                extra={
                                    "node": "semantic_search",
                                    "query_part": part,
                                    "selected_tools": tools_for_part,
                                },
                            )
        # Interleave from groups so each query part gets represented
        seen = set()
        selected_tools = []
        query_intent = state.get("query_intent", "sample")
        if len(query_parts) > 1:
            MAX_TOOLS = 6
        elif query_intent in ("count", "aggregate", "list_all", "extreme"):
            MAX_TOOLS = 3
        elif query_intent in ("comparison", "detail"):
            MAX_TOOLS = 4
        else:
            MAX_TOOLS = 6
        groups = [g for g in selected_tool_groups if g]

        if groups:
            max_per_part = max(1, min(MAX_TOOLS // max(len(groups), 1), 3))
            for group in groups:
                count = 0
                for t in group:
                    if t not in seen and len(selected_tools) < MAX_TOOLS:
                        seen.add(t)
                        selected_tools.append(t)
                        count += 1
                        if count >= max_per_part:
                            break
            for group in groups:
                for t in group:
                    if t not in seen and len(selected_tools) < MAX_TOOLS:
                        seen.add(t)
                        selected_tools.append(t)

        selected_tools = [t for t in selected_tools if t in tools_dict]

        if selected_tools:
            logger.info(
                            "Final tools selected",
                            extra={
                                "node": "semantic_search",
                                "selected_tools": selected_tools,
                            },
                        )
            return {
                "retrieved_tools": selected_tools,
                "selected_tools": selected_tools,
                "query_parts": query_parts,
                "skip_router": True,
            }

        # Before marking ambiguous, check if query has ERP context from conversation history
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name")
                    if tool_name and tool_name in tools_dict:
                        print(f"Using tool from conversation history: {tool_name}")
                        return {
                            "retrieved_tools": [tool_name],
                            "selected_tools": [tool_name],
                            "query_parts": query_parts,
                            "skip_router": True,
                        }

        print(f"No tools selected — routing to ambiguous handler: {user_query}")
        return {
            "retrieved_tools": [],
            "selected_tools": [],
            "query_parts": query_parts,
            "skip_router": True,
            "query_type": "ambiguous",
        }

    except Exception as e:
        print(f"Error in semantic search node: {e}")
        return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True}
