import re
from collections import Counter
from langsmith import traceable
from src.schema import MainState
from src.config import embedding_model
from src.tool_doc import TOOL_INTENT_REGISTRY
from src.utils import (
    _cosine_sim, _build_tool_embeddings, _tool_embeddings,
    normalize_text, add_unique, TH_EMBEDDING_RECALL_MIN, TH_RERANKER_TOP_K,
)
# --- COMMENTED OUT (zero-regex migration): word-list based routing ---
# from src.utils import CONNECTORS, ROUTE_KEYWORDS, DOMAIN_KEYWORDS, INVOICE_PATTERNS, TOOL_DOMAINS, ERP_AMBIGUOUS_THRESHOLD, max_erp_similarity, is_plain_english_query
# from src.prompts import META_QUESTION_PATTERNS_GLOBAL, GREETING_PATTERNS, CAPABILITY_PATTERNS, OOD_TOPICS, _STOP_WORDS, VAGUE_ACTION_WORDS
from src.tools_api import tools_dict
from langchain_core.messages import AIMessage


# --- COMMENTED OUT (zero-regex migration): domain classification ---
# def classify_domains(query: str, resolved_entities: list | None = None) -> tuple[set[str], set[str]]:
#     ...

# --- COMMENTED OUT (zero-regex migration): count/aggregate/intent classification ---
# _COUNT_LEMMAS = {...}
# _COUNT_AUX = {...}
# def _is_count_only(part: str) -> bool: ...
# def _merge_count_parts(parts: list[str]) -> list[str]: ...
# def _filter_registry_by_domain(domains: set[str]) -> dict: ...
# def _classify_intent(*queries: str) -> str: ...


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


# --- COMMENTED OUT (zero-regex migration): not needed with translator query_parts ---
# def split_query_parts(query: str) -> list[str]: ...
# def split_query_for_tools(...) -> list[str]: ...
# def _keyword_fallback(...) -> list[str]: ...
# def merge_unique_tools(...) -> list[str]: ...
# def is_multi_intent_query(...) -> bool: ...
# def _filter_for_list_intent(...) -> list[str]: ...
# def _keyword_whitelist_filter(...) -> list[str]: ...
# def _detect_conflict_groups(...) -> list[set[str]]: ...
# def _apply_mutual_exclusion(...) -> list[str]: ...
# def _looks_tokenized_query_parts(...) -> bool: ...
# def _matches_ood_topic(...) -> bool: ...
# def _get_all_domain_words() -> set[str]: ...
# def _has_domain_content(parts: list[str]) -> bool: ...


@traceable(name="semantic_search_node", run_type="retriever")
async def semantic_search(state: MainState) -> MainState:
    try:
        print("→ semantic_search")
        original_query = state.get("original_query") or state.get("user_query", "") or ""
        canonical_query = state.get("canonical_query", "") or ""
        user_query = canonical_query or original_query

        if not user_query:
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True}

        print(f"Original query: {original_query}")
        print(f"Canonical query: {canonical_query}")

        query_type = (state.get("query_type") or "").strip()

        # Translator-classified non-ERP queries get routed directly
        if query_type in ("greeting", "capability", "conversational", "ood", "ambiguous") and state.get("translator_used"):
            print(f"Translator classified as {query_type} — no tools needed: {user_query}")
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True, "query_type": query_type}

        query_parts = state.get("query_parts") or [user_query]
        print(f"Query parts for tool selection: {query_parts}")

        selected_tool_groups: list[list[str]] = []

        for part in query_parts:
            # Pure embedding-based reranker — no regex, no word-lists
            tools_for_part = score_tools_via_reranker(part, TOOL_INTENT_REGISTRY)
            if tools_for_part:
                print(f"Reranker tools for part '{part}': {tools_for_part}")
                selected_tool_groups.append(tools_for_part)
            else:
                print(f"No tools found for part '{part}' via reranker")

        # Flatten and deduplicate
        seen = set()
        selected_tools = []
        for group in selected_tool_groups:
            for t in group:
                if t not in seen:
                    seen.add(t)
                    selected_tools.append(t)

        selected_tools = [t for t in selected_tools if t in tools_dict]

        MAX_TOOLS = 6
        if len(selected_tools) > MAX_TOOLS:
            print(f"Trimming selected_tools from {len(selected_tools)} to {MAX_TOOLS}")
            selected_tools = selected_tools[:MAX_TOOLS]

        if selected_tools:
            print(f"Final selected tools: {selected_tools}")
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
