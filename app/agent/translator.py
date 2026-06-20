# ── Translator node ──
# Uses normalizer_llm to normalize Hinglish/Hindi/Gujarati queries to clean English.
# Runs before semantic_search.

import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.schemas.state import MainState
from app.core.config import normalizer_llm
from app.utils.utils import log_token_usage, extract_json_object
from app.prompts.prompts import TRANSLATOR_PROMPT_BASE
import logging

logger = logging.getLogger("erp_assistant.translator")
_translation_cache: dict[str, dict] = {}
_TRANSLATION_CACHE_MAX = 128

def _looks_tokenized_query_parts(parts: list[str]) -> bool:
    if not parts or len(parts) <= 2:
        return False
    single_word_count = sum(1 for p in parts if len(str(p).split()) == 1)
    return single_word_count / len(parts) >= 0.7


def _resolve_pronouns(query: str, conv_ctx: dict | None, last_tool: dict | None) -> tuple[str, list[dict]]:
    """Pronoun resolution is handled by the translator LLM via the CONVERSATION CONTEXT section in the prompt."""
    return query, []



def _build_translator_prompt(
    conversation_context: dict | None = None,
    last_tool_call: dict | None = None,
    summary: str | None = None,
    messages: list | None = None,
) -> str:
    lines = [TRANSLATOR_PROMPT_BASE.rstrip()]
    ctx_lines = []
    if summary:
        ctx_lines.append(f"- Conversation summary: {summary}")
    if last_tool_call:
        for tool_name, args in last_tool_call.items():
            safe_args = {k: v for k, v in args.items() if k in ("filters", "search", "term", "invoiceNo")}
            ctx_lines.append(f"- Last tool: {tool_name} with {json.dumps(safe_args)}")
    if conversation_context:
        entities = conversation_context.get("entities", [])
        names = [e.get("name", "") for e in entities if e.get("name")]
        if names:
            ctx_lines.append(f"- Known entities: {', '.join(names[-5:])}")
        failures = conversation_context.get("tool_failures", [])
        for f in failures[-3:]:
            ctx_lines.append(f"- {f.get('tool')} returned no data for '{f.get('entity')}'")

    if messages:
        # Get last 4 messages to avoid context bloat but provide short-term chronology
        recent_msgs = messages[-4:]
        msg_lines = []
        for msg in recent_msgs:
            if isinstance(msg, HumanMessage):
                msg_lines.append(f"  User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    tc_names = [tc.get("name") for tc in msg.tool_calls if tc.get("name")]
                    msg_lines.append(f"  Assistant called tools: {', '.join(tc_names)}")
                elif msg.content:
                    msg_lines.append(f"  Assistant: {msg.content}")
        if msg_lines:
            ctx_lines.append("- Recent dialogue history (in chronological order):\n" + "\n".join(msg_lines))

    if ctx_lines:
        lines.append("")
        lines.append("CONVERSATION CONTEXT (resolve pronouns and sequential references using this):")
        lines.extend(ctx_lines)
        lines.append("")
        lines.append("PRONOUN & CONTEXT RESOLUTION RULES (CRITICAL):")
        lines.append("- If the current query has pronouns (uska, iska, unka, iski, inki, uska, woh, uss, is, es, in sab, its, this, that, these, those), replace them with the actual entity name from the CONTEXT above.")
        lines.append("- If the query refers to a previous action or query (e.g. 'usse pehle', 'before that', 'then that one', 'what did I ask?'), resolve it to the corresponding query/intent from the 'Recent dialogue history' or 'Conversation summary' above.")
        lines.append("- If the query has multiple independent intents, output each as a separate item in query_parts[].")
        lines.append("- Example: context has PR-269, query='to uska taxable amount kitna hai? aur april ka gst'")
        lines.append("  → query_parts: ['Show taxable amount of PR-269', 'Show GST details for April']")
    return "\n".join(lines)

@traceable(name="translator_node", run_type="chain")
async def translator_node(state: MainState) -> MainState:
    try:
        logger.info("Translator node started", extra={"node": "translator"})
        user_query = state.get("user_query", "") or ""
        import string
        user_query = user_query.strip().rstrip(string.punctuation + "/\\")
        if not user_query:
            return {
                "original_query": "",
                "canonical_query": "",
                "user_query": "",
                "translator_used": False,
                "translator_confidence": "low",
                "detected_language": "unknown",
                "document_type": "unknown",
                "query_type": "unknown",
            }
        resolved_query, pre_resolved_entities = _resolve_pronouns(
            user_query,
            state.get("conversation_context"),
            state.get("last_tool_call"),
        )
        if pre_resolved_entities:
            print(f"[PRONOUN] Resolved pronouns: {pre_resolved_entities}")
            print(f"[PRONOUN] Original: {user_query} → Resolved: {resolved_query}")
            user_query = resolved_query
        ctx = state.get("conversation_context") or {}
        ltc = state.get("last_tool_call") or {}
        summary = state.get("summary") or ""
        cache_key = f"{user_query}::{summary[:100] if summary else ''}"
        cached = _translation_cache.get(cache_key)
        if cached:
            data = cached
        else:
            prompt = _build_translator_prompt(
                conversation_context=ctx,
                last_tool_call=ltc,
                summary=summary,
                messages=state.get("messages", []),
            )
            response = await normalizer_llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_query),
            ])
            input_text = prompt + "\n" + user_query
            log_token_usage(response, "translator", input_text=input_text, output_text=response.content)
            data = extract_json_object(response.content)
            if len(_translation_cache) >= _TRANSLATION_CACHE_MAX:
                _translation_cache.pop(next(iter(_translation_cache)))
            _translation_cache[cache_key] = data
        canonical_query = data.get("canonical_query") or user_query
        language = data.get("language", "mixed")
        if canonical_query.lower().strip() == user_query.lower().strip() and language != "english":
            print(f"[TRANSLATOR] Same output as input — treating as untranslated: {canonical_query}")
            return {
                "original_query": user_query,
                "canonical_query": user_query,
                "user_query": user_query,
                "translator_used": False,
                "translator_confidence": "low",
                "detected_language": "unknown",
                "document_type": "unknown",
                "query_type": "unknown",
                "query_intent": "sample",
            }
        language = data.get("language", "mixed")
        if language == "hindi" and not re.search(r'[\u0900-\u097F]', user_query):
            language = "hinglish"
        confidence = data.get("confidence", "medium")
        query_type = data.get("query_type", "")
        query_intent = data.get("query_intent", "sample")
        query_parts = data.get("query_parts") or []
        llm_resolved = data.get("resolved_entities") or []
        resolved_entities = (pre_resolved_entities or []) + (llm_resolved or [])
        final_canonical = user_query if query_type == "conversational" else canonical_query
        if _looks_tokenized_query_parts(query_parts):
            print(f"[TRANSLATOR FIX] Tokenized query_parts detected: {query_parts} -> using canonical query")
            query_parts = [final_canonical]
        elif not query_parts:
            query_parts = [final_canonical]
        print("Original query:", user_query)
        print("Canonical query:", canonical_query)
        print("Detected language:", language)
        print("Translator confidence:", confidence)
        print("Query type:", query_type)
        print("Query intent:", query_intent)
        if query_parts:
            print("Query parts:", query_parts)
        if resolved_entities:
            print("Resolved entities:", resolved_entities)
        doc_type = data.get("document_type", "unknown")
        return {
            "original_query": user_query,
            "canonical_query": final_canonical,
            "user_query": final_canonical,
            "translator_used": True,
            "translator_confidence": confidence,
            "detected_language": language,
            "document_type": doc_type,
            "query_type": query_type,
            "query_intent": query_intent,
            "query_parts": query_parts,
            "resolved_entities": resolved_entities,
        }
    except Exception as e:
        print(f"Translator failed: {e}")
        user_query = state.get("user_query", "") or ""
        return {
            "original_query": user_query,
            "canonical_query": user_query,
            "user_query": user_query,
            "translator_used": False,
            "translator_confidence": "low",
            "detected_language": "unknown",
            "document_type": "unknown",
            "query_type": "unknown",
            "query_intent": "sample",
            "query_parts": [user_query],
        }
