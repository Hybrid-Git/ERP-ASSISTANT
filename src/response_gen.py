import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import summary_llm
from src.utils import LIST_WORDS, TOOL_DOMAINS
from src.prompts import HINGLISH_PRONOUNS
from src.deterministic_final import make_summary


def _clean_llm_response(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*\n.*?\n```', '', text, flags=re.DOTALL)
    text = text.strip()
    meta_prefixes = (
        "based on the provided", "based on the data", "based on the tool",
        "here are the key", "here is the summary", "here is the detail",
        "from the tool results", "from the provided", "according to the tool",
        "the tool results show", "the data shows that", "the following are",
        "below are the", "i have analyzed the", "after reviewing the",
        "in response to your",
    )
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        if any(stripped.startswith(p) for p in meta_prefixes) and len(stripped) < 80:
            continue
        if stripped in ("json", "```json", "```", "````"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    text = re.sub(r'```\s*$', '', text).strip()
    text = re.sub(r'\{[^}]*"[^}]*"[^}]*\}', '', text).strip()
    if not text or not text.strip("` \n\r"):
        return ""
    return text


def _strip_hallucinated_values(text: str, tool_data: dict, original_query: str = "") -> str:
    actual_fields = set()
    for _tool_name, records in tool_data.items():
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict):
                    for k in rec:
                        actual_fields.add(k.lower())
    if not actual_fields:
        return text
    query_tokens = set(re.findall(r'\b\w+\b', original_query.lower()))
    field_like = re.compile(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b|\b[a-zA-Z]+_[a-zA-Z]+\b')
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    cleaned = []
    for sent in sentences:
        if not sent.strip():
            continue
        words = set(m.group(0).lower() for m in field_like.finditer(sent))
        hallucinated = any(
            w not in actual_fields and w not in query_tokens
            for w in words
        )
        if hallucinated:
            print(f"[HALLUCINATION] Stripped sentence containing invented field(s): {words - actual_fields} | sentence: {sent.strip()[:100]}")
        else:
            cleaned.append(sent)
    result = " ".join(cleaned).strip()
    if not result:
        print(f"[HALLUCINATION] All sentences stripped — returning empty")
    return result


@traceable(name="response_generation_node", run_type="chain")
async def response_generation_node(state: MainState):
    memory_answer = state.get("memory_answer", "")
    final_response = state.get("final_response", {})
    if memory_answer:
        query_type = state.get("query_type", "")
        if query_type in ("capability", "ambiguous", "greeting", "ood", "conversational"):
            return {"response_text": memory_answer, "memory_answer": ""}
        tool_data = final_response.get("data", {}) if isinstance(final_response, dict) else {}
        has_real_data = any(
            isinstance(recs, list) and len(recs) > 0
            for recs in tool_data.values()
        )
        if not has_real_data:
            return {"response_text": memory_answer, "memory_answer": ""}
        print(f"[RESPONSE_GEN] Skipping memory_answer — real tool data exists")
        memory_answer = ""
    messages = state.get("messages", [])

    original_query = (
        state.get("original_query", "")
        or state.get("user_query", "")
        or ""
    )
    if not original_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                original_query = getattr(msg, "content", "") or ""
                break
    detected_language = state.get("detected_language") or "auto"

    if detected_language not in ("hinglish", "hindi"):
        hinglish_words = {"batao", "chaia", "wale", "ka", "ki", "kya", "hai", "kitne", "konse", "konsa", "karli", "hua", "hue"}
        tokens = re.findall(r"\w+", original_query.lower())
        if any(w in tokens for w in hinglish_words):
            detected_language = "hinglish"

    previous_summary = state.get("summary", "") or ""
    conversation_context = state.get("conversation_context", {})

    list_mode = bool(re.search(
        r'\b(' + '|'.join(re.escape(w) for w in LIST_WORDS) + r')\b',
        original_query, re.IGNORECASE
    ))

    system_prompt = (
        "You are an ERP assistant. Write a SHORT natural conversational reply using ONLY the tool results below.\n"
        "\n"
        "PERSONALITY:\n"
        "- Be warm and conversational — like a helpful human, not a robot.\n"
        "- Vary your phrasing. Don't repeat the same sentence patterns.\n"
        "- If the user used Hinglish/Hindi, mirror their language and tone.\n"
        "- Never say 'As an AI' or 'As a language model' — just be helpful.\n"
        "- Only end with a question (like 'Kuch aur?' or 'Anything else?') if the user's query itself ended with a question mark ('?'). Otherwise end with '.'.\n"
        "- If the user asks about something you already covered earlier, "
        "refer back naturally: 'Jaisa aapne pehle pucha tha...'\n"
        "\n"
        "PRIORITY RULE: When multiple tools returned results, focus on the tool(s) most relevant to the user's actual question. "
        "Ignore tool outputs that don't relate to what the user asked about. "
        "For example, if the user asked about 'Bangalore customers' and `get_stock_levels` also returned data, just answer about the customer.\n"
        "\n"
        "HARD RULES:\n"
        "0. CRITICAL — NEVER output raw JSON, JSON blocks (```json … ```), or any data dump. "
        "Your ENTIRE reply MUST be a plain conversational paragraph in the user's language. "
        "NO exceptions. If you are tempted to include JSON, stop and write natural sentences instead.\n"
    )
    if previous_summary:
        system_prompt += f"\nFor background, the conversation history is:\n{previous_summary}\n\n"
    if conversation_context:
        entities = conversation_context.get("entities", [])
        if entities:
            system_prompt += f"KNOWN ENTITIES:\n{json.dumps(entities[-3:], indent=2, ensure_ascii=False)}\n\n"
    if detected_language == "hinglish":
        system_prompt += (
            "1. LANGUAGE: The user wrote in Hinglish (Hindi words written with English letters).\n"
            "   Your ENTIRE reply MUST use ONLY a-z A-Z 0-9 and basic punctuation (. , ? !).\n"
            "   Do NOT use Devanagari (Hindi script), Chinese, or any other non-Latin characters.\n"
            "   Write Hindi words with English letters: 'aap', 'hai', 'nahi', 'se', 'ka', 'kaunsa'.\n"
            "   Use the exact same words the user used when possible.\n"
            "\n"
            "EXAMPLE:\n"
            "  User: muje customer details chaie\n"
            "  Tool result: Customer name: Rohan\n"
            "  Correct reply: aapke customer Rohan hai\n"
            "  WRONG: आपके ग्राहक रोहन हैं (NO Devanagari at all)\n"
        )
    elif detected_language == "hindi":
        system_prompt += (
            "1. LANGUAGE: The user wrote in Hindi. Reply in Hinglish (Hindi words with English letters).\n"
            "   Your ENTIRE reply MUST use ONLY a-z A-Z 0-9 and basic punctuation (. , ? !).\n"
            "   Do NOT use Devanagari (Hindi script), Chinese, or any other non-Latin characters.\n"
            "   Write Hindi words with English letters: 'aap', 'hai', 'nahi', 'se', 'ka'.\n"
            "   Use the exact same words the user used when possible.\n"
            "\n"
            "EXAMPLE:\n"
            "  User: muje customer details chaie\n"
            "  Tool result: Customer name: Rohan\n"
            "  Correct reply: aapke customer Rohan hai\n"
            "  WRONG: आपके ग्राहक रोहन हैं (NO Devanagari at all)\n"
        )
    else:
        system_prompt += (
            "1. LANGUAGE: The user wrote in English. Reply in English.\n"
        )
    system_prompt += (
        "2. Tool results are definitive answers. NEVER end your reply with '?' — use '.' even when stating a fact.\n"
        "   WRONG: aapke customer Rohan hai?\n"
        "   CORRECT: aapke customer Rohan hai.\n"
        "3. NEVER invent field names, values, IDs, or numbers. If a value is missing, say so plainly.\n"
        "   The TOOL RESULTS JSON below is the ONLY source of truth. Do NOT add IDs, names, or values not present in it.\n"
        "   WRONG: \"productId F12 ka quantity 5 se jada hai\" (when tool results have no productId or F12)\n"
        "   CORRECT: \"is baare mein data nahi mila\" or \"closingQty wala column nahi tha\"\n"
        "4. NO headings of any kind. NO 'Summary:', 'Details:', 'Sample Record:', "
        "'Additional Information:', 'Key Points:', 'Sample Invoice Details:', 'Outstanding X Summary:' — "
        "zero headings. Just plain conversational sentences.\n"
        "   WRONG: 'Sales Invoices Summary: AI/0324/0010 ka netAmount 29315.7 hai'\n"
        "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai'\n"
        "5. Do NOT use bullet points or numbered lists unless the user explicitly asked for a list.\n"
        "6. Answer ONLY what was asked. No analysis, advice, recommendations, "
        "'Next Steps:', 'Additional Insights:', or 'Key Points'. "
        "If the user asked for netAmount, give netAmount and nothing else. "
        "1-3 sentences max.\n"
        "   WRONG: 'Additional Insights: The invoices are from Oct-Dec 2024. Next Steps: Review high amounts.'\n"
        "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai, ledger CGLAM LIFESTYLE hai.'\n"
        "7. Never begin your response with 'Based on the provided', 'Based on the data', "
        "'Here are the key points', 'Additional Insights', 'Summary Breakdown', "
        "'Suggestions', 'Example Queries', 'Here is the summary', or any similar meta-framing. "
        "Start directly with the answer to the user's query. "
        "The 'For context' note below is for your reference only — do not repeat it or comment on it.\n"
        "   WRONG: 'Based on the provided summary, here are some key points and suggestions:'\n"
        "   CORRECT: 'aapke sales invoice mai ledger names hain: B2C_ANDHRA PRADESH, Hirva Beauty...'\n"
        "8. If TOOL RESULTS are empty [] for the relevant tool, that means no data was found. "
        "Say so plainly: 'data nahi mila' or 'kuch nahi mila' in the user's language. "
        "Do NOT repeat back any entity name, company name, ID, or invoice number that the user mentioned — "
        "the tool found nothing, so there is nothing to confirm. Do NOT say records are hidden.\n"
        "9. If the tool results contain MULTIPLE records that match the user's query (e.g. same invoiceNo from different parties/dates), "
        "report ALL of them. Never omit any. List each with its distinguishing fields so the user can tell them apart.\n"
        "   WRONG: 'PR-269 ka net amount 3292.7 hai' (only one of two records)\n"
        "   CORRECT: 'PR-269 ke 2 records hain: Bigfoot se 3292.7 aur Amazon se 1215.95.'\n"
        "10. If the user's query has MULTIPLE distinct parts (e.g. 'TDS aur B2B', 'sales aur purchase'), "
        "your FIRST response MUST call a SEPARATE tool for EACH part — never skip a part. "
        "You have the full tool list; use every tool that matches a query part.\n"
        "11. When one tool found the exact record the user asked for (e.g. specific invoiceNo), "
        "answer ONLY from that tool's data. Ignore all other tool outputs completely. "
        "Do NOT describe, summarize, or mention any other tool's results.\n"
        "   WRONG: 'Purchase invoices have 684 records but Sales has AI/0324/0010 with netAmount 29315.7'\n"
        "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai' (purchase data not mentioned at all)\n"
        "    When the user asks for 'detail', 'sara detail', 'all details', or 'full info', "
        "include EVERY field value from the tool results in your reply (voucherCount, taxableAmount, "
        "igst, cgst, sgst, cess, tax, invoiceAmount, etc.). Do NOT cherry-pick only 1-2 fields.\n"
        "12. If the user asks for the ID of a category/item (e.g. 'X ka id', 'find id of X'), "
        "use the search-ledger tool. Set `groupType` based on the noun: "
        "expense for office expenses/salary/rent, party for customers/vendors, "
        "asset for fixed assets. Infer the noun from the query.\n"
        "13. B2B / B2C (Small/Large) / Exports / Nil-Rated / Exempt are GST categories, "
        "NOT sales categories. When the user asks for B2B, B2C, or any GST-category data, "
        "count, or split, ALWAYS use `get_gst_summary`. It returns ALL categories together — "
        "find the relevant one (b2b, b2cSmall, b2cLarge, exports, nilRated) in the result data. "
        "Do NOT use `get_sales_summary` for B2B or B2C data — it returns overall sales totals, "
        "not the GST category breakdown.\n"
         "14. LIST TRUNCATION — CONVERSATIONAL: When truncation_info shows more records exist "
         "than shown, just mention the total count naturally and ask if they want to see some. "
         "Do NOT list any individual records when total > 10 — "
         "just say how many were found and offer. "
         "When total <= 10, show 2-3 records as a short sample then offer the rest. "
         "Vary your phrasing each time — never repeat the same sentence pattern. "
         "Do NOT say 'Add a filter' — the user didn't ask for that.\n"
         "    EXAMPLES:\n"
         "      'B2C_PUNJAB ke 293 outstanding ITEM invoices hain. Kuch dikhaun?'\n"
         "      'That's the first 5 of 120. Want to see the rest?'\n"
         "      'Aapke 27 records hain. Dikhaun?'\n"
         "    WRONG: 'Showing first 10 of 854. Add a filter.' (too robotic)\n"
        "15. FIELD DISPLAY RULES:\n"
        "    - For list queries (default): show max 5 most important fields per record. "
        "Pick the 5 fields most relevant to the user's question (e.g. id, name, "
        "invoiceNo, netAmount, outstanding).\n"
        "    - For records with 5 or fewer fields: show all fields.\n"
        "    - When the user asks for 'all details' / 'sabhi detail' / 'full info' / "
        "'sara detail' (already covered by rule 11): show EVERY field — this overrides "
        "the 5-field limit completely.\n"
        "16. NEVER output Python code, SQL, or any code. "
         "NEVER invent field names or values not present in TOOL RESULTS. "
         "NEVER write 'Key Observations', 'Next Steps', 'Data Validation', "
         "'Potential Use Cases', 'Sample Records', or any section headings — "
         "just answer conversationally and stop.\n"
          "17. CRITICAL — You can ONLY see the records shown in TOOL RESULTS. "
          "If more records exist (truncation_info mentions them), you may say so "
          "and offer to show them, but you MUST NOT describe, summarize, or invent "
          "ANY data for records beyond what you can actually see. "
          "No totals, no aggregates, no 'sabse zyada' or 'sabse kam' — "
          "only what is directly visible in TOOL RESULTS. "
          "If asked to 'show more', say you need to fetch them first.\n"
          "18. CONVERSATIONAL ENDING — Always end your reply with a natural follow-up "
          "invitation like 'Anything else?' or 'Kuch aur?' or 'Aur kya dekhna hai?' "
          "unless the user's query was a direct command, a simple yes/no answer, "
          "or a request that clearly does not invite follow-up (like 'thank you', 'ok', 'bye'). "
          "Do NOT force it when it feels unnatural.\n"
           "/no_think\n"
    )
    if list_mode:
        system_prompt += (
            "19. LIST MODE — The user asked for a list or multiple records. "
            "Format each record as a bullet point ('- '). "
            "Show max 1-2 most relevant fields per record (e.g. just name, or name + outstanding). "
            "After the bullets, naturally mention the total count and ask if they want to see more.\n"
            "    Example:\n"
            "      - Masjid To Churchgate\n"
            "      - Vasai-Dahisar\n"
            "      - Ranjeet\n\n"
            "      Ye 100 records mein se 10 hain. Aur dikhaun?\n"
            "    If 0 records found, just say 'kuch nahi mila' — no bullets.\n"
        )

    final_response_prompt = dict(final_response)
    truncation_info = final_response_prompt.pop("truncation_info", {}) or {}
    summary_text = final_response_prompt.pop("summary", "") or ""
    final_response_prompt.pop("tools_used", None)
    summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
    summary_text = re.sub(r'; [^:]+:\s*', '; ', summary_text)
    data = final_response_prompt.get("data", {})

    doc_type = (state.get("document_type", "") or "").lower()
    if doc_type == "general":
        orig_q = (state.get("original_query", "") or "").lower()
        is_follow_up = any(re.search(rf'\b{re.escape(p)}\b', orig_q) for p in HINGLISH_PRONOUNS)
        if is_follow_up:
            customer_vendor_domains = {"customer", "vendor"}
            filtered = {}
            for tool_name, records in data.items():
                td = set(TOOL_DOMAINS.get(tool_name, []))
                if td & customer_vendor_domains:
                    filtered[tool_name] = records
                else:
                    print(f"[DOMAIN-FILTER] {tool_name}: {len(records) if isinstance(records, list) else 'non-list'} records removed (domain {td} not in customer/vendor)")
            if filtered:
                final_response_prompt["data"] = filtered

    detail_mode = bool(re.search(
        r'\b(all details?|sabhi detail|saari detail|full info|complete info|sara detail|poora detail|saare detail)\b',
        original_query, re.IGNORECASE
    ))

    invoice_match = final_response_prompt.pop("_invoice_match", None)
    if invoice_match and isinstance(invoice_match, dict):
        matched_tool = invoice_match.get("tool_name", "")
        if matched_tool and matched_tool in final_response_prompt.get("data", {}):
            filtered_data = {matched_tool: final_response_prompt["data"][matched_tool]}
            final_response_prompt["data"] = filtered_data
            summary_text = make_summary(filtered_data, [])
            summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
            summary_text = re.sub(r'; [^:]+:\s*', '; ', summary_text)
    final_response_prompt.pop("summary", None)
    final_response_prompt.pop("_invoice_match", None)
    data = final_response_prompt.get("data", {})

    MAX_SAMPLE = 20 if detail_mode else (10 if list_mode else 5)
    for tool_name, records in data.items():
        if isinstance(records, list) and len(records) > MAX_SAMPLE:
            data[tool_name] = records[:MAX_SAMPLE]

    # Sync truncation_info shown count with actual data after further truncation
    if truncation_info:
        for tool_name in truncation_info:
            tool_records = data.get(tool_name, [])
            if isinstance(tool_records, list):
                truncation_info[tool_name]["shown"] = len(tool_records)
    truncation_note = ""
    if truncation_info:
        total = sum(v.get("total", 0) for v in truncation_info.values())
        shown = sum(v.get("shown", 0) for v in truncation_info.values())
        remaining = total - shown
        truncation_note = f"\nMORE RECORDS AVAILABLE: {shown} shown out of {total}. {remaining} more available. Ask conversationally if user wants more.\n"

    TOOL_KEY_MAP = {
        "get_customer": "customers",
        "get_sales_invoice": "sales_invoices",
        "get_purchase_invoice": "purchase_invoices",
        "get_stock_levels": "stock_levels",
        "get_gst_summary": "gst_summary",
        "get_sales_summary": "sales_summary",
        "get_purchase_summary": "purchase_summary",
        "get_sales_trend": "sales_trend",
        "get_trial_balance": "trial_balance",
        "get_top_customer": "top_customers",
        "get_item_details": "items",
        "search_ledger": "ledger_matches",
    }
    tool_data = final_response_prompt.get("data", {})
    if isinstance(tool_data, dict):
        renamed = {}
        for tool_name, records in tool_data.items():
            domain_key = TOOL_KEY_MAP.get(tool_name, tool_name.replace("get_", "", 1))
            renamed[domain_key] = records
        final_response_prompt["data"] = renamed

    detail_note = "\nDETAIL MODE: Show EVERY field of each record.\n" if detail_mode else ""

    human_prompt = (
        f"USER QUERY:\n{original_query}\n\n"
        f"CRITICAL — Only use field names and values that are present in TOOL RESULTS below. "
        f"Do NOT invent any field name, ID, or value.\n"
        f"TOOL RESULTS (JSON):\n{json.dumps(final_response_prompt, indent=2, ensure_ascii=False)}\n\n"
    )
    if summary_text:
        human_prompt += f"For context (do not repeat this verbatim): {summary_text}\n\n"
    human_prompt += truncation_note + detail_note

    try:
        full_content = ""
        async for chunk in summary_llm.with_config({"tags": ["response_stream"]}).astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]):
            if hasattr(chunk, "content") and chunk.content:
                full_content += chunk.content
        response_text = full_content.strip()
        if not response_text:
            raise ValueError("Empty response from LLM")
        response_text = _clean_llm_response(response_text)
        response_text = _strip_hallucinated_values(
            response_text,
            final_response_prompt.get("data", {}),
            original_query=original_query,
        )
        if not response_text.strip():
            raise ValueError("Response had only meta-framing/JSON after cleaning")
        data = final_response.get("data", {})
        has_any_data = any(
            isinstance(recs, list) and len(recs) > 0
            for recs in data.values()
        )
        if not has_any_data:
            empty_indicators = ["data nahi mila", "kuch nahi mila", "no data found",
                                "no records found", "nahi mila", "kuch nahi"]
            if not any(ind in response_text.lower() for ind in empty_indicators):
                print(f"[HALLUCINATION GUARD] Empty data but response didn't acknowledge: '{response_text[:150]}...'")
                response_text = "data nahi mila"
    except Exception as e:
        print(f"Error in response generation node: {e}")
        summary_text = final_response.get("summary", "") if isinstance(final_response, dict) else ""
        if summary_text and len(summary_text) > 20:
            response_text = summary_text
        else:
            data = final_response.get("data", {}) if isinstance(final_response, dict) else {}
            parts = []
            for tool_name, records in data.items() if isinstance(data, dict) else []:
                if records and isinstance(records, list):
                    count = len(records)
                    parts.append(f"{tool_name}: {count} record(s)")
            response_text = (
                "; ".join(parts)
                if parts
                else (str(final_response)[:500] if isinstance(final_response, dict) else str(final_response)[:500])
            )
    return {"response_text": response_text}
