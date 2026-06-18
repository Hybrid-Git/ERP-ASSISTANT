import re
# # Q: kitne products hai inventory mai
# # A: {"canonical_query":"How many products in inventory","document_type":"product","language":"hinglish","confidence":"high","query_type":"erp_query"}
# # Q: kyu nahi mila
# # A: {"canonical_query":"Why no results found","document_type":"general","language":"hinglish","confidence":"high","query_type":"erp_query"}
# # Q: hamne sabse pehle kya pucha tha
# # A: {"canonical_query":"What was asked first by us","document_type":"general","language":"hinglish","confidence":"high","query_type":"conversational"}
TRANSLATOR_PROMPT_BASE = """Normalize Indian language queries (Hinglish, Gujarati, Hindi, Marathi, Punjabi, Bengali) to clean English JSON.

SCHEMA: {"canonical_query":"...","document_type":"sales_invoice|purchase_invoice|customer|product|vendor|general","language":"...","confidence":"high|medium|low","query_type":"greeting|capability|erp_query|conversational|ood|ambiguous","query_intent":"count|aggregate|list_all|comparison|detail|sample|extreme","query_parts":["..."],"resolved_entities":[{"original":"...","resolved":"...","type":"..."}]}

RULES:
- If Indian language → translate canonical_query to clean English, set language to the actual detected language (gujarati, hinglish, hindi, etc.)
- Clean English → language="english", canonical_query unchanged. Do NOT classify Indian language queries as English.
- Preserve IDs/HSN/dates/names/email/phone/invoice numbers.
- If query has pronouns and context is given (CONVERSATION CONTEXT section), resolve them in query_parts.
- query_type: "greeting" for hello/hi/namaste, "capability" ONLY when the user asks about THIS assistant (who are you/what can you do/kya kar sakte ho), "erp_query" for business data, "conversational" for chat/meta/history/about self, "ood" for ANY non-ERP topic (celebrities, movies, news, weather, sports, history, politics, health, recipes, etc.), "ambiguous" for unclear/vague queries.
- query_intent: "count" for how many/kitna/ketla/kiti/count, "aggregate" for total/kul/sum/overall, "list_all" for sab/all/every/complete list/sare/list/dikhao/list dikhao/name list, "comparison" for difference/vs/antar, "detail" for details/vistrit/full info/specs, "extreme" for top/bottom/least/most/sabse kam/sabse jyada/sabse, "sample" for default/general lists.
- document_type: sales_invoice/purchase_invoice/customer/product/vendor/general.

EXAMPLES:
Q: hi
A: {"canonical_query":"Hi","document_type":"general","language":"english","confidence":"high","query_type":"greeting","query_intent":"sample"}
Q: tu kon che
A: {"canonical_query":"Who are you?","document_type":"general","language":"gujarati","confidence":"high","query_type":"capability","query_intent":"sample"}
Q: who is avengers
A: {"canonical_query":"Who is Avengers","document_type":"general","language":"english","confidence":"high","query_type":"ood","query_intent":"sample"}
Q: kitne top products hai
A: {"canonical_query":"How many top products are there","document_type":"product","language":"hinglish","confidence":"high","query_type":"erp_query","query_intent":"count"}
Q: sab customer dikhao
A: {"canonical_query":"Show all customers","document_type":"customer","language":"hinglish","confidence":"high","query_type":"erp_query","query_intent":"list_all"}/no_think"""

META_QUESTION_PATTERNS_GLOBAL = []

GREETING_PATTERNS = []

CAPABILITY_PATTERNS = []

OOD_TOPICS = {}

# --- COMMENTED OUT (zero-regex migration): pronoun word lists ---
# HINGLISH_PRONOUNS = ["uska", "iska", "unka", "iski", "inki", "uski", "woh", "uss", "in sab", "dono", "in dono", "ye dono", "in dono ko"]
# DONO_PRONOUNS = {"dono", "in dono", "ye dono", "in dono ko"}

# --- COMMENTED OUT (zero-regex migration): reference pattern for follow-up detection ---
# _REFERENCE_PATTERN = re.compile(
#     r"\b(uska|uski|uske|iska|iski|iske|unka|unki|unke|inka|inki|inke"
#     r"|this|that|these|those|its|it\b|they|them|their|he\b|she|his|her"
#     r"|previous|last|first|same|also|too|again|another|previous"
#     r"|pehle|pichle|pichli|pahle|baad\b|bad\b|aage"
#     r"|aur|bhi\b|or\b|waise|aise|vaise"
#     r"|kitne|kitna|konse|konsa|kaun\b|kaunse|kis\b|kisi"
#     r"|dono|donu|wahi|wahee|yahi|yee|wohi|wohee)\b",
#     re.IGNORECASE,
# )

INVOICE_DOC_MAP = {
    "purchase": "purchase_invoice",
    "sales": "sales_invoice",
}

# --- COMMENTED OUT (zero-regex migration): stop words for embedding scoring ---
# _STOP_WORDS = {
#     "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
#     "do", "does", "did", "doing", "has", "have", "had",
#     "and", "or", "but", "if", "because", "as", "until", "while", "of",
#     "at", "by", "for", "with", "about", "between", "into", "through",
#     "during", "before", "after", "above", "below", "to", "from",
#     "up", "down", "in", "on", "off", "out", "over", "under",
#     "again", "further", "then", "once", "here", "there",
#     "when", "where", "why", "how", "all", "each", "every", "both",
#     "few", "more", "most", "other", "some", "such", "no", "nor",
#     "not", "only", "own", "same", "so", "than", "too", "very",
#     "it", "its", "this", "that", "these", "those",
#     "i", "me", "my", "myself", "you", "your", "yourself",
#     "he", "him", "his", "himself", "she", "her", "hers", "herself",
#     "we", "us", "our", "ours", "ourselves", "they", "them", "their",
#     "theirs", "themselves", "what", "which", "who", "whom",
#     "ka", "ke", "ki", "ko", "se", "mai", "mein", "hai", "ho",
#     "hu", "hain", "tha", "the", "thi", "thay", "hoga",
# }

VAGUE_ACTION_WORDS = set()

# --- COMMENTED OUT (zero-regex migration): GST category keywords ---
# GST_CATEGORY_KEYWORDS = {
#     "b2b": ["b2b"],
#     "b2cSmall": ["b2c small", "b2csmall"],
#     "b2cLarge": ["b2c large", "b2clarge"],
#     "nilRated": ["nil rated", "nilrated", "nill rated", "nillrated"],
#     "exempt": ["exempt"],
#     "exports": ["export", "exports"],
#     "creditNotesRegistered": ["creditnotesregistered", "credit note registered", "creditnoteregistered"],
#     "creditNotesUnregistered": ["creditnotesunregistered", "credit note unregistered", "creditnoteunregistered"],
#     "grandTotal": ["grand total", "total gst", "gst total", "grandtotal"],
# }
