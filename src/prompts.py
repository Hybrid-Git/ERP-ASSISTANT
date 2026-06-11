import re

TRANSLATOR_PROMPT_BASE = """Normalize Hinglish/Hindi/Gujarati → clean English JSON.

SCHEMA: {"canonical_query":"...","document_type":"sales_invoice|purchase_invoice|customer|product|general","language":"...","confidence":"high|medium|low","query_type":"erp_query|conversational|ood|mixed","query_parts":["..."],"resolved_entities":[{"original":"...","resolved":"...","type":"..."}]}

WORD MAP: bill=sales_invoice, bikri=sales, kharidi=purchase, grahak=customer, rakam=amount, baki=outstanding, kam=less, zyada=greater, dikhao/batao=show, aur=and, kitne/kitna=how_many/much, hai/ho=is_are, kya=what, konse/konsa/jiska=which, kyu=why, chaia/chahiye=need, nahi=not, hamare/mera/uska/uski=our/my/his, wala/wale=with, sari/saari=all

RULES:
- query_type: "ood" if asking about non-ERP topics (movies, sports, recipes, general knowledge, news, weather, etc.), "conversational" if asking about conversation history (what we discussed, what was asked, recap, etc.), "erp_query" if asking about ERP data (customers/stock/GST/invoices), "mixed" if asking about both history AND data.
- Preserve IDs/HSN/dates/names. Clean English → language="english", query unchanged. Bare number/name → treat as lookup.

EXAMPLES:
Q: A/0326/C0077 sales bill ka customer name batao
A: {"canonical_query":"Show customer name for sales invoice A/0326/C0077","document_type":"sales_invoice","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: kitne products hai inventory mai
A: {"canonical_query":"How many products in inventory","document_type":"product","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: kyu nahi mila
A: {"canonical_query":"Why no results found","document_type":"general","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: hamne sabse pehle kya pucha tha
A: {"canonical_query":"What was asked first by us","document_type":"general","language":"hinglish","confidence":"high","query_type":"conversational"}
Q: muje avengers ke bare mai janna hai
A: {"canonical_query":"Tell me about Avengers","document_type":"general","language":"hinglish","confidence":"high","query_type":"ood"}
/no_think"""

META_QUESTION_PATTERNS_GLOBAL = [
    r"what (have|did|was|were|is|are) we (discussed?|talked?|said?|done|covered|asked)",
    r"what (have|did|was|were|is|are) (i|you|we|the) (discussed?|talked?|said?|done|covered|asked).*\b(first|previous|last|pichl|pehle)",
    r"which (products|items|customers) (have|were) (discussed|talked|mentioned)",
    r"what (was|were) (discussed|talked|mentioned|said)",
    r"(summarize|summary|recap) (the |our |this )?(conversation|chat|discussion)",
    r"conversation (history|so far|till now)",
    r"kya (baat|discuss|hua|kaha)",
    r"humne kya (baat|discuss|kiya|kaha|kari)",
    r"aur\s+usse?\s+pehle",
    r"(es?|is|us)\s+se?\s+pehle",
    r"(kiska|kiski|kiske)\s+(id|name|number|details|baat)\s+manga",
    r"(baat|bat)\s+(hua|hui|kiya|kia|kari|karke?\b)",
    r"(pichl[ei])\s+(baat|baar|query|sawal|question)",
    r"\b(shayad|thana|thahi)\b",
    r"(maine|hamne|humne)\s+(sabse\s+)?(pehle|pahle)\s+kya\s+(pucha|kaha|manga|poocha)",
    r"what\s+(was|were|did|have)\s+.*?\b(first|pehle|pahle|previous)\b",
    r"(usse?|is|es)\s+bhi\s+pehle",
    r"(first|pehle|pahle)\s+(query|question|sawal|baat)",
    r"(sabse\s+)?(pehle|pahle)\s+(kya\s+)?(pucha|kaha|manga|question|query)",
]

GREETING_PATTERNS = [
    r"^(hello|hi|hey|hii|hiii|heyy|holla|namaste|namaskar|vanakkam|howdy|greetings|salam)\s*[!?.]*$",
    r"^(good\s*morning|good\s*afternoon|good\s*evening|good\s*night|gm|gn)\s*[!?.]*$",
    r"^(hey\s+there|hi\s+there|hello\s+there)\s*[!?.]*$",
    r"^(how\s+are\s+(you|u)|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|wassup|sup)\s*[!?.]*$",
    r"^(kaise\s+ho|kya\s+haal|kya\s+kar\s+rahe|kya\s+kar\s+raha|kya\s+kar\s+rahi)\s*[!?.]*$",
    r"^(aap|ap|tu|tum|tumlog)\s+kaise\s+ho\s*[!?.]*$",
    r"^(aap|ap)\s+kese\s+ho\s*[!?.]*$",
    r"^(hello|hi|hey|hii|hiii|heyy|holla)\s+how\s+(are|r)\s+(you|u)\s*[!?.]*$",
    r"^(hello|hi|hey|hii|hiii|heyy|holla)\s+(how's|how is)\s+(it|everyone|you|things|going)\s*[!?.]*$",
]

CAPABILITY_PATTERNS = [
    r"what (can|do) (you|u) do",
    r"what('s| is) your purpose",
    r"what ('s|is) (this |the )?(chatbot|assistant|bot|tool) (for|about)",
    r"(tell|show) me (about|what) (you|u) (can |)do",
    r"what are your capabilities",
    r"how (can|do) (you|u) (help|assist)",
    r"what kind of (questions|queries) (can|do) (you|u) (answer|handle)",
    r"what is the use of (this |the )?(chatbot|assistant|bot|tool)",
    r"(what|which) (all |)(things|work|tasks) (can|do) (you|u) (do|help|handle)",
    r"(kaam|use|upayog) kya hai",
    r"kya kar sakte ho",
    r"kya (kaam|sahayta) kar sakte ho",
    r"aap kya kar sakte hain",
    r"ye (kya|kaisa) (hai|tool|chatbot)",
    r"aap (kya|kaise) (help|madad|sahayta) kar (sakte|sakta)",
]

OOD_PATTERNS = [
    r"^(who|what|why|when|where|how)\s+(is|are|was|were|does|do|did|can|could|will|would|shall|should)\s+",
    r"(tell me about|explain|describe|define)\s",
]

HINGLISH_PRONOUNS = ["uska", "iska", "unka", "iski", "inki", "uski", "woh", "uss", "in sab", "dono", "in dono", "ye dono", "in dono ko"]
DONO_PRONOUNS = {"dono", "in dono", "ye dono", "in dono ko"}

_REFERENCE_PATTERN = re.compile(
    r"\b(uska|uski|uske|iska|iski|iske|unka|unki|unke|inka|inki|inke"
    r"|this|that|these|those|its|it\b|they|them|their|he\b|she|his|her"
    r"|previous|last|first|same|also|too|again|another|previous"
    r"|pehle|pichle|pichli|pahle|baad\b|bad\b|aage"
    r"|aur|bhi\b|or\b|waise|aise|vaise"
    r"|kitne|kitna|konse|konsa|kaun\b|kaunse|kis\b|kisi"
    r"|dono|donu|wahi|wahee|yahi|yee|wohi|wohee)\b",
    re.IGNORECASE,
)

INVOICE_DOC_MAP = {
    "purchase": "purchase_invoice",
    "sales": "sales_invoice",
}

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "has", "have", "had",
    "and", "or", "but", "if", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "on", "off", "out", "over", "under",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "myself", "you", "your", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom",
    "ka", "ke", "ki", "ko", "se", "mai", "mein", "hai", "ho",
    "hu", "hain", "tha", "the", "thi", "thay", "hoga",
}

GST_CATEGORY_KEYWORDS = {
    "b2b": ["b2b"],
    "b2cSmall": ["b2c small", "b2csmall"],
    "b2cLarge": ["b2c large", "b2clarge"],
    "nilRated": ["nil rated", "nilrated", "nill rated", "nillrated"],
    "exempt": ["exempt"],
    "exports": ["export", "exports"],
    "creditNotesRegistered": ["creditnotesregistered", "credit note registered", "creditnoteregistered"],
    "creditNotesUnregistered": ["creditnotesunregistered", "credit note unregistered", "creditnoteunregistered"],
    "grandTotal": ["grand total", "total gst", "gst total", "grandtotal"],
}
