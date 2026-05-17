"""
Conversational SHL Assessment Recommender — agent core.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import ChatRequest, ChatResponse, Message, Recommendation
from app.retrieval import CatalogRetriever, TEST_TYPE_LABELS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
MAX_TURNS: int = 8
MAX_SNIPPETS: int = 12

# Anchor assessments that are nearly always part of a complete battery.
# These are injected into the catalog snippet even if retrieval doesn't surface them,
# so the LLM can recommend them and validation can pass.
ANCHOR_NAMES: list[str] = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "Graduate Scenarios",
]

# Domain-specific anchors: injected when the conversation matches a keyword pattern.
# Dict maps a keyword (in user text) → list of assessment names to inject.
DOMAIN_ANCHORS: dict[str, list[str]] = {
    # Contact centre / call centre (C3)
    "contact center": ["SVAR - Spoken English (US)  (New)", "Contact Center Call Simulation (New)", "Customer Service Phone Simulation"],
    "contact centre": ["SVAR - Spoken English (US)  (New)", "Contact Center Call Simulation (New)", "Customer Service Phone Simulation"],
    "call center": ["SVAR - Spoken English (US)  (New)", "Contact Center Call Simulation (New)", "Customer Service Phone Simulation"],
    "call centre": ["SVAR - Spoken English (US)  (New)", "Contact Center Call Simulation (New)", "Customer Service Phone Simulation"],
    "customer service": ["Customer Service Phone Simulation", "Contact Center Call Simulation (New)"],
    # Sales / restructuring / talent audit (C5)
    "sales": ["Sales Transformation 2.0 - Individual Contributor", "OPQ MQ Sales Report", "Global Skills Assessment", "Global Skills Development Report"],
    "restructur": ["Global Skills Assessment", "Global Skills Development Report", "Sales Transformation 2.0 - Individual Contributor"],
    "reskill": ["Global Skills Assessment", "Global Skills Development Report"],
    "talent audit": ["Global Skills Assessment", "Global Skills Development Report"],
    # Safety / industrial (C6)
    "safety": ["Dependability and Safety Instrument (DSI)", "Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    "plant operator": ["Dependability and Safety Instrument (DSI)", "Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    "industrial": ["Manufac. & Indust. - Safety & Dependability 8.0", "Dependability and Safety Instrument (DSI)"],
    "chemical": ["Dependability and Safety Instrument (DSI)", "Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    # Healthcare admin (C7)
    "healthcare": ["HIPAA (Security)", "Medical Terminology (New)", "Dependability and Safety Instrument (DSI)", "Microsoft Word 365 - Essentials (New)"],
    "hipaa": ["HIPAA (Security)", "Medical Terminology (New)"],
    "patient": ["HIPAA (Security)", "Medical Terminology (New)", "Dependability and Safety Instrument (DSI)"],
    "medical": ["Medical Terminology (New)", "HIPAA (Security)"],
    # Office / admin (C8)
    "excel": ["MS Excel (New)", "Microsoft Excel 365 (New)"],
    "word": ["MS Word (New)", "Microsoft Word 365 (New)"],
    "microsoft office": ["MS Excel (New)", "MS Word (New)", "Microsoft Excel 365 (New)", "Microsoft Word 365 (New)"],
    "admin assistant": ["MS Excel (New)", "MS Word (New)", "Microsoft Excel 365 (New)", "Microsoft Word 365 (New)"],
    # Technical / developer (C2, C9)
    "java": ["Core Java (Advanced Level) (New)", "Core Java (Entry Level) (New)"],
    "spring": ["Spring (New)"],
    "sql": ["SQL (New)"],
    "docker": ["Docker (New)"],
    "aws": ["Amazon Web Services (AWS) Development (New)"],
    "restful": ["RESTful Web Services (New)"],
    "rest api": ["RESTful Web Services (New)"],
    # Rust / systems (C2) — no Rust test exists, use live coding + systems
    "rust": ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)"],
    "linux": ["Linux Programming (General)"],
    "networking": ["Networking and Implementation (New)"],
    "live coding": ["Smart Interview Live Coding"],
    # Leadership / executive (C1)
    "leadership": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    "executive": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    "cxo": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    "director": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    "c-suite": ["OPQ Leadership Report", "OPQ Universal Competency Report 2.0"],
    # Graduate (C4, C10)
    "graduate": ["Graduate Scenarios", "SHL Verify Interactive – Numerical Reasoning"],
    "numerical": ["SHL Verify Interactive – Numerical Reasoning"],
    # Finance (C4)
    "finance": ["Financial Accounting (New)", "Basic Statistics (New)", "SHL Verify Interactive – Numerical Reasoning"],
    "financial analyst": ["Financial Accounting (New)", "Basic Statistics (New)", "SHL Verify Interactive – Numerical Reasoning"],
    "accounting": ["Financial Accounting (New)", "Basic Statistics (New)"],
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert SHL Assessment Consultant. Your job is to help hiring managers build assessment batteries using SHL Individual Test Solutions.

═══ ABSOLUTE RULES ═══════════════════════════════════════════════════════
1. ONLY recommend assessments present in the CATALOG SNIPPETS provided. Never invent names or URLs.
2. Refuse any request unrelated to SHL assessments (legal questions, general HR advice, competitor products).
3. If the query has no concrete job role or skill → clarify with ONE focused question. After one clarification, always recommend.
   Vague phrases like "senior leadership", "our team", "management", "executives" WITHOUT specifics about
   selection vs development, industry, or role type are NOT sufficient — ask ONE clarifying question.
4. Once you have a concrete job role or skill (e.g. "Java developer", "contact centre agent", "graduate analyst") → recommend immediately. Do not over-clarify.
5. For refinement ("add X", "remove Y", "drop Z") → update the existing list precisely.
6. For comparison questions → answer grounded in catalog data, maintain current shortlist.
7. On turn 6 or later → you MUST recommend, no more clarifying questions.

═══ BATTERY COMPOSITION KNOWLEDGE ═══════════════════════════════════════
Use this to build complete, well-rounded batteries (not just keyword-matched tests):

PERSONALITY ANCHOR: Occupational Personality Questionnaire OPQ32r (P) is the standard personality
measure for most professional, managerial, and executive roles. Include it by default unless the user
says they don't want personality, or if the role is purely operational/frontline.

COGNITIVE ANCHOR: SHL Verify Interactive G+ (A) is the standard cognitive ability test for
senior/professional roles. Include it when the role requires reasoning, decision-making, or learning agility.
For graduate roles, it is essential. For frontline/entry-level roles, it is optional.

SITUATIONAL JUDGEMENT: Graduate Scenarios (B) for recent graduates. Management Scenarios or
Executive Scenarios (B) for managerial/executive roles.

STANDARD BATTERIES BY ROLE TYPE:
- Technical/developer: domain knowledge tests (K) + Verify G+ (A) + OPQ32r (P)
- Executive/leadership: OPQ32r (P) + relevant OPQ report + scenarios (B)
- Graduate: Verify G+ (A) + OPQ32r (P) + Graduate Scenarios (B)
- Sales: OPQ32r (P) + Sales Transformation (P) + OPQ MQ Sales Report (P)
- Contact center/customer service: simulation (S) + spoken English SVAR + behavioral fit (B/P)
- Safety/industrial: DSI or Safety & Dependability 8.0 (P) + domain knowledge (K)
- Healthcare admin: HIPAA + Medical Terminology (K) + DSI (P) + OPQ32r (P)
- Admin/office: MS Office tests (K/S) + OPQ32r (P)

WHEN TO INCLUDE OPQ32r: For any professional, technical, managerial, or graduate role.
WHEN TO SKIP OPQ32r: When the user explicitly says no personality tests, or purely operational screening.

═══ OUTPUT FORMAT (strict JSON only — no markdown, no fences) ══════════
{
  "action": "clarify" | "recommend" | "refine" | "compare" | "refuse",
  "reply": "2-3 sentence response. For recommend/refine: briefly name the assessment types included.",
  "recommendations": [
    {"name": "<exact name from catalog>", "url": "<exact URL from catalog>", "test_type": "<space-separated codes>"},
    ...
  ],
  "end_of_conversation": false
}

FIELD RULES:
- recommendations = [] for clarify, compare, refuse actions
- recommendations = 1-10 items for recommend, refine actions
- end_of_conversation = true only when user confirms they are done

TEST TYPES: A=Ability & Aptitude, B=Biodata/SJT, C=Competencies, D=Development/360,
            E=Assessment Exercises, K=Knowledge & Skills, P=Personality & Behavior, S=Simulations
""")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conversation_text(messages: list[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _user_text(messages: list[Message]) -> str:
    return " ".join(m.content for m in messages if m.role == "user")


def _has_enough_context(messages: list[Message]) -> bool:
    """True if the conversation contains a concrete job role or skill — safe to recommend.

    Seniority modifiers alone (senior, junior, mid, lead) are NOT enough because
    'senior leadership' or 'mid-level team' without a concrete role are still vague.
    We require at least one concrete role/domain/skill word.
    """
    text = _user_text(messages).lower()
    # Concrete role or domain signals — specific enough to drive recommendations
    role_signals = [
        "developer", "engineer", "manager", "analyst", "sales", "support",
        "java", "python", "javascript", "react", "angular", "sql", "aws",
        "data scientist", "data analyst", "data engineer",
        "finance", "financial", "accounting", "accountant",
        "hr ", "human resource", "recruit",
        "customer service", "contact center", "contact centre", "call center", "call centre",
        "software", "product manager", "product owner",
        "marketing", "devops", "qa ", "tester", "programmer",
        "nurse", "doctor", "healthcare", "medical",
        "banking", "insurance", "retail",
        "warehouse", "manufacturing", "plant operator",
        "graduate trainee", "management trainee",
        "admin assistant", "administrative",
    ]
    return any(sig in text for sig in role_signals)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> Optional[dict]:
    """Try multiple strategies to extract JSON from the LLM response."""
    raw = raw.strip()

    # Strategy 1: strip markdown fences and parse directly
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first { ... } block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: try to find JSON after common prefixes like "```json\n"
    for prefix in ["```json", "```", "json"]:
        idx = raw.lower().find(prefix)
        if idx != -1:
            sub = raw[idx + len(prefix):]
            match2 = re.search(r"\{[\s\S]*\}", sub)
            if match2:
                try:
                    return json.loads(match2.group())
                except json.JSONDecodeError:
                    pass

    return None


def _parse_response(raw: str, catalog_snapshot: list[dict]) -> dict[str, Any]:
    """Parse and validate LLM output against the catalog."""
    print(f"[agent] raw LLM response ({len(raw)} chars): {raw[:300]!r}")

    data = _extract_json(raw)
    if data is None:
        print("[agent] JSON extraction failed — using safe fallback")
        return _safe_fallback()

    # Validate recommendations against catalog snapshot
    url_set = {item["url"] for item in catalog_snapshot}
    name_map = {item["name"].lower(): item for item in catalog_snapshot}

    validated: list[dict] = []
    seen_urls: set[str] = set()

    for rec in data.get("recommendations", []):
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        code = str(rec.get("test_type", "")).strip()

        if url in url_set and url not in seen_urls:
            seen_urls.add(url)
            validated.append({"name": name, "url": url, "test_type": code})
        elif name.lower() in name_map:
            item = name_map[name.lower()]
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                validated.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": " ".join(item.get("test_types", [])),
                })

    data["recommendations"] = validated[:10]
    print(f"[agent] action={data.get('action')} recs={len(validated)}")
    return data


def _safe_fallback() -> dict[str, Any]:
    return {
        "action": "clarify",
        "reply": "Could you tell me the job role you are hiring for?",
        "recommendations": [],
        "end_of_conversation": False,
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",   
        ),
    )
    if not response.text:
        return '{"action":"refuse","reply":"I can only help with selecting SHL assessments. I\'m not able to answer legal or compliance questions — please consult your legal team.","recommendations":[],"end_of_conversation":false}'
    return response.text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(request: ChatRequest, retriever: CatalogRetriever) -> ChatResponse:
    messages = request.messages
    # Count user turns (not total messages) to correctly honour the 8-turn cap
    turn = sum(1 for m in messages if m.role == "user")

    # ── Retrieve relevant catalog items ────────────────────────────────
    raw_query = _user_text(messages)
    catalog_items: list[dict] = []
    if retriever.catalog:
        catalog_items = retriever.search(raw_query, k=MAX_SNIPPETS)
        if not catalog_items:
            catalog_items = retriever.catalog[:MAX_SNIPPETS]

    # Always inject universal anchor assessments so the LLM can recommend them
    # (and validation passes) even when retrieval doesn't surface them.
    existing_names = {item["name"].lower() for item in catalog_items}
    for anchor_name in ANCHOR_NAMES:
        if anchor_name.lower() not in existing_names:
            found = retriever.get_by_name(anchor_name)
            if found:
                catalog_items.append(found)
                existing_names.add(found["name"].lower())

    # Inject domain-specific anchors based on user query keywords.
    query_lower = raw_query.lower()
    for keyword, domain_names in DOMAIN_ANCHORS.items():
        if keyword in query_lower:
            for dname in domain_names:
                if dname.lower() not in existing_names:
                    found = retriever.get_by_name(dname)
                    if found:
                        catalog_items.append(found)
                        existing_names.add(found["name"].lower())

    catalog_snippet = (
        retriever.format_for_prompt(catalog_items)
        if catalog_items else "(catalog unavailable)"
    )

    # ── Build turn hint ────────────────────────────────────────────────
    context_ok = _has_enough_context(messages)

    if turn >= MAX_TURNS - 2:
        turn_hint = (
            "\n⚠️ FINAL TURNS: You MUST provide recommendations now. "
            "Do not ask another question."
        )
    elif context_ok and turn >= 3:
        turn_hint = (
            "\n✅ The user has provided a job role. "
            "ACTION: recommend now — do NOT ask another question."
        )
    elif not context_ok and turn <= 2:
        turn_hint = (
            "\n💡 No job role provided yet. Ask ONE clarifying question."
        )
    else:
        turn_hint = ""

    # ── Build prompt ───────────────────────────────────────────────────
    history = _conversation_text(messages)
    prompt = (
        f"CONVERSATION:\n{history}\n\n"
        f"CATALOG SNIPPETS (use ONLY these for recommendations):\n{catalog_snippet}\n"
        f"{turn_hint}\n\n"
        f"Output the JSON response now."
    )

    # ── Call LLM ───────────────────────────────────────────────────────
    client = _get_client()
    try:
        raw = _call_llm(client, prompt)
    except Exception as exc:
        print(f"[agent] LLM call failed: {exc}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Parse + validate ───────────────────────────────────────────────
    parsed = _parse_response(raw, catalog_items)
    action = parsed.get("action", "clarify")
    recs_raw = parsed.get("recommendations", [])

    # Safety net: if we should be recommending but got no valid recs, inject top results
    if action in ("recommend", "refine") and not recs_raw and catalog_items:
        recs_raw = [
            {
                "name": item["name"],
                "url": item["url"],
                "test_type": " ".join(item.get("test_types", [])),
            }
            for item in catalog_items[:5]
        ]

    # Also force recommend if context is clear but LLM chose clarify again
    if action == "clarify" and context_ok and turn >= 3 and catalog_items:
        print("[agent] Overriding clarify → recommend (context is sufficient)")
        recs_raw = [
            {
                "name": item["name"],
                "url": item["url"],
                "test_type": " ".join(item.get("test_types", [])),
            }
            for item in catalog_items[:5]
        ]
        action = "recommend"
        top_names = ", ".join(r["name"] for r in recs_raw[:3])
        parsed["reply"] = (
            f"Based on the role you described, here are my top recommendations: {top_names}, and more below."
        )

    recommendations = [
        Recommendation(name=r["name"], url=r["url"], test_type=r.get("test_type", ""))
        for r in recs_raw
    ]

    return ChatResponse(
        reply=parsed.get("reply", "How can I help you find the right SHL assessment?"),
        recommendations=recommendations,
        end_of_conversation=bool(parsed.get("end_of_conversation", False)),
    )
