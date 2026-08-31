"""
Cloud Agent for the AI Financial Dream Planner (Groq API edition).

Responsibilities (and NON-responsibilities) of this Agent, per assignment rule
"The Agent must not perform deterministic financial calculations itself":
  1. Understand a natural-language request.
  2. Decide which tool(s) from agent_tools.TOOL_REGISTRY are needed.
  3. Call those tools (all deterministic maths lives in financial_calculator.py).
  4. Optionally use Groq's hosted LLM ONLY to phrase a friendly explanation of
     numbers the tools already computed - the LLM is never allowed to invent or
     alter a number.

Uses Groq's OpenAI-compatible Chat Completions API
(https://api.groq.com/openai/v1/chat/completions). If GROQ_API_KEY is not
set, or the Groq API is unreachable, the Agent still works correctly using
rule-based intent detection + regex entity extraction; it just falls back to
a templated (non-LLM) explanation instead of an LLM-generated one.

NOTE ON MODEL NAMES: Groq periodically deprecates and shuts down older
model IDs (see https://console.groq.com/docs/deprecations). A request to a
decommissioned model ID returns an HTTP error, which looks identical to any
other Groq failure from the outside. If the AI advisor note or RAG
narration suddenly stops working with no code changes on your end, check
that page first before assuming the API key is the problem.
"""
import os
import re
import sys

import requests
from dotenv import load_dotenv

from src.planner import FinancialPlanner
from src.goal_cost_lookup import GoalCostLookup
from src.rag.rag_engine import RAGEngine

load_dotenv()

KNOWLEDGE_INTENT_KEYWORDS = ["why", "what is", "what's", "explain", "how does", "how is", "how do",
                             "define", "meaning of", "tell me about"]

# Very small, explicit prompt-injection guard: strip / refuse to follow any
# instruction-like phrases embedded in user input. The Agent's control flow
# below never lets user text change *which* tools are called or *how*
# calculations are performed, so even if this list misses a phrase, the
# worst case is a mis-classified intent - never a bypassed calculation.
INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the) instructions",
    r"disregard (all|any|previous|the) instructions",
    r"reveal (the )?system prompt",
    r"you are now",
    r"act as (an? )?(?!fresher)",
    r"forget (all|everything|your rules)",
]

EDUCATION_KEYWORDS = ["b.tech", "btech", "b.e.", "be", "mba", "mca", "bca", "b.sc", "bsc", "m.sc", "msc", "m.tech", "mtech"]
JOB_ROLE_KEYWORDS = ["software engineer", "web developer", "ui ux designer", "ux designer",
                      "technical support engineer", "project coordinator", "qa engineer",
                      "business analyst", "devops engineer"]

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

# llama-3.3-70b-versatile was shut down by Groq on 08/16/26 (see the module
# docstring note above); openai/gpt-oss-120b is Groq's own recommended
# replacement for it. Update this - and the GROQ_MODEL value in .env - if
# Groq deprecates this one too.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqAgent:
    def __init__(self, model_name: str = None, api_key: str = None):
        # GROQ_MODEL / GROQ_API_KEY are read from the environment (.env file),
        # never hardcoded, so the key is not committed to source control.
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.planner = FinancialPlanner()
        self.city_lookup = GoalCostLookup()
        self.rag = RAGEngine()
        # Set after every _call_groq attempt: None on success, or a short
        # human-readable reason on failure (bad key, decommissioned model,
        # rate limit, timeout, ...). Read this after a call returns "" to
        # see WHY, instead of only seeing the generic fallback text.
        self.last_error = None

    # ---------- Prompt-injection guard ----------
    def _sanitize(self, text: str) -> str:
        cleaned = text
        for pattern in INJECTION_PATTERNS:
            cleaned = re.sub(pattern, "[blocked instruction]", cleaned, flags=re.IGNORECASE)
        return cleaned

    # ---------- Intent detection ----------
    def _detect_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(kw in lowered for kw in KNOWLEDGE_INTENT_KEYWORDS):
            return "knowledge"
        if re.search(r"\d+\s*(years?|yrs?)", lowered) or "plan" in lowered or "save" in lowered:
            return "plan"
        return "unclear"

    # ---------- Entity extraction (regex, deterministic - no LLM guessing numbers) ----------
    def _extract_fields(self, text: str) -> dict:
        lowered = text.lower()
        fields = {}

        m = re.search(r"\bi[' ]?am\s+(\d{1,2})\b|\bage\s+(\d{1,2})\b|\b(\d{1,2})\s*(?:years?\s*old)\b", lowered)
        if m:
            fields["age"] = int(next(g for g in m.groups() if g))

        for city in self.city_lookup.get_cities():
            if city.lower() in lowered:
                fields["city"] = city
                break

        for edu in EDUCATION_KEYWORDS:
            if edu in lowered:
                fields["education"] = edu.upper().replace("BTECH", "B.Tech").replace("BE", "B.E.")
                break

        for role in JOB_ROLE_KEYWORDS:
            if role in lowered:
                fields["job_role"] = role.title()
                break

        m = re.search(r"marr(?:y|iage)[^\d]{0,15}(\d{1,2})\s*(?:years?|yrs?)", lowered)
        if m:
            fields["marriage_years"] = int(m.group(1))

        m = re.search(r"car[^\d]{0,15}(\d{1,2})\s*(?:years?|yrs?)", lowered)
        if m:
            fields["car_years"] = int(m.group(1))

        m = re.search(r"(?:home|house)[^\d]{0,15}(\d{1,2})\s*(?:years?|yrs?)", lowered)
        if m:
            fields["home_years"] = int(m.group(1))

        m = re.search(r"(\d{1,3})\s*%\s*(?:of my income|savings?|invest)?", lowered)
        if m:
            fields["savings_percent"] = float(m.group(1))

        return fields

    # ---------- Groq call (optional narration only, never computes numbers) ----------
    def _call_groq(self, prompt: str) -> str:
        self.last_error = None
        if not self.api_key:
            self.last_error = "No GROQ_API_KEY set in the environment/.env file."
            return ""
        try:
            res = requests.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 400,
                },
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

            # Non-200: pull out whatever explanation Groq gave us instead of
            # discarding it. Groq's error body follows the same
            # {"error": {"message": ...}} shape as OpenAI's API.
            try:
                detail = res.json().get("error", {}).get("message", res.text)
            except Exception:
                detail = res.text
            self.last_error = f"HTTP {res.status_code} from Groq: {str(detail)[:300]}"

        except requests.exceptions.Timeout:
            self.last_error = "Groq request timed out (>15s)."
        except requests.exceptions.ConnectionError as e:
            self.last_error = f"Could not reach Groq (network/connection error): {e}"
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"

        # Never let a Groq failure crash the app - log it loudly to stderr so
        # it's visible in the terminal running `python main.py` / uvicorn,
        # then fall back to rule-based behaviour as documented above.
        print(f"[GroqAgent] Groq call failed, falling back to templated response - {self.last_error}",
              file=sys.stderr)
        return ""

    def enrich_plan(self, plan_data: dict) -> str:
        """Optional narrative commentary on an ALREADY-COMPUTED plan. The LLM is
        instructed to only comment, never to output new numbers."""
        prompt = (
            "You are a financial-planning assistant. The numbers below were already "
            "calculated deterministically by the application - do NOT change, recompute, "
            "or invent any numbers. Just give 3 short, concise, practical recommendations "
            "in plain language.\n\n"
            f"Predicted Monthly Salary: {plan_data.get('predicted_monthly_salary')}\n"
            f"Available Monthly Capacity: {plan_data.get('available_monthly_capacity')}\n"
            f"Total Required Monthly SIP: {plan_data.get('total_required_monthly_sip')}\n"
            f"Overall Status: {plan_data.get('overall_status')}\n"
        )
        response = self._call_groq(prompt)
        if response:
            return response
        reason = f" ({self.last_error})" if self.last_error else ""
        return ("Groq API unavailable" + reason + " - standard advice: automate your "
                "investments, build a 3-6 month emergency fund first, and revisit this "
                "plan yearly.")

    # ---------- Main NL entry point ----------
    def handle_message(self, message: str) -> dict:
        safe_message = self._sanitize(message)
        intent = self._detect_intent(safe_message)

        if intent == "knowledge":
            rag_result = self.rag.answer(safe_message)
            if rag_result["grounded"]:
                narrated = self._call_groq(
                    "Answer the user's question using ONLY the context below. If the context "
                    "does not contain the answer, say the information is unavailable. Do not "
                    f"invent facts.\n\nContext: {rag_result['answer']}\n\nQuestion: {safe_message}"
                )
                return {
                    "intent": "knowledge",
                    "answer": narrated or rag_result["answer"],
                    "grounded": True,
                    "sources": rag_result["sources"],
                }
            return {"intent": "knowledge", "answer": rag_result["answer"], "grounded": False, "sources": []}

        if intent == "plan":
            fields = self._extract_fields(safe_message)
            required = ["age", "city", "savings_percent"]
            missing = [f for f in required if f not in fields]
            if missing or not any(k in fields for k in ["marriage_years", "car_years", "home_years"]):
                return {
                    "intent": "plan",
                    "error": "Could not extract enough information from the message.",
                    "extracted": fields,
                    "missing_or_needed": missing + (["at least one goal timeline"]
                                                      if not any(k in fields for k in
                                                                 ["marriage_years", "car_years", "home_years"])
                                                      else []),
                }
            fields.setdefault("education", "B.Tech")
            fields.setdefault("job_role", "Software Engineer")
            try:
                plan = self.planner.generate_plan(
                    age=fields["age"], city=fields["city"], education=fields["education"],
                    job_role=fields["job_role"], savings_percent=fields["savings_percent"],
                    marriage_years=fields.get("marriage_years"), car_years=fields.get("car_years"),
                    home_years=fields.get("home_years"),
                )
            except ValueError as e:
                return {"intent": "plan", "error": str(e), "extracted": fields}
            plan["ai_advisor_feedback"] = self.enrich_plan(plan)
            return {"intent": "plan", "plan": plan, "extracted": fields}

        return {
            "intent": "unclear",
            "answer": ("I couldn't tell if you want a financial plan or an explanation. Try something "
                       "like: 'I am 22, live in Bangalore, want to marry in 5 years, buy a car in 4 years "
                       "and a home in 10 years, and save 20% of my income.'"),
        }
