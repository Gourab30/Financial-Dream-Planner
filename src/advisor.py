
import os
import requests
from dotenv import load_dotenv
 
load_dotenv(override=True)
 
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY", "")
 
FALLBACK_NOTE = (
    "Groq API unavailable - standard advice: automate your investments, "
    "build a 3-6 month emergency fund first, and revisit this plan yearly."
)
 
 
def get_advisor_note(plan: dict) -> str:
    """Ask Groq for a few short tips based on numbers we already worked out."""
    if not API_KEY:
        return FALLBACK_NOTE + "\n\n[debug: GROQ_API_KEY is empty - .env not loaded, or not found in this folder]"
 
    prompt = (
        "You are a financial-planning assistant. The numbers below were already "
        "calculated by the app - do not change or invent any numbers, just give "
        "3 short, practical recommendations in plain language.\n\n"
        f"Predicted Monthly Salary: {plan.get('predicted_monthly_salary')}\n"
        f"Available Monthly Capacity: {plan.get('available_monthly_capacity')}\n"
        f"Total Required Monthly SIP: {plan.get('total_required_monthly_sip')}\n"
        f"Overall Status: {plan.get('overall_status')}\n"
    )
 
    try:
        res = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        debug = f"Groq returned {res.status_code}: {res.text[:300]}"
        print(f"[advisor] {debug}")
        return f"{FALLBACK_NOTE}\n\n[debug: {debug}]"
    except Exception as e:
        debug = f"Groq request failed: {e}"
        print(f"[advisor] {debug}")
        return f"{FALLBACK_NOTE}\n\n[debug: {debug}]"
 
