# AI-Powered Financial Dream & Goal Planner

A financial planning assistant for freshers/early-career professionals:
predicts current salary with a trained regression model, projects the future
cost of Marriage / New Car / New Home goals, calculates the required monthly
investment for each, checks feasibility against the user's savings target,
recommends a broad investment category per goal, and answers explanatory
questions using a local RAG knowledge base — all through a FastAPI backend, a
natural-language Agent, and a plain HTML/JS frontend. LLM narration (the
"AI advisor feedback" text and the phrasing of grounded RAG answers) is
powered by the [Groq API](https://console.groq.com) — all financial math
stays deterministic and is never delegated to the LLM.

## What changed from the original files you uploaded (bug fixes)

| Issue found | Fix |
|---|---|
| `main.py` / `PlanRequest` took `experience_years` and used it in the request — **violates the Fresher Rule** ("Experience must not be a salary-model feature") | New `PlanRequest` schema uses `age, city, education, job_role` only. `Experience` never appears anywhere in the pipeline (asserted in `train_model.py`). |
| No `src/` package existed — `main.py` imported `src.schemas`, `src.planner`, `src.goal_cost_lookup`, `src.groq_agent` etc. but none of those files were present, so the app could not run. | Added the full `src/` package (see Architecture below). |
| Only one flat `years_ahead` field for *all* goals — the brief requires Marriage, Car and Home to be planned **separately** with independent timelines. | `PlanRequest` now has `marriage_years`, `car_years`, `home_years` (each optional/independent). |
| No inflation, no feasibility classification, no shortfall calculation, no investment-category rule existed. | Added to `financial_calculator.py`: `classify_feasibility`, `calculate_shortfall`, `recommend_investment_category`, with a documented `DEFAULT_ANNUAL_INFLATION`. |
| No ML training code, no city/goal cost dataset loader. | Added `src/train_model.py` (trains + compares 3 regressors, saves the best) and `src/goal_cost_lookup.py`. |
| `salary_data.csv` has a trailing empty `Unnamed: 5` column from a trailing comma. | `train_model.load_data()` strips unnamed/empty columns before training. |
| No RAG, no Agent tool routing, no prompt-injection defence, no CORS (so a plain HTML file could never call the API from a browser). | Added `src/rag/`, `src/agent_tools.py`, wrote `src/groq_agent.py` (Groq API, previously Ollama), added `CORSMiddleware` in `main.py`. |
| No automated tests. | Added `tests/test_cases.py` — 18 cases, runnable with or without `pytest`. |

## Architecture

```
                         ┌────────────────────────┐
   Browser (frontend/    │      FastAPI (main.py)  │
   index.html)  ───────► │  /plan  /plan/agent      │
        ▲                │  /agent/message /rag/... │
        │  JSON over      └───────────┬─────────────┘
        │  CORS                       │
        │                 ┌───────────┼─────────────────────┐
        │                 ▼           ▼                     ▼
        │        FinancialPlanner   GroqAgent           RAGEngine
        │        (planner.py)     (groq_agent.py)     (rag/rag_engine.py)
        │                 │           │  intent + regex        │ TF-IDF vector
        │                 │           │  extraction, then       │ store over 3
        │                 │           │  calls the SAME         │ local .txt docs
        │                 │           │  planner + tools         ▼
        │                 ▼           ▼                  grounded answer or
        │      SalaryPredictor   agent_tools.py           "not available"
        │      (ML model,       (5 named tools wrapping
        │       models/*.pkl)    the calculator/lookup/
        │                        predictor - Agent never
        │                        computes numbers itself)
        │                 │
        │                 ▼
        │      FinancialCalculator + GoalCostLookup
        │      (deterministic math; data/city_goal_costs.csv)
        └── (optional) Groq-hosted LLM (openai/gpt-oss-120b) —
            ONLY used to phrase already-computed numbers / ground
            RAG answers in natural language; never allowed to
            invent a number.
```

**Data flow (DFD, textual):**
`User input → Agent/API → [Salary Prediction Tool → predicted salary]
→ [Future Cost Tool → inflated goal cost] → [Investment Calculator Tool →
required SIP] → [Feasibility Tool → shortfall/status] → [Recommendation Tool →
category] → Plan JSON → (optional) LLM narration → Frontend`

## Installation

```bash
cd financial_planner
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(Optional, for LLM narration) get a free [Groq API key](https://console.groq.com/keys)
and set it in a `.env` file in the project root (copy `.env.example` and fill
in your key):

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

The app works fully without this — it just falls back to a templated advisor
message and returns raw RAG text instead of LLM-phrased text. `.env` is
git-ignored, so the key is never committed.

**If the advisor note ever falls back to "Groq API unavailable" again:** the
fallback message now includes the actual reason (bad/missing key, rate
limit, timeout, or a decommissioned model ID), and the same detail is
printed to the server console. Groq periodically retires older model IDs
(check https://console.groq.com/docs/deprecations) — that's exactly what
happened to the `llama-3.3-70b-versatile` default this project originally
shipped with, which Groq shut down on 08/16/26. If the error mentions the
model, swap `GROQ_MODEL` in `.env` for a current one from that page.

## Running it

```bash
# 1. Train the salary model + run a sample plan + agent demo in the terminal
python run_demo.py

# 2. Start the API
python main.py
# or: uvicorn main:app --reload
# API docs: http://127.0.0.1:8000/docs

# 3. Open the frontend
# Just double-click frontend/index.html (or serve it with any static server).
# It calls http://127.0.0.1:8000 by default - change the API_BASE constant
# at the top of frontend/index.html if you run the API elsewhere.
```

## Running tests

```bash
python tests/test_cases.py          # no dependencies beyond requirements.txt
# or
pytest tests/test_cases.py -v
```
See `tests/test_results.md` for the full 18-case expected-vs-actual table.

## Key assumptions (documented, all easy to change in one place)

- **Inflation**: the brief's header literally says "0.06% per year", which as
  a decimal (0.0006) is unrealistically small for multi-year goal costs. We
  interpret the intended value as **6% per year** (`DEFAULT_ANNUAL_INFLATION =
  0.06` in `financial_calculator.py`) — the standard reading of "0.06" as a
  rate fraction. Change that one constant if a literal 0.06% was intended.
- **Expected investment return**: user-adjustable, defaults to 9%/year,
  compounded monthly, contributions at the start of each month (annuity-due).
- **Goal cost lookup**: `city_goal_costs.csv` has 10 area-types per city; since
  the user only supplies a city (not an area), we average all area-type rows
  for that city (`goal_cost_lookup.py`).
- **Feasibility rule**: `required / available <= 1.0` → Achievable;
  `<= 1.25` → Challenging; else → Highly Challenging (student-defined, see
  `financial_calculator.py` docstring).
- **Investment category rule**: ≤3 yrs → capital-preservation; 4–7 yrs →
  balanced; >7 yrs → long-term growth-oriented (broad categories only — never
  an individual stock/fund, never a guaranteed return).
- **Model selection rule**: highest R² on a held-out test split wins; ties
  broken by lower MAE (`train_model.py`).
- **RAG "vector store"**: a local scikit-learn `TfidfVectorizer` matrix instead
  of a paid embeddings API — satisfies "no paid API" while still doing real
  chunk → embed → store → retrieve → ground. Answers require both a minimum
  cosine-similarity score **and** at least 2 shared vocabulary terms with the
  matched chunk, specifically to stop a single coincidentally-shared word
  (e.g. "capital" in "capital of France" vs "capital-preservation") from
  producing a false grounded answer. Known limitation: a genuine one-word
  in-scope question (e.g. just "SIP?") may occasionally be under-grounded;
  ask a fuller question for best results.
- **Agent / prompt-injection defence**: the Agent's Python control flow always
  decides which tool to call and how to calculate — user text can only ever
  change *which* pre-defined tool call happens, never bypass the deterministic
  math or reveal internals. Common injection phrasings ("ignore previous
  instructions", "reveal the system prompt", ...) are also explicitly
  pattern-stripped before intent detection.

## Viva prep — quick answers

- **Why is salary prediction a regression problem?** The target (Monthly
  Salary) is a continuous numeric value, not a category.
- **Why compare 3 models?** No single algorithm is best for every dataset;
  comparing Linear Regression (simple, interpretable baseline), a Decision
  Tree (captures non-linearity, prone to overfitting) and a Random Forest
  (ensemble, usually more robust) lets us pick empirically rather than by
  assumption.
- **Why was the final model selected?** Highest R² on the held-out test set
  (documented, reproducible rule in `train_model.py`), ties broken by MAE.
- **How are City/Education/Job_Role encoded?** `OneHotEncoder(handle_unknown="ignore")`
  inside a `ColumnTransformer`, so unseen categories at prediction time don't crash.
- **Why is Experience excluded?** All users are treated as freshers per the
  assignment's Fresher Rule; it isn't even a column in `salary_data.csv`.
- **How is future cost calculated?** `base_cost * (1 + inflation) ** years`.
- **How is monthly investment calculated?** Future-value-of-an-annuity-due
  formula, solved for the monthly payment, given the future goal cost, the
  expected annual return (compounded monthly) and the number of months.
- **How is the shortfall calculated?** `required_monthly_sip - available_monthly_capacity`
  (positive = shortfall, negative = surplus).
- **How does changing the timeline affect the plan?** Longer years → lower
  required SIP (more time to compound) but higher future cost (more years of
  inflation) — both effects are recomputed by calling `generate_plan` again
  with different `*_years` values (see the `/plan` endpoint, callable
  repeatedly from the frontend's "recalculate" flow).
- **Why tools instead of an LLM for the math?** LLM output is probabilistic
  and can hallucinate numbers; the tools in `agent_tools.py` are deterministic
  and reproducible — the Agent only ever decides *which* tool to call.
- **What is RAG and why here?** Retrieval-Augmented Generation: retrieve
  relevant text chunks from a local knowledge base and ground the answer in
  them, instead of letting a model answer from unverified internal "memory."
- **What happens when RAG can't find relevant info?** It explicitly returns
  "This information is not available in the project knowledge base." instead
  of guessing (see `rag_engine.answer()` and test #15).
- **How does the Agent decide which tool to call?** Rule-based intent
  detection (keyword + regex) over the user's message — see
  `groq_agent.py::_detect_intent` / `_extract_fields`.
- **How did you test hallucination and prompt injection?** `test_15` (off-topic
  question correctly refused) and `test_16` (injection phrase doesn't leak
  internals or bypass tool routing) in `tests/test_cases.py`.

## Still to prepare for full submission (not code)

This delivers all code/data/tests/docs deliverables from the assignment
checklist. Two deliverables are inherently outside what a chat assistant can
produce for you: the **demonstration video** (screen-record yourself running
`run_demo.py`, the API docs, and the frontend) and the **live viva** itself —
the Q&A prep above should cover it.
