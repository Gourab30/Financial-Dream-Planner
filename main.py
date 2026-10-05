import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.schemas import PlanRequest, PlanResponse
from src.planner import FinancialPlanner
from src.goal_cost_lookup import GoalCostLookup
from src.advisor import get_advisor_note
from src.rag.rag_engine import RAGEngine

app = FastAPI(title="AI Financial Dream Planner")

# lets the standalone frontend/index.html (opened as a local file, or served
# from a different port) call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

planner = FinancialPlanner()
lookup = GoalCostLookup()
rag = RAGEngine()


class RagQuery(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"message": "AI Financial Dream Planner API Active"}


@app.get("/cities")
def get_cities():
    return {"cities": lookup.get_cities()}


@app.post("/plan", response_model=PlanResponse)
def create_plan(request: PlanRequest):
    try:
        return planner.generate_plan(
            age=request.age,
            city=request.city,
            education=request.education,
            job_role=request.job_role,
            savings_percent=request.savings_percent,
            expected_return=request.expected_investment_return,
            marriage_years=request.marriage_years,
            car_years=request.car_years,
            # home_years=request.home_years,
            name=request.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/plan/advice")
def create_plan_with_advice(request: PlanRequest):
    
    try:
        plan = planner.generate_plan(
            age=request.age,
            city=request.city,
            education=request.education,
            job_role=request.job_role,
            savings_percent=request.savings_percent,
            expected_return=request.expected_investment_return,
            marriage_years=request.marriage_years,
            car_years=request.car_years,
            # home_years=request.home_years,
            name=request.name,
        )
        plan["ai_advisor_feedback"] = get_advisor_note(plan)
        return plan
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag/query")
def rag_query(payload: RagQuery):
    """Looks the question up in the local knowledge base - returns a grounded
    answer, or says it isn't covered instead of guessing."""
    return rag.answer(payload.query)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
