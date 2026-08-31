"""
Tool definitions available to the Agent. Per assignment rule 10.3 / viva Q11:
"Why should financial calculations be implemented as tools instead of relying
on an LLM?" - because LLM output is probabilistic and can hallucinate numbers,
while these functions are deterministic and reproducible. The Agent's ONLY
job is to decide which tool(s) to call and with what arguments; it is never
allowed to compute a financial number itself.
"""
from src.financial_calculator import FinancialCalculator, DEFAULT_ANNUAL_INFLATION
from src.goal_cost_lookup import GoalCostLookup
from src.salary_predictor import SalaryPredictor

_calc = FinancialCalculator()
_lookup = GoalCostLookup()
_salary_model = SalaryPredictor()


def salary_prediction_tool(age: int, city: str, education: str, job_role: str) -> dict:
    """Tool: calls the trained ML regression model to predict current monthly salary."""
    salary = _salary_model.predict_monthly_salary(age, city, education, job_role)
    return {"tool": "salary_prediction_tool", "predicted_monthly_salary": salary}


def future_cost_tool(city: str, goal_type: str, years: int, annual_inflation: float = DEFAULT_ANNUAL_INFLATION) -> dict:
    """Tool: looks up today's base cost for a goal/city and inflates it forward."""
    base_cost = _lookup.get_base_cost_for_goal(city, goal_type)
    future_cost = _calc.calculate_future_goal_cost(base_cost, annual_inflation, years)
    return {"tool": "future_cost_tool", "base_cost_today": base_cost, "future_cost": future_cost}


def investment_calculator_tool(target_amount: float, annual_return_rate: float, years: int) -> dict:
    """Tool: calculates the required monthly SIP to reach target_amount in `years`."""
    sip = _calc.calculate_required_monthly_sip(target_amount, annual_return_rate, years)
    return {"tool": "investment_calculator_tool", "required_monthly_sip": sip}


def feasibility_tool(available_monthly_capacity: float, required_monthly_sip: float) -> dict:
    """Tool: computes shortfall/surplus and Achievable/Challenging/Highly Challenging status."""
    shortfall = _calc.calculate_shortfall(available_monthly_capacity, required_monthly_sip)
    status = _calc.classify_feasibility(available_monthly_capacity, required_monthly_sip)
    return {"tool": "feasibility_tool", "shortfall_or_surplus": shortfall, "status": status}


def recommendation_tool(years: int) -> dict:
    """Tool: selects a broad investment category using the documented time-horizon rule."""
    category = _calc.recommend_investment_category(years)
    return {"tool": "recommendation_tool", "recommended_category": category}


TOOL_REGISTRY = {
    "salary_prediction_tool": salary_prediction_tool,
    "future_cost_tool": future_cost_tool,
    "investment_calculator_tool": investment_calculator_tool,
    "feasibility_tool": feasibility_tool,
    "recommendation_tool": recommendation_tool,
}
