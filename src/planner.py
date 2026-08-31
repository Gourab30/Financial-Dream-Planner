from src.financial_calculator import FinancialCalculator, DEFAULT_ANNUAL_INFLATION
from src.goal_cost_lookup import GoalCostLookup
from src.salary_predictor import SalaryPredictor

GOAL_YEAR_FIELDS = {
    "Marriage": "marriage_years",
    "New Car": "car_years",
    "New Home": "home_years",
}


class FinancialPlanner:
    """
    Orchestrates: Salary Prediction Tool -> Future Cost Tool -> Investment
    Calculator Tool -> Feasibility Tool -> Recommendation Tool.
    Every number returned here is deterministic; the ML model is used ONLY for
    the salary prediction step.
    """

    def __init__(self):
        self.calculator = FinancialCalculator()
        self.cost_lookup = GoalCostLookup()
        self.salary_predictor = SalaryPredictor()

    def _validate_categoricals(self, city, education, job_role):
        cities = self.cost_lookup.get_cities()
        if city not in cities:
            raise ValueError(f"Unknown city '{city}'. Supported cities: {', '.join(cities)}")
        # Education / Job_Role unknowns are allowed through to the model (OneHotEncoder
        # handles unseen categories), but we flag it so the caller can warn the user.

    def generate_plan(self, age: int, city: str, education: str, job_role: str,
                       savings_percent: float, expected_return: float = 0.09,
                       marriage_years: int = None, car_years: int = None, home_years: int = None,
                       name: str = None, annual_inflation: float = DEFAULT_ANNUAL_INFLATION) -> dict:

        if not any([marriage_years, car_years, home_years]):
            raise ValueError("At least one goal (marriage_years, car_years or home_years) must be provided")
        if savings_percent <= 0 or savings_percent > 100:
            raise ValueError("savings_percent must be between 0 and 100")

        self._validate_categoricals(city, education, job_role)

        predicted_salary = self.salary_predictor.predict_monthly_salary(age, city, education, job_role)
        available_capacity = round(predicted_salary * (savings_percent / 100.0), 2)

        requested_goals = {
            "Marriage": marriage_years,
            "New Car": car_years,
            "New Home": home_years,
        }

        goal_results = []
        total_required = 0.0

        for goal_type, years in requested_goals.items():
            if years is None:
                continue
            base_cost = self.cost_lookup.get_base_cost_for_goal(city, goal_type)
            future_cost = self.calculator.calculate_future_goal_cost(base_cost, annual_inflation, years)
            required_sip = self.calculator.calculate_required_monthly_sip(future_cost, expected_return, years)
            category = self.calculator.recommend_investment_category(years)
            # Standalone status: could this goal alone be funded from full capacity?
            status = self.calculator.classify_feasibility(available_capacity, required_sip)

            total_required += required_sip
            goal_results.append({
                "goal_type": goal_type,
                "years": years,
                "base_cost_today": base_cost,
                "future_cost": future_cost,
                "required_monthly_sip": required_sip,
                "recommended_category": category,
                "status": status,
            })

        total_required = round(total_required, 2)
        shortfall_or_surplus = self.calculator.calculate_shortfall(available_capacity, total_required)
        overall_status = self.calculator.classify_feasibility(available_capacity, total_required)

        return {
            "name": name,
            "predicted_monthly_salary": predicted_salary,
            "available_monthly_capacity": available_capacity,
            "goals": goal_results,
            "total_required_monthly_sip": total_required,
            "overall_shortfall_or_surplus": shortfall_or_surplus,
            "overall_status": overall_status,
            "assumptions": {
                "annual_inflation": annual_inflation,
                "expected_investment_return": expected_return,
                "savings_percent": savings_percent,
                "max_sip_ratio_warning": 0.40,
                "note": "Educational simulation only. Not professional financial advice. Returns are not guaranteed.",
            },
        }
