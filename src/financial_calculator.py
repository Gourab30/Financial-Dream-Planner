
DEFAULT_ANNUAL_INFLATION = 0.06          # 6% per year
DEFAULT_MAX_SIP_RATIO = 0.40             # SIP cannot sanely exceed 40% of salary

# ratio = required SIP / available capacity
#   <= 1.00  -> Achievable
#   <= 1.25  -> Challenging (short by up to 25%)
#   >  1.25  -> Highly Challenging
CHALLENGING_THRESHOLD = 1.25

CAR_SIP_PERCENT = 0.80
CAR_LOAN_PERCENT = 0.20
class FinancialCalculator:
    """Pure, deterministic financial calculations. No randomness, no LLM calls."""

    @staticmethod
    def calculate_future_goal_cost(base_cost: float, annual_inflation: float = DEFAULT_ANNUAL_INFLATION,
                                    years: int = 0) -> float:
        if base_cost < 0:
            raise ValueError("base_cost cannot be negative")
        if years < 0:
            raise ValueError("years cannot be negative")
        return round(base_cost * ((1 + annual_inflation) ** years), 2)

    @staticmethod
    def calculate_required_monthly_sip(target_amount: float, annual_return_rate: float, years: int) -> float:
        if years <= 0:
            raise ValueError("years must be a positive integer")
        if target_amount < 0:
            raise ValueError("target_amount cannot be negative")

        months = years * 12
        monthly_rate = annual_return_rate / 12
        if monthly_rate == 0:
            return round(target_amount / months, 2)

        factor = (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        monthly_investment = target_amount / factor
        return round(monthly_investment, 2)

    @staticmethod
    def check_feasibility(projected_monthly_salary: float, required_monthly_sip: float,
                           max_sip_ratio: float = DEFAULT_MAX_SIP_RATIO):
        if projected_monthly_salary <= 0:
            raise ValueError("projected_monthly_salary must be positive")
        sip_ratio = required_monthly_sip / projected_monthly_salary
        is_feasible = sip_ratio <= max_sip_ratio
        return is_feasible, round(sip_ratio * 100, 2)

    @staticmethod
    def calculate_shortfall(available_monthly_capacity: float, required_monthly_sip: float) -> float:
        """Positive = shortfall (need more money). Negative = surplus."""
        return round(required_monthly_sip - available_monthly_capacity, 2)

    @staticmethod
    def classify_feasibility(available_monthly_capacity: float, required_monthly_sip: float) -> str:
        """Returns one of: Achievable / Challenging / Highly Challenging."""
        if available_monthly_capacity <= 0:
            return "Highly Challenging"
        ratio = required_monthly_sip / available_monthly_capacity
        if ratio <= 1.0:
            return "Achievable"
        if ratio <= CHALLENGING_THRESHOLD:
            return "Challenging"
        return "Highly Challenging"

    @staticmethod
    def recommend_investment_category(years: int) -> str:
        """
        Broad category based on how far away the goal is. Educational only -
        never a specific stock/fund, never a promised return.
          <= 3 years  -> short-term, capital-preservation (debt funds / FDs)
          4-7 years   -> medium-term, balanced mix of equity + debt
          > 7 years   -> long-term, more equity-heavy for growth
        """
        if years <= 3:
            return "Short-Term: Capital-preservation oriented (e.g. debt funds / FDs / liquid funds)"
        if years <= 7:
            return "Medium-Term: Diversified balanced category (hybrid equity + debt funds)"
        return "Long-Term: Diversified long-term growth oriented category (equity-oriented mutual funds)"
    @staticmethod
    def calculate_car_financing(future_car_cost: float):
        sip_amount = future_car_cost * CAR_SIP_PERCENT
        loan_amount = future_car_cost * CAR_LOAN_PERCENT

        return round(sip_amount, 2), round(loan_amount, 2)