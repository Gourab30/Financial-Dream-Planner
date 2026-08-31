"""
Deterministic financial math. NOTHING in this file is ever produced by an LLM -
the Agent only ever *calls* these functions; it never invents these numbers itself.

Inflation assumption: annual goal-cost inflation is set to 6% per year
(DEFAULT_ANNUAL_INFLATION = 0.06), the standard real-world assumption used for
this kind of goal-cost projection in India. It is kept as a single named
constant so it can be changed in one place if a different rate is needed.
"""

DEFAULT_ANNUAL_INFLATION = 0.06          # 6% per year
DEFAULT_MAX_SIP_RATIO = 0.40             # SIP cannot sanely exceed 40% of salary

# Feasibility classification thresholds, applied to (required / available) ratio.
# Documented rule (student-defined, see README section "Feasibility Rule"):
#   ratio <= 1.00               -> "Achievable"
#   1.00 <  ratio <= 1.25       -> "Challenging"      (up to 25% short)
#   ratio  > 1.25               -> "Highly Challenging"
CHALLENGING_THRESHOLD = 1.25


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
        Documented category rule (broad, educational only - never individual stocks
        and never a guaranteed-return claim):
          <= 3 years   -> Short-term  -> capital-preservation oriented (e.g. debt/FD-like)
          4-7 years    -> Medium-term -> diversified balanced (hybrid equity+debt)
          > 7 years    -> Long-term   -> diversified long-term growth (equity-oriented)
        """
        if years <= 3:
            return "Short-Term: Capital-preservation oriented (e.g. debt funds / FDs / liquid funds)"
        if years <= 7:
            return "Medium-Term: Diversified balanced category (hybrid equity + debt funds)"
        return "Long-Term: Diversified long-term growth oriented category (equity-oriented mutual funds)"
