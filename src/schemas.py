from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PlanRequest(BaseModel):
    name: Optional[str] = None
    age: int = Field(..., ge=18, le=65)
    city: str
    education: str
    job_role: str

    # Each goal is optional - a user can plan for any subset of Marriage / New Car / New Home.
    marriage_years: Optional[int] = Field(None, ge=1, le=50)
    car_years: Optional[int] = Field(None, ge=1, le=50)
    # home_years: Optional[int] = Field(None, ge=1, le=50)

    savings_percent: float = Field(..., description="Percent of monthly income the user wants to invest, e.g. 20 for 20%")
    expected_investment_return: float = Field(0.09, description="Assumed annual investment return, e.g. 0.09 for 9%")

    @field_validator("savings_percent")
    @classmethod
    def validate_savings_percent(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("savings_percent must be between 0 and 100 (exclusive of 0)")
        return v

    @field_validator("expected_investment_return")
    @classmethod
    def validate_return(cls, v):
        if v < 0 or v > 0.5:
            raise ValueError("expected_investment_return should be a reasonable annual decimal, e.g. 0.05-0.20")
        return v


class GoalResult(BaseModel):
    goal_type: str
    years: int
    base_cost_today: float
    future_cost: float
    required_monthly_sip: float
    recommended_category: str
    status: str  # Achievable / Challenging / Highly Challenging (vs total capacity, standalone)


class PlanResponse(BaseModel):
    name: Optional[str]
    predicted_monthly_salary: float
    available_monthly_capacity: float
    goals: list[GoalResult]
    total_required_monthly_sip: float
    overall_shortfall_or_surplus: float
    overall_status: str
    assumptions: dict
