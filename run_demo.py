import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_model import train_and_evaluate
from src.planner import FinancialPlanner
from src.groq_agent import GroqAgent


def main():
    print("--- 1. Training & Evaluating Salary Prediction Models ---")
    train_and_evaluate()

    print("\n--- 2. Generating Sample Goal-wise Financial Plan ---")
    planner = FinancialPlanner()
    plan = planner.generate_plan(
        age=22,
        city="Bangalore",
        education="B.Tech",
        job_role="Software Engineer",
        savings_percent=20,
        expected_return=0.09,
        marriage_years=5,
        car_years=4,
        home_years=10,
        name="Rahul",
    )
    for key, val in plan.items():
        print(f"{key}: {val}")

    print("\n--- 3. Same Plan via the Natural-Language Agent ---")
    agent = GroqAgent()
    message = (
        "I am 22, live in Bangalore, expect to start as a software engineer, "
        "want to marry in 5 years, buy a car in 4 years and a home in 10 years. "
        "I want to save 20% of my income. What should my plan look like?"
    )
    result = agent.handle_message(message)
    print(result)

    print("\n--- 4. Sample RAG Knowledge Question ---")
    print(agent.handle_message("Why is an emergency fund recommended before investing?"))


if __name__ == "__main__":
    main()
