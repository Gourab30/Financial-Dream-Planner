import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "city_goal_costs.csv")

GOAL_COLUMN_MAP = {
    "Marriage": "Marriage_Cost_Current",
    "New Car": "Car_Cost_Current",
    "New Home": "Home_Cost_Current",
}


class GoalCostLookup:
    """
    Looks up today's base cost of a goal (Marriage / New Car / New Home) for a city.

    The dataset has several Area_Type rows per city (Central, North, Premium, ...).
    Since the user only supplies a City (not an area type), we use the city-wide
    AVERAGE across all area types as the base cost. This is a documented,
    reproducible rule (see README "Goal Cost Lookup Rule").
    """

    def __init__(self, csv_path: str = DATA_PATH):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"City/goal cost dataset not found at {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df["City"] = self.df["City"].astype(str).str.strip()

        self._city_avg = self.df.groupby("City")[list(GOAL_COLUMN_MAP.values())].mean().round(2)
        self._cities_lower = {c.lower(): c for c in self._city_avg.index}

    def get_cities(self):
        return sorted(self._city_avg.index.tolist())

    def get_base_costs(self, city: str) -> dict:
        """Returns {'Marriage': x, 'New Car': y, 'New Home': z} base (current) cost for a city."""
        key = self._cities_lower.get(str(city).strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown city '{city}'. Supported cities: {', '.join(self.get_cities())}"
            )
        row = self._city_avg.loc[key]
        return {goal: float(row[col]) for goal, col in GOAL_COLUMN_MAP.items()}

    def get_base_cost_for_goal(self, city: str, goal_type: str) -> float:
        if goal_type not in GOAL_COLUMN_MAP:
            raise ValueError(f"Unknown goal_type '{goal_type}'. Must be one of {list(GOAL_COLUMN_MAP.keys())}")
        return self.get_base_costs(city)[goal_type]
