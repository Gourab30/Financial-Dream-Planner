"""
Trains and compares 3 regression models to predict Monthly_Salary for a fresher.

Fresher Rule (mandatory): Experience is NOT an input feature anywhere in this
project - it is not in the salary dataset and must never be added. Features
used are exactly: Age, City, Education, Job_Role.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "salary_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")
COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.json")

NUMERIC_FEATURES = ["Age"]
CATEGORICAL_FEATURES = ["City", "Education", "Job_Role"]
TARGET = "Monthly_Salary"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop any unnamed / fully-empty trailing columns caused by a trailing comma in the CSV
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    assert "Experience" not in df.columns, "Fresher rule violation: Experience column must not be used"
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # passes Age through unchanged
    )


def train_and_evaluate(random_state: int = 42) -> dict:
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    candidates = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state, max_depth=5),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state, n_estimators=200, max_depth=8),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results[name] = {"MAE": round(mae, 2), "R2": round(r2, 4)}
        fitted_pipelines[name] = pipe
        print(f"{name:22s} -> MAE: {mae:9.2f} | R2: {r2:.4f}")

    # Documented model-selection rule: highest R2 wins; ties broken by lower MAE.
    best_name = sorted(results.items(), key=lambda kv: (-kv[1]["R2"], kv[1]["MAE"]))[0][0]
    best_pipeline = fitted_pipelines[best_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    comparison_report = {
        "results": results,
        "selected_model": best_name,
        "selection_rule": "Highest R2 on held-out test set; ties broken by lowest MAE.",
        "features_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_features": ["Experience (fresher rule - all users assumed to be freshers)"],
    }
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\nSelected model: {best_name}  (saved to {MODEL_PATH})")
    return comparison_report


if __name__ == "__main__":
    train_and_evaluate()
