import os
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "salary_model.pkl")


def _apply_sklearn_compat_shim():
    """The saved model was trained on an older scikit-learn version, and
    newer versions can fail to load it because of an internal class that
    got moved/removed. This just patches that back in so the pickle still
    loads. Doesn't do anything if it's not needed."""
    try:
        from sklearn.compose import _column_transformer as _ct
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass


_apply_sklearn_compat_shim()


class SalaryPredictor:
    """Loads the model trained by train_model.py and predicts a monthly salary
    for a given age/city/education/job role. This is the only place the ML
    model gets used - everything after this is plain arithmetic."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python -m src.train_model` first."
            )
        self.pipeline = joblib.load(model_path)

    def predict_monthly_salary(self, age: int, city: str, education: str, job_role: str) -> float:
        row = pd.DataFrame([{
            "Age": age,
            "City": str(city).strip(),
            "Education": str(education).strip(),
            "Job_Role": str(job_role).strip(),
        }])
        pred = self.pipeline.predict(row)[0]
        return round(max(float(pred), 0.0), 2)
