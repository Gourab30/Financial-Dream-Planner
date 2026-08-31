import os
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "salary_model.pkl")


def _apply_sklearn_compat_shim():
    """The bundled salary_model.pkl was trained with an older scikit-learn
    version. Newer scikit-learn versions can fail to unpickle it because an
    internal (private) helper class referenced by the pickle no longer
    exists. This adds a harmless stand-in so the file still loads correctly
    on newer scikit-learn installs. Safe to run on any version -  it's a
    no-op if the class is already present or the module layout differs.
    """
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
    """Loads the model trained by train_model.py and predicts current monthly salary.

    This is the ONLY place a machine-learning model is used for a number in the
    whole app (the "Salary Prediction Tool"). Everything downstream (future goal
    cost, SIP, feasibility) is deterministic arithmetic in financial_calculator.py.
    """

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
