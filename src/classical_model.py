from pathlib import Path
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import MODELS_DIR


class WasteClassifierSVM:
    def __init__(self, C: float = 10.0, gamma: str = "scale"):
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, gamma=gamma, kernel="rbf",
                        probability=True, random_state=42)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WasteClassifierSVM":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def save(self, path: str = None) -> str:
        path = path or str(Path(MODELS_DIR) / "svm_hog.joblib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        return path

    @classmethod
    def load(cls, path: str) -> "WasteClassifierSVM":
        obj = cls.__new__(cls)
        obj._pipeline = joblib.load(path)
        return obj
