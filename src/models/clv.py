"""CLV (Closing Line Value) prediction model.

Predicts the direction and magnitude of line movement between opening and
closing Pinnacle odds for each outcome (H/D/A).  Uses LightGBM regression
rather than classification so we can threshold on predicted magnitude.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

import lightgbm as lgb


class CLVModel:
    """LightGBM regressor for predicting closing-line value (line movement).

    Parallel to GradientBoostingModel in baseline.py but uses regression
    (objective="regression") instead of multi-class classification.  Each
    instance predicts CLV for a single outcome (H, D, or A).
    """

    def __init__(self, config: dict):
        """Initialize from config.

        Args:
            config: Dict with LightGBM params (n_estimators, learning_rate, etc.).
        """
        self.config = config
        self.feature_names: list[str] | None = None
        self.imputer: SimpleImputer | None = None
        self.model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "CLVModel":
        """Fit LightGBM regressor with optional early stopping.

        Args:
            X_train: Training features.
            y_train: Continuous target (CLV = closing_prob - opening_prob).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets.

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Impute NaN with median (fit on train only, keep all columns)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = self.imputer.fit_transform(X_train.values)

        y_tr = np.asarray(y_train, dtype=np.float64)

        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.get("n_estimators", 500),
            learning_rate=self.config.get("learning_rate", 0.05),
            max_depth=self.config.get("max_depth", 5),
            num_leaves=self.config.get("num_leaves", 31),
            min_child_samples=self.config.get("min_child_samples", 50),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=self.config.get("random_state", 42),
            verbose=self.config.get("verbose", -1),
            objective="regression",
        )

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            X_v = self.imputer.transform(X_val.values)
            y_v = np.asarray(y_val, dtype=np.float64)
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X_tr, y_tr, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict CLV (continuous value).

        Args:
            X: Feature DataFrame or array.

        Returns:
            1-D array of predicted CLV values.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                avail = [c for c in self.feature_names if c in X.columns]
                X = X[avail]
            X = X.values
        X = self.imputer.transform(X)
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a DataFrame."""
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        """Serialize to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "feature_names": self.feature_names,
                "config": self.config,
                "model_type": "clv_regression",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CLVModel":
        """Deserialize a saved CLV model."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.imputer = data["imputer"]
        obj.feature_names = data["feature_names"]
        obj.config = data["config"]
        return obj
