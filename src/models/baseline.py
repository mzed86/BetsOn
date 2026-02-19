"""Baseline models for match outcome prediction."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


CLASSES = ["H", "D", "A"]
_CLASS_MAP = {"H": 0, "D": 1, "A": 2}


class OddsBaseline:
    """Baseline that returns normalized B365 implied probabilities as predictions.

    No fitting required — just converts bookmaker odds to probabilities.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return normalized implied probabilities from B365 odds.

        Args:
            X: DataFrame that must contain b365_prob_h, b365_prob_d, b365_prob_a.

        Returns:
            (N, 3) array of [P(H), P(D), P(A)].
        """
        probs = X[["b365_prob_h", "b365_prob_d", "b365_prob_a"]].values.copy()
        # Handle any NaN rows by filling with class priors
        nan_mask = np.isnan(probs).any(axis=1)
        if nan_mask.any():
            probs[nan_mask] = [0.437, 0.265, 0.298]  # dataset priors
        # Normalize to sum to 1 (should already be close due to overround removal)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / row_sums
        return probs


class BaselineModel:
    """Logistic regression baseline with imputation and scaling.

    Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression(multinomial)
    """

    def __init__(self, config: dict):
        """Initialize the model pipeline from config.

        Args:
            config: Dict with keys C, max_iter, solver, random_state, multi_class.
        """
        self.config = config
        self.feature_names: list[str] | None = None
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=config.get("C", 1.0),
                max_iter=config.get("max_iter", 1000),
                multi_class=config.get("multi_class", "multinomial"),
                solver=config.get("solver", "lbfgs"),
                random_state=config.get("random_state", 42),
            )),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """Fit the pipeline on training data.

        Args:
            X: Feature DataFrame.
            y: Target series (H/D/A).

        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.pipeline.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame or array.

        Returns:
            (N, 3) array of [P(H), P(D), P(A)], ordered to match CLASSES.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        probs = self.pipeline.predict_proba(X)
        # Ensure column order matches CLASSES [H, D, A]
        clf = self.pipeline.named_steps["clf"]
        class_order = list(clf.classes_)
        if class_order != CLASSES:
            idx = [class_order.index(c) for c in CLASSES]
            probs = probs[:, idx]
        return probs

    def save(self, path: str | Path) -> None:
        """Serialize the full pipeline to disk.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names,
             "config": self.config, "model_type": "logistic_regression"},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        """Deserialize a saved model.

        Args:
            path: Path to saved model file.

        Returns:
            Loaded BaselineModel instance.
        """
        data = joblib.load(path)
        model = cls.__new__(cls)
        model.pipeline = data["pipeline"]
        model.feature_names = data["feature_names"]
        model.config = data["config"]
        return model


class GradientBoostingModel:
    """LightGBM multi-class model with median imputation.

    Can capture non-linear feature interactions that logistic regression misses.
    """

    def __init__(self, config: dict):
        """Initialize from config.

        Args:
            config: Dict with LightGBM params (n_estimators, learning_rate, etc.).
        """
        self.config = config
        self.feature_names: list[str] | None = None
        self.imputer: SimpleImputer | None = None
        self.model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "GradientBoostingModel":
        """Fit LightGBM with optional early stopping on validation set.

        Args:
            X_train: Training features.
            y_train: Training labels (H/D/A).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Impute NaN with median (fit on train only)
        self.imputer = SimpleImputer(strategy="median")
        X_tr = self.imputer.fit_transform(X_train.values)

        # Encode target as integers
        y_tr = np.array([_CLASS_MAP[v] for v in y_train.values])

        self.model = lgb.LGBMClassifier(
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
            num_class=3,
            objective="multiclass",
        )

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            X_v = self.imputer.transform(X_val.values)
            y_v = np.array([_CLASS_MAP[v] for v in y_val.values])
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X_tr, y_tr, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame or array.

        Returns:
            (N, 3) array of [P(H), P(D), P(A)].
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.imputer.transform(X)
        probs = self.model.predict_proba(X)
        # LightGBM class order is [0, 1, 2] = [H, D, A] (our _CLASS_MAP)
        return probs

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
            {"model": self.model, "imputer": self.imputer,
             "feature_names": self.feature_names, "config": self.config,
             "model_type": "gradient_boosting"},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GradientBoostingModel":
        """Deserialize a saved model."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.imputer = data["imputer"]
        obj.feature_names = data["feature_names"]
        obj.config = data["config"]
        return obj
