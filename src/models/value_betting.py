"""Value betting models: DisagreementSelector, CLVClassifier, MetaModel.

Three strategies for identifying profitable bets:
  A) DisagreementSelector — no ML, selects bets where B365 odds are generous
     relative to Pinnacle opening (percentile-based cutoffs).
  B) CLVClassifier — binary classifier for P(CLV > threshold).
  C) MetaModel — stacking ensemble combining all signals.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


OUTCOMES = ["H", "D", "A"]
OUTCOME_SUFFIXES = {"H": "h", "D": "d", "A": "a"}
OUTCOME_CODES = {"H": 0, "D": 1, "A": 2}

# B365 odds column names per outcome
B365_ODDS_COLS = {"H": "B365H", "D": "B365D", "A": "B365A"}
B365_PROB_COLS = {"H": "b365_prob_h", "D": "b365_prob_d", "A": "b365_prob_a"}
PS_PROB_COLS = {"H": "ps_prob_h", "D": "ps_prob_d", "A": "ps_prob_a"}
DISAGREE_COLS = {"H": "odds_disagree_h", "D": "odds_disagree_d", "A": "odds_disagree_a"}


class DisagreementSelector:
    """Percentile-based selector: bet where B365 is most generous vs Pinnacle.

    No ML — learns percentile cutoffs from training data.  A low percentile
    means B365 assigns a LOWER probability (= HIGHER odds = more generous)
    than Pinnacle for that outcome.
    """

    def __init__(self, config: dict):
        """Initialize from disagreement config section.

        Args:
            config: Dict with 'percentiles' list and 'default_percentile'.
        """
        self.percentiles = config.get("percentiles", [1, 2, 3, 5, 7, 10, 15, 20])
        self.default_percentile = config.get("default_percentile", 3)
        self.percentile_cutoffs: dict[int, dict[str, float]] | None = None

    def fit(self, df: pd.DataFrame) -> "DisagreementSelector":
        """Learn percentile cutoffs from training data.

        For each outcome, computes odds_disagree = b365_prob - ps_prob.
        Lower disagree means B365 gives lower probability = higher odds = more generous.

        Args:
            df: Training DataFrame with odds_disagree_h/d/a columns.

        Returns:
            self
        """
        self.percentile_cutoffs = {}
        for p in self.percentiles:
            self.percentile_cutoffs[p] = {}
            for outcome in OUTCOMES:
                suffix = OUTCOME_SUFFIXES[outcome]
                col = f"odds_disagree_{suffix}"
                values = df[col].dropna().values
                self.percentile_cutoffs[p][suffix] = float(np.nanpercentile(values, p))
        return self

    def select(self, df: pd.DataFrame, percentile: int | None = None) -> dict[str, np.ndarray]:
        """Select bets where B365 is most generous (disagree below cutoff).

        Args:
            df: DataFrame with odds_disagree_h/d/a columns.
            percentile: Which percentile cutoff to use. Defaults to default_percentile.

        Returns:
            Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.

        Raises:
            ValueError: If percentile not in fitted cutoffs.
        """
        if self.percentile_cutoffs is None:
            raise RuntimeError("DisagreementSelector not fitted. Call fit() first.")

        if percentile is None:
            percentile = self.default_percentile

        if percentile not in self.percentile_cutoffs:
            raise ValueError(
                f"Percentile {percentile} not in fitted cutoffs. "
                f"Available: {sorted(self.percentile_cutoffs.keys())}"
            )

        masks = {}
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            col = f"odds_disagree_{suffix}"
            cutoff = self.percentile_cutoffs[percentile][suffix]
            values = df[col].values
            masks[outcome] = values <= cutoff
        return masks

    def fit_multibook(self, df: pd.DataFrame) -> "DisagreementSelector":
        """Learn percentile cutoffs for best-across-books disagreement.

        Uses odds_disagree_best_{suffix} = min_book_prob - ps_prob, which
        represents the most generous book's probability minus Pinnacle.

        Args:
            df: Training DataFrame with odds_disagree_best_h/d/a columns.

        Returns:
            self
        """
        self.multibook_cutoffs: dict[int, dict[str, float]] = {}
        for p in self.percentiles:
            self.multibook_cutoffs[p] = {}
            for outcome in OUTCOMES:
                suffix = OUTCOME_SUFFIXES[outcome]
                col = f"odds_disagree_best_{suffix}"
                if col in df.columns:
                    values = df[col].dropna().values
                    if len(values) > 0:
                        self.multibook_cutoffs[p][suffix] = float(np.nanpercentile(values, p))
                    else:
                        self.multibook_cutoffs[p][suffix] = np.nan
                else:
                    self.multibook_cutoffs[p][suffix] = np.nan
        return self

    def select_multibook(
        self, df: pd.DataFrame, percentile: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Select bets where best-book disagreement is below cutoff.

        Args:
            df: DataFrame with odds_disagree_best_h/d/a columns.
            percentile: Which percentile cutoff to use.

        Returns:
            Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.
        """
        if not hasattr(self, "multibook_cutoffs") or self.multibook_cutoffs is None:
            raise RuntimeError("Multibook cutoffs not fitted. Call fit_multibook() first.")

        if percentile is None:
            percentile = self.default_percentile

        if percentile not in self.multibook_cutoffs:
            raise ValueError(
                f"Percentile {percentile} not in multibook cutoffs. "
                f"Available: {sorted(self.multibook_cutoffs.keys())}"
            )

        masks = {}
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            col = f"odds_disagree_best_{suffix}"
            cutoff = self.multibook_cutoffs[percentile][suffix]
            if col in df.columns and np.isfinite(cutoff):
                masks[outcome] = df[col].values <= cutoff
            else:
                # Fall back to standard disagreement
                col_std = f"odds_disagree_{suffix}"
                if self.percentile_cutoffs is not None and percentile in self.percentile_cutoffs:
                    cutoff_std = self.percentile_cutoffs[percentile][suffix]
                    masks[outcome] = df[col_std].values <= cutoff_std
                else:
                    masks[outcome] = np.zeros(len(df), dtype=bool)
        return masks

    def compute_league_filter(
        self,
        df: pd.DataFrame,
        ftr: np.ndarray | pd.Series,
        b365_h: np.ndarray | pd.Series,
        b365_d: np.ndarray | pd.Series,
        b365_a: np.ndarray | pd.Series,
        leagues: np.ndarray | pd.Series,
        percentile: int,
        min_roi_pct: float = 0.0,
        min_bets: int = 20,
    ) -> set[str]:
        """Compute per-league ROI and return profitable league names.

        Args:
            df: DataFrame with odds_disagree columns.
            ftr: Match results (H/D/A).
            b365_h/d/a: B365 decimal odds.
            leagues: League identifier per match.
            percentile: Which percentile cutoff to use.
            min_roi_pct: Minimum ROI to include a league.
            min_bets: Minimum number of bets to include a league.

        Returns:
            Set of league names where ROI >= min_roi_pct and bets >= min_bets.
        """
        from src.evaluation.value_metrics import compute_flat_stake_roi

        ftr = np.asarray(ftr)
        leagues = np.asarray(leagues)
        b365_h = np.asarray(b365_h, dtype=np.float64)
        b365_d = np.asarray(b365_d, dtype=np.float64)
        b365_a = np.asarray(b365_a, dtype=np.float64)

        masks = self.select(df, percentile=percentile)
        profitable = set()

        for league in np.unique(leagues):
            l_mask = leagues == league
            league_masks = {
                o: np.asarray(masks[o], dtype=bool) & l_mask for o in OUTCOMES
            }
            roi = compute_flat_stake_roi(ftr, league_masks, b365_h, b365_d, b365_a)
            if roi["n_bets"] >= min_bets and roi["roi_pct"] >= min_roi_pct:
                profitable.add(str(league))

        return profitable

    def select_with_league_filter(
        self,
        df: pd.DataFrame,
        leagues: np.ndarray | pd.Series,
        percentile: int | None = None,
        allowed_leagues: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Select bets, then filter to allowed leagues only.

        Args:
            df: DataFrame with odds_disagree columns.
            leagues: League identifier per match.
            percentile: Which percentile cutoff to use.
            allowed_leagues: Set of league names to allow. If None, no filtering.

        Returns:
            Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.
        """
        masks = self.select(df, percentile=percentile)
        if allowed_leagues is None:
            return masks

        leagues = np.asarray(leagues)
        league_mask = np.isin(leagues, list(allowed_leagues))
        return {o: masks[o] & league_mask for o in OUTCOMES}

    def save(self, path: str | Path) -> None:
        """Serialize to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "percentiles": self.percentiles,
                "default_percentile": self.default_percentile,
                "percentile_cutoffs": self.percentile_cutoffs,
                "multibook_cutoffs": getattr(self, "multibook_cutoffs", None),
                "profitable_leagues": getattr(self, "profitable_leagues", None),
                "model_type": "disagreement_selector",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DisagreementSelector":
        """Deserialize from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.percentiles = data["percentiles"]
        obj.default_percentile = data["default_percentile"]
        obj.percentile_cutoffs = data["percentile_cutoffs"]
        obj.multibook_cutoffs = data.get("multibook_cutoffs", None)
        obj.profitable_leagues = data.get("profitable_leagues", None)
        return obj


class GenericDisagreementSelector:
    """Market-generic percentile-based disagreement selector.

    Works with arbitrary outcome sets (e.g. Over/Under, H/D/A for corners).
    """

    def __init__(
        self,
        outcomes: list[str],
        outcome_suffixes: dict[str, str],
        config: dict,
        disagree_prefix: str = "odds_disagree",
    ):
        """Initialize from config.

        Args:
            outcomes: List of outcome labels (e.g. ["Over", "Under"]).
            outcome_suffixes: Map outcome -> column suffix (e.g. {"Over": "over"}).
            config: Dict with 'percentiles' and 'default_percentile'.
            disagree_prefix: Column prefix for disagreement values.
        """
        self.outcomes = outcomes
        self.outcome_suffixes = outcome_suffixes
        self.percentiles = config.get("percentiles", [1, 2, 3, 5, 7, 10, 15, 20])
        self.default_percentile = config.get("default_percentile", 5)
        self.disagree_prefix = disagree_prefix
        self.percentile_cutoffs: dict[int, dict[str, float]] | None = None
        self.profitable_leagues: set[str] | None = None

    def fit(self, df: pd.DataFrame) -> "GenericDisagreementSelector":
        """Learn percentile cutoffs from training data.

        Args:
            df: Training DataFrame with disagreement columns.

        Returns:
            self
        """
        self.percentile_cutoffs = {}
        for p in self.percentiles:
            self.percentile_cutoffs[p] = {}
            for outcome in self.outcomes:
                suffix = self.outcome_suffixes[outcome]
                col = f"{self.disagree_prefix}_{suffix}"
                values = df[col].dropna().values
                self.percentile_cutoffs[p][suffix] = float(np.nanpercentile(values, p))
        return self

    def select(
        self, df: pd.DataFrame, percentile: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Select bets where disagreement is below the percentile cutoff.

        Args:
            df: DataFrame with disagreement columns.
            percentile: Which percentile cutoff to use. Defaults to default_percentile.

        Returns:
            Dict mapping outcome -> boolean mask.
        """
        if self.percentile_cutoffs is None:
            raise RuntimeError("GenericDisagreementSelector not fitted. Call fit() first.")

        if percentile is None:
            percentile = self.default_percentile

        if percentile not in self.percentile_cutoffs:
            raise ValueError(
                f"Percentile {percentile} not in fitted cutoffs. "
                f"Available: {sorted(self.percentile_cutoffs.keys())}"
            )

        masks = {}
        for outcome in self.outcomes:
            suffix = self.outcome_suffixes[outcome]
            col = f"{self.disagree_prefix}_{suffix}"
            cutoff = self.percentile_cutoffs[percentile][suffix]
            masks[outcome] = df[col].values <= cutoff
        return masks

    def compute_league_filter(
        self,
        df: pd.DataFrame,
        results: np.ndarray | pd.Series,
        odds: dict[str, np.ndarray],
        leagues: np.ndarray | pd.Series,
        percentile: int,
        min_roi_pct: float = 0.0,
        min_bets: int = 10,
    ) -> set[str]:
        """Compute per-league ROI and return profitable league names.

        Args:
            df: DataFrame with disagreement columns.
            results: Actual outcomes array.
            odds: Dict outcome -> decimal odds arrays.
            leagues: League identifier per match.
            percentile: Which percentile cutoff to use.
            min_roi_pct: Minimum ROI to include a league.
            min_bets: Minimum number of bets to include a league.

        Returns:
            Set of profitable league names.
        """
        from src.evaluation.value_metrics import compute_flat_stake_roi_generic

        results = np.asarray(results)
        leagues = np.asarray(leagues)

        masks = self.select(df, percentile=percentile)
        profitable = set()

        for league in np.unique(leagues):
            l_mask = leagues == league
            league_masks = {
                o: np.asarray(masks[o], dtype=bool) & l_mask for o in self.outcomes
            }
            league_odds = {o: np.asarray(odds[o], dtype=np.float64) for o in odds}
            roi = compute_flat_stake_roi_generic(results, league_masks, league_odds)
            if roi["n_bets"] >= min_bets and roi["roi_pct"] >= min_roi_pct:
                profitable.add(str(league))

        return profitable

    def select_with_league_filter(
        self,
        df: pd.DataFrame,
        leagues: np.ndarray | pd.Series,
        percentile: int | None = None,
        allowed_leagues: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Select bets, then filter to allowed leagues only.

        Args:
            df: DataFrame with disagreement columns.
            leagues: League identifier per match.
            percentile: Which percentile cutoff to use.
            allowed_leagues: Set of league names to allow. If None, no filtering.

        Returns:
            Dict outcome -> boolean mask.
        """
        masks = self.select(df, percentile=percentile)
        if allowed_leagues is None:
            return masks

        leagues = np.asarray(leagues)
        league_mask = np.isin(leagues, list(allowed_leagues))
        return {o: masks[o] & league_mask for o in self.outcomes}

    def save(self, path: str | Path) -> None:
        """Serialize to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "outcomes": self.outcomes,
                "outcome_suffixes": self.outcome_suffixes,
                "percentiles": self.percentiles,
                "default_percentile": self.default_percentile,
                "disagree_prefix": self.disagree_prefix,
                "percentile_cutoffs": self.percentile_cutoffs,
                "profitable_leagues": self.profitable_leagues,
                "model_type": "generic_disagreement_selector",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GenericDisagreementSelector":
        """Deserialize from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.outcomes = data["outcomes"]
        obj.outcome_suffixes = data["outcome_suffixes"]
        obj.percentiles = data["percentiles"]
        obj.default_percentile = data["default_percentile"]
        obj.disagree_prefix = data["disagree_prefix"]
        obj.percentile_cutoffs = data["percentile_cutoffs"]
        obj.profitable_leagues = data.get("profitable_leagues", None)
        return obj


class CLVClassifier:
    """Binary classifier for P(CLV > threshold).

    Uses LightGBM with is_unbalance=True since positive CLV events are rare.
    One classifier per outcome (H, D, A).
    """

    def __init__(self, config: dict, clv_threshold: float = 0.02):
        """Initialize from config.

        Args:
            config: Dict with LightGBM params.
            clv_threshold: CLV threshold for binarizing target.
        """
        self.config = config
        self.clv_threshold = clv_threshold
        self.feature_names: list[str] | None = None
        self.imputer: SimpleImputer | None = None
        self.model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_clv: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val_clv: pd.Series | np.ndarray | None = None,
    ) -> "CLVClassifier":
        """Fit binary classifier on binarized CLV target.

        Args:
            X_train: Training features.
            y_train_clv: Continuous CLV values (will be binarized).
            X_val: Optional validation features for early stopping.
            y_val_clv: Optional validation CLV (will be binarized).

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Impute NaN with median (keep all columns even if all-NaN)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = self.imputer.fit_transform(X_train.values)

        # Binarize: 1 if CLV > threshold, else 0
        y_tr = (np.asarray(y_train_clv, dtype=np.float64) > self.clv_threshold).astype(int)

        self.model = lgb.LGBMClassifier(
            n_estimators=self.config.get("n_estimators", 800),
            learning_rate=self.config.get("learning_rate", 0.03),
            max_depth=self.config.get("max_depth", 5),
            num_leaves=self.config.get("num_leaves", 31),
            min_child_samples=self.config.get("min_child_samples", 50),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            reg_alpha=self.config.get("reg_alpha", 0.1),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=self.config.get("random_state", 42),
            verbose=self.config.get("verbose", -1),
            objective="binary",
            is_unbalance=True,
        )

        fit_kwargs: dict = {}
        if X_val is not None and y_val_clv is not None:
            X_v = self.imputer.transform(X_val.values)
            y_v = (np.asarray(y_val_clv, dtype=np.float64) > self.clv_threshold).astype(int)
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X_tr, y_tr, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict P(CLV > threshold).

        Args:
            X: Feature DataFrame or array.

        Returns:
            1-D array of probabilities.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                avail = [c for c in self.feature_names if c in X.columns]
                X = X[avail]
            X = X.values
        X = self.imputer.transform(X)
        # predict_proba returns (N, 2): [P(0), P(1)]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame | np.ndarray, cutoff: float = 0.5) -> np.ndarray:
        """Predict binary decision (bet or not).

        Args:
            X: Feature DataFrame or array.
            cutoff: Probability cutoff for positive prediction.

        Returns:
            1-D boolean array.
        """
        return self.predict_proba(X) >= cutoff

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
                "clv_threshold": self.clv_threshold,
                "model_type": "clv_classifier",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CLVClassifier":
        """Deserialize from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.imputer = data["imputer"]
        obj.feature_names = data["feature_names"]
        obj.config = data["config"]
        obj.clv_threshold = data["clv_threshold"]
        return obj


class MetaModel:
    """Stacking ensemble combining disagreement, CLV classifier, CLV regressor,
    and outcome model signals.

    Operates in long format: 3 rows per match (one per outcome H/D/A).
    Predicts P(profitable) for each (match, outcome) pair.
    """

    def __init__(self, config: dict):
        """Initialize from meta_model config section.

        Args:
            config: Dict with 'learner' key and learner-specific params.
        """
        self.config = config
        self.learner_type = config.get("learner", "logistic_regression")
        self.feature_names: list[str] | None = None
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.model = None

    @staticmethod
    def build_meta_features(
        df: pd.DataFrame,
        disagree_selector: DisagreementSelector | None = None,
        clv_classifiers: dict[str, CLVClassifier] | None = None,
        clv_regressors: dict | None = None,
        outcome_model=None,
        dc_model=None,
        clv_feature_cols: list[str] | None = None,
        outcome_feature_cols: list[str] | None = None,
        percentile: int = 3,
    ) -> pd.DataFrame:
        """Build long-format meta-features: 3 rows per match (one per outcome).

        Args:
            df: Match-level DataFrame with odds, features, etc.
            disagree_selector: Fitted DisagreementSelector (or None).
            clv_classifiers: Dict outcome -> fitted CLVClassifier (or None).
            clv_regressors: Dict outcome -> fitted CLVModel (or None).
            outcome_model: Fitted GradientBoostingModel (or None).
            dc_model: Fitted DixonColesModel (or None).
            clv_feature_cols: Feature columns for CLV models.
            outcome_feature_cols: Feature columns for outcome model.
            percentile: Percentile to use for disagreement selector.

        Returns:
            Long-format DataFrame with meta-features indexed by (match_idx, outcome).
        """
        n = len(df)
        rows = []

        # Pre-compute signals that are per-match
        # Disagreement masks
        disagree_masks = None
        if disagree_selector is not None and disagree_selector.percentile_cutoffs is not None:
            try:
                disagree_masks = disagree_selector.select(df, percentile)
            except (ValueError, KeyError):
                disagree_masks = None

        # CLV classifier probabilities
        clv_cls_probs = {}
        if clv_classifiers is not None and clv_feature_cols is not None:
            avail = [c for c in clv_feature_cols if c in df.columns]
            for outcome in OUTCOMES:
                if outcome in clv_classifiers:
                    clv_cls_probs[outcome] = clv_classifiers[outcome].predict_proba(df[avail])

        # CLV regressor predictions
        clv_reg_preds = {}
        if clv_regressors is not None and clv_feature_cols is not None:
            avail = [c for c in clv_feature_cols if c in df.columns]
            for outcome in OUTCOMES:
                if outcome in clv_regressors:
                    clv_reg_preds[outcome] = clv_regressors[outcome].predict(df[avail])

        # Outcome model predictions
        outcome_probs = None
        if outcome_model is not None and outcome_feature_cols is not None:
            avail = [c for c in outcome_feature_cols if c in df.columns]
            if len(avail) > 0:
                outcome_probs = outcome_model.predict_proba(df[avail])

        # Dixon-Coles predictions
        dc_probs = None
        if dc_model is not None and "home_team" in df.columns and "away_team" in df.columns:
            dc_probs = dc_model.predict_proba(df["home_team"].values, df["away_team"].values)

        # Build long-format rows
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            outcome_idx = OUTCOME_CODES[outcome]
            b365_odds_col = B365_ODDS_COLS[outcome]
            b365_prob_col = B365_PROB_COLS[outcome]
            ps_prob_col = PS_PROB_COLS[outcome]
            disagree_col = DISAGREE_COLS[outcome]

            row_data = {
                "match_idx": np.arange(n),
                "outcome": outcome,
                "outcome_code": outcome_idx,
            }

            # Raw disagreement and rank
            if disagree_col in df.columns:
                disagree_raw = df[disagree_col].values
                row_data["disagree_raw"] = disagree_raw
                valid = np.isfinite(disagree_raw)
                rank = np.full(n, np.nan)
                if valid.sum() > 0:
                    from scipy.stats import rankdata
                    rank[valid] = rankdata(disagree_raw[valid]) / valid.sum()
                row_data["disagree_rank"] = rank

            # Disagreement selector mask
            if disagree_masks is not None and outcome in disagree_masks:
                row_data["disagree_selected"] = disagree_masks[outcome].astype(float)

            # CLV classifier probability
            if outcome in clv_cls_probs:
                row_data["clv_classifier_prob"] = clv_cls_probs[outcome]

            # CLV regressor prediction
            if outcome in clv_reg_preds:
                row_data["clv_regressor_pred"] = clv_reg_preds[outcome]

            # Outcome model probability and edge
            if outcome_probs is not None:
                row_data["outcome_model_prob"] = outcome_probs[:, outcome_idx]
                if b365_prob_col in df.columns:
                    row_data["outcome_model_edge"] = (
                        outcome_probs[:, outcome_idx] - df[b365_prob_col].values
                    )

            # Dixon-Coles probability and edge
            if dc_probs is not None:
                row_data["dc_prob"] = dc_probs[:, outcome_idx]
                if b365_prob_col in df.columns:
                    row_data["dc_edge"] = (
                        dc_probs[:, outcome_idx] - df[b365_prob_col].values
                    )

            # Odds features
            if b365_odds_col in df.columns:
                row_data["b365_odds"] = df[b365_odds_col].values
            if b365_prob_col in df.columns:
                row_data["b365_implied"] = df[b365_prob_col].values
            if ps_prob_col in df.columns:
                row_data["ps_opening_prob"] = df[ps_prob_col].values

            rows.append(pd.DataFrame(row_data))

        meta_df = pd.concat(rows, ignore_index=True)
        return meta_df

    def _get_feature_cols(self, X: pd.DataFrame) -> list[str]:
        """Get numeric feature columns (exclude identifiers)."""
        exclude = {"match_idx", "outcome", "FTR"}
        return [c for c in X.columns if c not in exclude and X[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]

    def fit(self, X_meta: pd.DataFrame, y_meta: np.ndarray | pd.Series) -> "MetaModel":
        """Fit the meta-model on stacked meta-features.

        Args:
            X_meta: Meta-feature DataFrame (long format, numeric columns only).
            y_meta: Binary target (1 if outcome won, 0 otherwise).

        Returns:
            self
        """
        self.feature_names = self._get_feature_cols(X_meta)
        X = X_meta[self.feature_names].values

        # Impute NaN (keep all columns even if all-NaN)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X = self.imputer.fit_transform(X)

        y = np.asarray(y_meta, dtype=int)

        if self.learner_type == "logistic_regression":
            lr_cfg = self.config.get("logistic_regression", {})
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.model = LogisticRegression(
                C=lr_cfg.get("C", 1.0),
                max_iter=lr_cfg.get("max_iter", 1000),
                solver=lr_cfg.get("solver", "lbfgs"),
                random_state=lr_cfg.get("random_state", 42),
            )
            self.model.fit(X, y)
        elif self.learner_type == "lightgbm":
            lgbm_cfg = self.config.get("lightgbm", {})
            self.scaler = None
            self.model = lgb.LGBMClassifier(
                n_estimators=lgbm_cfg.get("n_estimators", 200),
                learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                max_depth=lgbm_cfg.get("max_depth", 3),
                num_leaves=lgbm_cfg.get("num_leaves", 15),
                min_child_samples=lgbm_cfg.get("min_child_samples", 30),
                subsample=lgbm_cfg.get("subsample", 0.8),
                colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
                reg_alpha=lgbm_cfg.get("reg_alpha", 0.1),
                reg_lambda=lgbm_cfg.get("reg_lambda", 1.0),
                random_state=lgbm_cfg.get("random_state", 42),
                verbose=lgbm_cfg.get("verbose", -1),
                objective="binary",
                is_unbalance=True,
            )
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown learner type: {self.learner_type}")

        return self

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        """Predict P(profitable) for each (match, outcome) row.

        Args:
            X_meta: Meta-feature DataFrame.

        Returns:
            1-D array of probabilities.
        """
        X = X_meta[self.feature_names].values
        X = self.imputer.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Serialize to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "config": self.config,
                "learner_type": self.learner_type,
                "model_type": "meta_model",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetaModel":
        """Deserialize from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.imputer = data["imputer"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.config = data["config"]
        obj.learner_type = data["learner_type"]
        return obj


def combine_disagree_clv(
    disagree_masks: dict[str, np.ndarray],
    clv_classifiers: dict[str, "CLVClassifier"],
    X_features: pd.DataFrame,
    clv_cutoff: float = 0.3,
) -> dict[str, np.ndarray]:
    """Combine disagreement selection with CLV classifier re-ranking.

    For each outcome: combined = disagree_mask & (clv_prob >= clv_cutoff).
    Disagreement is the broad filter, CLV classifier is the re-ranker.

    Args:
        disagree_masks: Dict outcome -> boolean mask from DisagreementSelector.
        clv_classifiers: Dict outcome -> fitted CLVClassifier.
        X_features: Feature DataFrame for CLV classifier prediction.
        clv_cutoff: Minimum CLV classifier probability to keep a bet.

    Returns:
        Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.
    """
    combined = {}
    for outcome in OUTCOMES:
        d_mask = np.asarray(disagree_masks[outcome], dtype=bool)
        if outcome in clv_classifiers:
            clv_prob = clv_classifiers[outcome].predict_proba(X_features)
            combined[outcome] = d_mask & (clv_prob >= clv_cutoff)
        else:
            combined[outcome] = d_mask
    return combined
