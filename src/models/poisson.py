"""Dixon-Coles Poisson model for football match outcome prediction.

Estimates per-team attack/defense strengths via MLE, with temporal decay
weighting and the Dixon-Coles low-scoring correction (tau factor).

Reference:
    Dixon, M.J. and Coles, S.G. (1997) "Modelling Association Football
    Scores and Inefficiencies in the Football Betting Market"
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Default prior for match outcome when teams are unknown
PRIOR_PROBS = np.array([0.437, 0.265, 0.298])  # [H, D, A]


def _tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts P(x,y) for correlated low scores (0-0, 1-0, 0-1, 1-1).

    Args:
        x: Home goals.
        y: Away goals.
        lambda_: Home scoring rate.
        mu: Away scoring rate.
        rho: Correlation parameter (typically small and negative).

    Returns:
        Multiplicative correction factor.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Dixon-Coles model with temporal decay weighting.

    Parameters are per-team attack/defense strengths, a home advantage term,
    and the rho low-scoring correlation parameter.
    """

    def __init__(self, config: dict | None = None):
        """Initialize from config dict.

        Args:
            config: Optional dict with max_goals, decay_rate, max_iter.
        """
        config = config or {}
        self.max_goals = config.get("max_goals", 7)
        self.decay_rate = config.get("decay_rate", 0.003)
        self.max_iter = config.get("max_iter", 500)

        # Fitted parameters
        self.teams: list[str] | None = None
        self.team_idx: dict[str, int] | None = None
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    def fit(
        self,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> "DixonColesModel":
        """Fit the Dixon-Coles model via maximum likelihood.

        Args:
            home_teams: Array of home team names.
            away_teams: Array of away team names.
            home_goals: Array of home goals scored.
            away_goals: Array of away goals scored.
            dates: Optional datetime array for temporal decay weights.

        Returns:
            self
        """
        home_teams = np.asarray(home_teams)
        away_teams = np.asarray(away_teams)
        home_goals = np.asarray(home_goals, dtype=np.float64)
        away_goals = np.asarray(away_goals, dtype=np.float64)

        # Filter out NaN goals
        valid = np.isfinite(home_goals) & np.isfinite(away_goals)
        home_teams = home_teams[valid]
        away_teams = away_teams[valid]
        home_goals = home_goals[valid].astype(int)
        away_goals = away_goals[valid].astype(int)

        if len(home_teams) == 0:
            logger.warning("No valid matches for Dixon-Coles fitting")
            return self

        # Build team index
        self.teams = sorted(set(home_teams) | set(away_teams))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        # Temporal decay weights
        if dates is not None:
            dates = np.asarray(dates)[valid]
            if hasattr(dates[0], "timestamp"):
                # datetime-like
                max_date = dates.max()
                days_ago = np.array([(max_date - d).days for d in dates], dtype=np.float64)
            else:
                days_ago = np.zeros(len(dates))
            weights = np.exp(-self.decay_rate * days_ago)
        else:
            weights = np.ones(len(home_teams))

        # Map to indices
        home_idx = np.array([self.team_idx[t] for t in home_teams])
        away_idx = np.array([self.team_idx[t] for t in away_teams])

        # Parameter vector: [attack_0..n-1, defense_0..n-1, home_adv, rho]
        # Constraint: sum(attack) = n_teams (or attack[0] pinned)
        n_params = 2 * n_teams + 2

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defense = params[n_teams:2 * n_teams]
            home_adv = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            # Scoring rates
            lambda_ = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] - defense[home_idx])

            # Poisson log-likelihood
            log_lik = (
                poisson.logpmf(home_goals, lambda_)
                + poisson.logpmf(away_goals, mu)
            )

            # Dixon-Coles tau correction (vectorized for low scores)
            tau_vals = np.ones(len(home_goals))
            for hg, ag in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                mask = (home_goals == hg) & (away_goals == ag)
                if mask.any():
                    tau_v = np.array([
                        _tau(hg, ag, l, m, rho)
                        for l, m in zip(lambda_[mask], mu[mask])
                    ])
                    # Clamp tau to avoid log(0)
                    tau_v = np.maximum(tau_v, 1e-10)
                    tau_vals[mask] = tau_v

            log_lik += np.log(tau_vals)

            # Apply weights and sum constraint
            nll = -np.sum(weights * log_lik)

            # Soft constraint: sum(attack) = sum(defense) for identifiability
            nll += 100.0 * (np.sum(attack) - n_teams) ** 2
            nll += 100.0 * (np.sum(defense) - n_teams) ** 2

            return nll

        # Initial parameters
        x0 = np.concatenate([
            np.ones(n_teams),       # attack
            np.ones(n_teams),       # defense
            [0.25],                 # home_adv
            [-0.05],                # rho
        ])

        # Bounds: rho typically in [-0.3, 0.3]
        bounds = (
            [(0.01, 3.0)] * n_teams      # attack
            + [(0.01, 3.0)] * n_teams     # defense
            + [(-0.5, 1.0)]              # home_adv
            + [(-0.5, 0.5)]              # rho
        )

        logger.info("Fitting Dixon-Coles on %d matches, %d teams", len(home_teams), n_teams)
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "disp": False},
        )

        if not result.success:
            logger.warning("Dixon-Coles optimization did not converge: %s", result.message)

        self.attack = result.x[:n_teams]
        self.defense = result.x[n_teams:2 * n_teams]
        self.home_adv = float(result.x[2 * n_teams])
        self.rho = float(result.x[2 * n_teams + 1])

        logger.info(
            "Dixon-Coles fit complete: home_adv=%.3f, rho=%.4f, nll=%.1f",
            self.home_adv, self.rho, result.fun,
        )
        return self

    def _score_matrix(
        self, home_team: str, away_team: str,
    ) -> np.ndarray | None:
        """Compute the (max_goals+1) x (max_goals+1) score probability matrix.

        Returns:
            2-D array where [i,j] = P(home=i, away=j), or None if teams unknown.
        """
        if self.team_idx is None:
            return None

        if home_team not in self.team_idx or away_team not in self.team_idx:
            return None

        h_idx = self.team_idx[home_team]
        a_idx = self.team_idx[away_team]

        lambda_ = np.exp(
            self.attack[h_idx] - self.defense[a_idx] + self.home_adv
        )
        mu = np.exp(
            self.attack[a_idx] - self.defense[h_idx]
        )

        max_g = self.max_goals + 1
        score_matrix = np.zeros((max_g, max_g))

        for i in range(max_g):
            for j in range(max_g):
                p = poisson.pmf(i, lambda_) * poisson.pmf(j, mu)
                tau = _tau(i, j, lambda_, mu, self.rho)
                score_matrix[i, j] = p * max(tau, 0.0)

        # Renormalize to ensure probabilities sum to 1
        total = score_matrix.sum()
        if total > 0:
            score_matrix /= total

        return score_matrix

    def predict_over_under(
        self,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        threshold: float = 2.5,
    ) -> np.ndarray:
        """Predict P(Over) and P(Under) for each match from score matrix.

        Args:
            home_teams: Array of home team names.
            away_teams: Array of away team names.
            threshold: Goals threshold (default 2.5).

        Returns:
            (N, 2) array of [P(Over), P(Under)] probabilities.
        """
        home_teams = np.asarray(home_teams)
        away_teams = np.asarray(away_teams)
        n = len(home_teams)
        probs = np.zeros((n, 2))
        floor_thresh = int(threshold)  # 2 for threshold=2.5

        for i in range(n):
            sm = self._score_matrix(home_teams[i], away_teams[i])
            if sm is None:
                # Prior: roughly 50/50 for O/U 2.5
                probs[i] = [0.50, 0.50]
                continue

            max_g = sm.shape[0]
            p_under = sum(
                sm[h, a]
                for h in range(max_g)
                for a in range(max_g)
                if h + a <= floor_thresh
            )
            p_over = 1.0 - p_under

            if p_over + p_under > 0:
                probs[i] = [p_over, p_under]
            else:
                probs[i] = [0.50, 0.50]

        return probs

    def predict_btts(
        self,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
    ) -> np.ndarray:
        """Predict P(BTTS Yes) and P(BTTS No) for each match from score matrix.

        Args:
            home_teams: Array of home team names.
            away_teams: Array of away team names.

        Returns:
            (N, 2) array of [P(BTTS_Yes), P(BTTS_No)] probabilities.
        """
        home_teams = np.asarray(home_teams)
        away_teams = np.asarray(away_teams)
        n = len(home_teams)
        probs = np.zeros((n, 2))

        for i in range(n):
            sm = self._score_matrix(home_teams[i], away_teams[i])
            if sm is None:
                # Prior: ~55% BTTS Yes
                probs[i] = [0.55, 0.45]
                continue

            max_g = sm.shape[0]
            p_btts_yes = sum(
                sm[h, a]
                for h in range(1, max_g)
                for a in range(1, max_g)
            )
            p_btts_no = 1.0 - p_btts_yes

            if p_btts_yes + p_btts_no > 0:
                probs[i] = [p_btts_yes, p_btts_no]
            else:
                probs[i] = [0.55, 0.45]

        return probs

    def predict_proba(
        self,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
    ) -> np.ndarray:
        """Predict P(H), P(D), P(A) for each match.

        Args:
            home_teams: Array of home team names.
            away_teams: Array of away team names.

        Returns:
            (N, 3) array of [P(H), P(D), P(A)] probabilities.
        """
        home_teams = np.asarray(home_teams)
        away_teams = np.asarray(away_teams)
        n = len(home_teams)
        probs = np.zeros((n, 3))

        for i in range(n):
            sm = self._score_matrix(home_teams[i], away_teams[i])
            if sm is None:
                probs[i] = PRIOR_PROBS
                continue

            max_g = sm.shape[0]
            p_home = sum(sm[h, a] for h in range(max_g) for a in range(h))
            p_draw = sum(sm[h, h] for h in range(max_g))
            p_away = sum(sm[h, a] for h in range(max_g) for a in range(h + 1, max_g))

            total = p_home + p_draw + p_away
            if total > 0:
                probs[i] = [p_home / total, p_draw / total, p_away / total]
            else:
                probs[i] = PRIOR_PROBS

        return probs

    def save(self, path: str | Path) -> None:
        """Serialize to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "teams": self.teams,
                "team_idx": self.team_idx,
                "attack": self.attack,
                "defense": self.defense,
                "home_adv": self.home_adv,
                "rho": self.rho,
                "max_goals": self.max_goals,
                "decay_rate": self.decay_rate,
                "model_type": "dixon_coles",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DixonColesModel":
        """Deserialize from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.teams = data["teams"]
        obj.team_idx = data["team_idx"]
        obj.attack = data["attack"]
        obj.defense = data["defense"]
        obj.home_adv = data["home_adv"]
        obj.rho = data["rho"]
        obj.max_goals = data["max_goals"]
        obj.decay_rate = data["decay_rate"]
        obj.max_iter = 500
        return obj
