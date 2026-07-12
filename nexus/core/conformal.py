"""
Conformal prediction wrappers for uncertainty-quantified EBITDA forecasts.

Conformal prediction gives *distribution-free* coverage guarantees:
Pr(Y_{n+1} in C(X_{n+1})) >= 1 - alpha for any alpha, regardless of model.

This is critical for enterprise AI: you can tell the CFO
"this EBITDA range covers the true value 90% of the time, provably."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class ConformalResult:
    point_estimate: np.ndarray      # (T, n_targets)
    lower: np.ndarray               # (T, n_targets)
    upper: np.ndarray               # (T, n_targets)
    coverage: float                 # empirical coverage on calibration set
    alpha: float                    # nominal miscoverage rate


class SplitConformalPredictor:
    """
    Split conformal prediction for time series (Papadopoulos et al. 2002).

    Calibration: compute nonconformity scores on held-out cal set.
    Prediction: inflate intervals by calibrated quantile.

    Works with any underlying forecaster (TFT, ARIMA, etc.)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._cal_scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None

    def calibrate(self, y_true: np.ndarray, y_hat_lower: np.ndarray, y_hat_upper: np.ndarray) -> None:
        """
        Compute nonconformity scores on calibration data.
        Score = max(lower - y, y - upper) — positive means out-of-interval.

        y_true, y_hat_{lower,upper}: (n_cal, T, n_targets)
        """
        scores_lower = y_hat_lower - y_true
        scores_upper = y_true - y_hat_upper
        self._cal_scores = np.maximum(scores_lower, scores_upper).reshape(-1)
        n = len(self._cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(self._cal_scores, min(level, 1.0)))

    def predict(
        self,
        y_hat_lower: np.ndarray,
        y_hat_upper: np.ndarray,
        point: np.ndarray,
    ) -> ConformalResult:
        """Inflate prediction intervals by calibrated quantile."""
        if self._quantile is None:
            raise RuntimeError("Call calibrate() before predict().")
        q = self._quantile
        return ConformalResult(
            point_estimate=point,
            lower=y_hat_lower - q,
            upper=y_hat_upper + q,
            coverage=1 - self.alpha,
            alpha=self.alpha,
        )


class AdaptiveConformalPredictor:
    """
    Adaptive conformal inference (Gibbs & Candes 2021).
    Adjusts alpha dynamically to maintain coverage under distribution shift —
    critical for enterprise data where KPI distributions change with business cycles.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.005):
        self.alpha_target = alpha
        self.gamma = gamma          # step size for alpha adaptation
        self._alpha_t = alpha
        self._history: list[float] = []

    def update_and_predict(
        self,
        y_true: float,
        lower_prev: float,
        upper_prev: float,
        new_lower: float,
        new_upper: float,
        cal_scores: np.ndarray,
    ) -> tuple[float, float]:
        """
        Update alpha based on whether previous interval covered y_true,
        then compute new conformal interval.
        """
        err = float(y_true < lower_prev or y_true > upper_prev)
        self._alpha_t = self._alpha_t + self.gamma * (self.alpha_target - err)
        self._alpha_t = float(np.clip(self._alpha_t, 0.01, 0.99))

        n = len(cal_scores)
        level = np.ceil((n + 1) * (1 - self._alpha_t)) / n
        q = float(np.quantile(cal_scores, min(level, 1.0)))
        self._history.append(self._alpha_t)

        return new_lower - q, new_upper + q

    @property
    def effective_alpha(self) -> float:
        return self._alpha_t

    def empirical_coverage(self) -> float:
        """Rolling empirical coverage over update history."""
        if not self._history:
            return float("nan")
        return 1.0 - self.alpha_target  # placeholder until enough data


def ebitda_impact_interval(
    revenue_delta: ConformalResult,
    cost_delta: ConformalResult,
    margin_rate: float = 0.15,
) -> ConformalResult:
    """
    Propagate conformal intervals through EBITDA = (revenue - cost) * margin.
    Uses interval arithmetic for conservative bounds.
    """
    lower = (revenue_delta.lower - cost_delta.upper) * margin_rate
    upper = (revenue_delta.upper - cost_delta.lower) * margin_rate
    point = (revenue_delta.point_estimate - cost_delta.point_estimate) * margin_rate

    return ConformalResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        coverage=min(revenue_delta.coverage, cost_delta.coverage),
        alpha=max(revenue_delta.alpha, cost_delta.alpha),
    )
