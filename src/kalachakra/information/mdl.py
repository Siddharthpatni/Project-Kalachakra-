"""
Domain 25 — Minimum Description Length / information criteria.

Model-selection scores that trade goodness-of-fit against complexity, so a
model is only preferred when the bits it saves in describing the data exceed
the bits needed to describe its extra parameters. Guards the pipeline against
rewarding complexity for its own sake — central to the project's discipline of
not over-fitting astrological features.
"""

from __future__ import annotations

import numpy as np


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion (lower is better).

    Args:
        log_likelihood: Maximized log-likelihood of the model.
        n_params: Number of free parameters.

    Returns:
        ``2k − 2 ln L``.
    """
    return 2.0 * n_params - 2.0 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion (lower is better).

    Penalizes parameters more strongly than AIC as the sample grows.

    Args:
        log_likelihood: Maximized log-likelihood.
        n_params: Number of free parameters.
        n_samples: Number of observations.

    Returns:
        ``k ln n − 2 ln L``.
    """
    return n_params * np.log(n_samples) - 2.0 * log_likelihood


def two_part_mdl(residual_bits: float, model_bits: float) -> float:
    """Two-part MDL code length: bits to describe the model plus the data given it.

    Args:
        residual_bits: Bits to encode the data given the model (the fit term).
        model_bits: Bits to encode the model itself (the complexity term).

    Returns:
        Total description length; the model minimizing it is preferred.
    """
    return residual_bits + model_bits


def gaussian_log_likelihood(residuals: np.ndarray) -> float:
    """Maximized Gaussian log-likelihood of a residual vector.

    Convenience for feeding regression residuals into :func:`aic` / :func:`bic`.

    Args:
        residuals: Model residuals (observed − predicted).

    Returns:
        Log-likelihood under a fitted zero-mean Gaussian noise model.
    """
    n = residuals.size
    if n == 0:
        raise ValueError("residuals must be non-empty")
    variance = float(np.mean(residuals**2))
    if variance <= 0.0:
        return float("inf")
    return float(-0.5 * n * (np.log(2.0 * np.pi * variance) + 1.0))
