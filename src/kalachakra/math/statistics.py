"""
Domain 1: Mathematical Foundations — Statistics

Research References:
    - Fisher, "On the Mathematical Foundations of Theoretical Statistics" (1922)
    - Neyman & Pearson, "On the Problem of the Most Efficient Tests of
      Statistical Hypotheses" (1933)
    - Efron, "Bootstrap Methods: Another Look at the Jackknife" (1979),
      Annals of Statistics 7(1), 1-26
    - Wilks, "The Large-Sample Distribution of the Likelihood Ratio" (1938)

Provides hypothesis testing, estimation, and validation methods
essential for scientific rigor in Project Kalachakra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class HypothesisTestResult:
    """Result of a hypothesis test.

    Includes effect size and confidence interval for rigorous reporting.
    """

    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    n_samples: int = 0
    interpretation: str = ""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Result of bootstrap estimation."""

    estimate: float
    se: float  # Standard error
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    distribution: NDArray[np.floating] | None = None


# =============================================================================
# Maximum Likelihood Estimation
# =============================================================================


def mle_gaussian(data: NDArray[np.floating]) -> tuple[float, float]:
    """Maximum Likelihood Estimation for Gaussian parameters.

    Reference: Fisher (1922)

    MLE for μ and σ²:
        μ̂_ML = (1/N) Σᵢ xᵢ
        σ̂²_ML = (1/N) Σᵢ (xᵢ - μ̂)²

    Note: σ̂²_ML is biased. Unbiased estimate uses N-1 (Bessel's correction):
        σ̂²_unbiased = (1/(N-1)) Σᵢ (xᵢ - μ̂)²

    Returns:
        Tuple of (mu_hat, sigma_hat_unbiased).
    """
    data = np.asarray(data, dtype=np.float64)
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))  # Bessel's correction
    return mu, sigma


# =============================================================================
# Hypothesis Tests
# =============================================================================


def t_test_independent(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> HypothesisTestResult:
    """Welch's two-sample t-test (unequal variances).

    Reference: Welch, "The Generalization of 'Student's' Problem" (1947)

    Hypotheses:
        H₀: μ₁ = μ₂
        H₁: μ₁ ≠ μ₂ (or < or >)

    Test statistic:
        t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

    Degrees of freedom (Welch-Satterthwaite):
        ν = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]

    Effect size (Cohen's d):
        d = (x̄₁ - x̄₂) / s_pooled
        s_pooled = √[(s₁² + s₂²) / 2]

    Args:
        x: First sample.
        y: Second sample.
        alpha: Significance level.
        alternative: Direction of alternative hypothesis.

    Returns:
        HypothesisTestResult with statistic, p-value, effect size, CI.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    stat, p_val = sp_stats.ttest_ind(x, y, equal_var=False, alternative=alternative)

    # Cohen's d effect size
    s_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    cohens_d = (np.mean(x) - np.mean(y)) / s_pooled if s_pooled > 0 else 0.0

    # 95% CI for the difference in means
    diff = np.mean(x) - np.mean(y)
    se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
    nu = _welch_df(x, y)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, nu)
    ci = (diff - t_crit * se, diff + t_crit * se)

    return HypothesisTestResult(
        test_name="welch_t_test",
        statistic=float(stat),
        p_value=float(p_val),
        reject_null=float(p_val) < alpha,
        alpha=alpha,
        effect_size=float(cohens_d),
        confidence_interval=ci,
        n_samples=len(x) + len(y),
        interpretation=_interpret_cohens_d(float(cohens_d)),
    )


def _welch_df(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Welch-Satterthwaite degrees of freedom."""
    s1_sq, s2_sq = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    return num / den if den > 0 else 1.0


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Paired samples t-test.

    Reference: Student (Gosset), "The Probable Error of a Mean" (1908)

    For paired observations (xᵢ, yᵢ), test whether mean difference = 0:
        dᵢ = xᵢ - yᵢ
        t = d̄ / (s_d / √n)

    Args:
        x: First paired measurements.
        y: Second paired measurements.
        alpha: Significance level.

    Returns:
        HypothesisTestResult.
    """
    x, y = np.asarray(x), np.asarray(y)
    stat, p_val = sp_stats.ttest_rel(x, y)
    d = x - y
    effect = float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else 0.0

    return HypothesisTestResult(
        test_name="paired_t_test",
        statistic=float(stat),
        p_value=float(p_val),
        reject_null=float(p_val) < alpha,
        alpha=alpha,
        effect_size=effect,
        n_samples=len(x),
    )


def mann_whitney_u(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Mann-Whitney U test (non-parametric alternative to t-test).

    Reference: Mann & Whitney (1947)

    Tests whether two independent samples come from the same distribution.
    No normality assumption required.

    Effect size (rank-biserial correlation):
        r = 1 - 2U/(n₁·n₂)

    Args:
        x: First sample.
        y: Second sample.
        alpha: Significance level.

    Returns:
        HypothesisTestResult.
    """
    x, y = np.asarray(x), np.asarray(y)
    stat, p_val = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    r = 1 - 2 * stat / (n1 * n2)

    return HypothesisTestResult(
        test_name="mann_whitney_u",
        statistic=float(stat),
        p_value=float(p_val),
        reject_null=float(p_val) < alpha,
        alpha=alpha,
        effect_size=float(r),
        n_samples=n1 + n2,
    )


def permutation_test(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    statistic_fn: callable = lambda a, b: float(np.mean(a) - np.mean(b)),
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> HypothesisTestResult:
    """Exact permutation test.

    Reference: Fisher, "The Design of Experiments" (1935)
               Good, "Permutation, Parametric, and Bootstrap Tests" (2005)

    Algorithm:
        1. Compute observed test statistic T_obs
        2. For b = 1, ..., B:
            a. Randomly permute combined data
            b. Split into two groups
            c. Compute test statistic T_b
        3. p-value = (#{T_b ≥ |T_obs|} + 1) / (B + 1)

    No distributional assumptions required. The gold standard for
    testing whether two groups differ.

    Args:
        x: First sample.
        y: Second sample.
        statistic_fn: Function computing test statistic from two samples.
        n_permutations: Number of random permutations.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        HypothesisTestResult.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)

    observed_stat = statistic_fn(x, y)
    combined = np.concatenate([x, y])
    n1 = len(x)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic_fn(combined[:n1], combined[n1:])
        if abs(perm_stat) >= abs(observed_stat):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return HypothesisTestResult(
        test_name="permutation_test",
        statistic=float(observed_stat),
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
        n_samples=len(combined),
    )


# =============================================================================
# Multiple Testing Correction
# =============================================================================


def bonferroni_correction(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Bonferroni correction for multiple hypothesis testing.

    Reference: Bonferroni (1936)

    Formula:
        p_adjusted_i = min(m · pᵢ, 1)
        Reject H₀ᵢ if p_adjusted_i < α

    Controls Family-Wise Error Rate (FWER) ≤ α.
    Conservative — may have low power with many tests.

    Args:
        p_values: Array of raw p-values.
        alpha: Desired FWER.

    Returns:
        Tuple of (adjusted p-values, rejection decisions).
    """
    p = np.asarray(p_values)
    m = len(p)
    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted < alpha
    return adjusted, rejected


def benjamini_hochberg(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Benjamini-Hochberg procedure for FDR control.

    Reference: Benjamini & Hochberg, "Controlling the False Discovery Rate"
               (1995), J. Royal Stat. Soc. B 57(1), 289-300

    Algorithm:
        1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
        2. Find largest k such that p_(k) ≤ (k/m) · α
        3. Reject H₀_(1), ..., H₀_(k)

    Controls False Discovery Rate (FDR) ≤ α.
    Less conservative than Bonferroni — more power for many tests.

    Application in Kalachakra:
        When testing 50+ astrological features for significance,
        BH-FDR correction prevents false discoveries while maintaining
        reasonable statistical power.

    Args:
        p_values: Array of raw p-values.
        alpha: Desired FDR level.

    Returns:
        Tuple of (adjusted p-values, rejection decisions).
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)

    # Adjusted p-values (step-up)
    adjusted = np.zeros(m)
    adjusted[order] = np.minimum.accumulate(
        (p[order] * m / ranks)[::-1]
    )[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    rejected = adjusted < alpha
    return adjusted, rejected


# =============================================================================
# Bootstrap
# =============================================================================


def bootstrap(
    data: NDArray[np.floating],
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int = 42,
) -> BootstrapResult:
    """Non-parametric bootstrap estimation.

    Reference: Efron, "Bootstrap Methods: Another Look at the Jackknife"
               (1979), Annals of Statistics 7(1), 1-26
               Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)

    Algorithm:
        1. Compute θ̂ = statistic(data)
        2. For b = 1, ..., B:
            a. Sample n points with replacement from data
            b. Compute θ̂*_b = statistic(bootstrap_sample)
        3. Standard error: SE = std({θ̂*_b})
        4. Confidence interval:
            Percentile: [θ̂*_(α/2), θ̂*_(1-α/2)]
            BCa: Bias-corrected and accelerated (more accurate)

    Args:
        data: Original data array.
        statistic_fn: Function to compute the statistic of interest.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level for interval.
        method: "percentile" or "bca" (bias-corrected accelerated).
        seed: Random seed.

    Returns:
        BootstrapResult with estimate, SE, and confidence interval.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    # Original estimate
    theta_hat = float(statistic_fn(data))

    # Bootstrap resamples
    boot_stats = np.array([
        statistic_fn(data[rng.integers(0, n, n)])
        for _ in range(n_bootstrap)
    ])

    se = float(np.std(boot_stats, ddof=1))
    alpha_half = (1 - ci_level) / 2

    if method == "percentile":
        ci_lower = float(np.percentile(boot_stats, 100 * alpha_half))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha_half)))
    elif method == "bca":
        # BCa correction
        # Bias correction factor z₀
        z0 = sp_stats.norm.ppf(np.mean(boot_stats < theta_hat))

        # Acceleration factor (jackknife)
        jackknife_stats = np.array([
            statistic_fn(np.delete(data, i))
            for i in range(n)
        ])
        jack_mean = jackknife_stats.mean()
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a = num / den if den > 0 else 0.0

        # Adjusted quantiles
        z_alpha = sp_stats.norm.ppf(alpha_half)
        z_1alpha = sp_stats.norm.ppf(1 - alpha_half)

        q_lower = sp_stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        q_upper = sp_stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        ci_lower = float(np.percentile(boot_stats, 100 * q_lower))
        ci_upper = float(np.percentile(boot_stats, 100 * q_upper))
    else:
        raise ValueError(f"Unknown method: {method}")

    return BootstrapResult(
        estimate=theta_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        distribution=boot_stats,
    )


# =============================================================================
# Effect Size Measures
# =============================================================================


def cohens_d(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Cohen's d effect size for two independent samples.

    Formula:
        d = (x̄ - ȳ) / s_pooled
        s_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]

    Interpretation (Cohen, 1988):
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    x, y = np.asarray(x), np.asarray(y)
    n1, n2 = len(x), len(y)
    s_pooled = np.sqrt(
        ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1))
        / (n1 + n2 - 2)
    )
    return float((np.mean(x) - np.mean(y)) / s_pooled) if s_pooled > 0 else 0.0


def cramers_v(contingency_table: NDArray[np.floating]) -> float:
    """Cramér's V effect size for chi-squared test.

    Formula:
        V = √(χ² / (n · min(r-1, c-1)))

    Interpretation:
        V < 0.1: negligible
        0.1 ≤ V < 0.3: small
        0.3 ≤ V < 0.5: medium
        V ≥ 0.5: large
    """
    table = np.asarray(contingency_table)
    chi2 = sp_stats.chi2_contingency(table)[0]
    n = table.sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
