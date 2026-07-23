"""
Domain 1/25: Mathematical Foundations — Information Theory

Research References:
    - Shannon, "A Mathematical Theory of Communication" (1948), Bell System
      Technical Journal 27, 379–423 & 623–656
    - Cover & Thomas, "Elements of Information Theory" (2nd ed., 2006)
    - Kullback & Leibler, "On Information and Sufficiency" (1951),
      Ann. Math. Stat. 22(1), 79-86
    - Lin, "Divergence measures based on the Shannon entropy" (1991),
      IEEE Trans. Inf. Theory 37(1), 145-151
    - Rissanen, "Modeling by shortest data description" (1978), Automatica 14

This module is CRITICAL for Project Kalachakra. Before training any model,
use mutual information to measure whether each astrological feature carries
statistically significant information about target events. If MI ≈ 0,
the feature is noise regardless of model complexity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Entropy
# =============================================================================


def shannon_entropy(
    p: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Shannon entropy of a discrete probability distribution.

    Reference: Shannon (1948), Eq. 1

    Formula:
        H(X) = -Σᵢ p(xᵢ) · log_b(p(xᵢ))

    Properties:
        - H(X) ≥ 0 (non-negativity)
        - H(X) = 0 iff X is deterministic
        - H(X) ≤ log_b(|X|) (maximum when X is uniform)
        - H(X) is concave in p

    Interpretation:
        Average number of bits (base 2) or nats (base e) needed
        to encode samples from the distribution.

    Args:
        p: Probability distribution (must sum to 1).
        base: Logarithm base. 2 = bits, e = nats, 10 = hartleys.

    Returns:
        Shannon entropy.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]  # Avoid log(0)
    return float(-np.sum(p * np.log(p) / np.log(base)))


def joint_entropy(
    joint_p: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Joint entropy of two random variables.

    Reference: Cover & Thomas (2006), Theorem 2.6.4

    Formula:
        H(X,Y) = -Σᵢ Σⱼ p(xᵢ,yⱼ) · log(p(xᵢ,yⱼ))

    Properties:
        - H(X,Y) ≤ H(X) + H(Y) (equality iff X,Y independent)
        - H(X,Y) ≥ max(H(X), H(Y))

    Args:
        joint_p: Joint probability distribution matrix (|X| × |Y|).
        base: Logarithm base.

    Returns:
        Joint entropy.
    """
    joint_p = np.asarray(joint_p, dtype=np.float64)
    flat = joint_p.flatten()
    flat = flat[flat > 0]
    return float(-np.sum(flat * np.log(flat) / np.log(base)))


def conditional_entropy(
    joint_p: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Conditional entropy H(Y|X).

    Reference: Cover & Thomas (2006), Section 2.2

    Formula:
        H(Y|X) = H(X,Y) - H(X)
               = -Σᵢ Σⱼ p(xᵢ,yⱼ) · log(p(yⱼ|xᵢ))

    Chain rule:
        H(X,Y) = H(X) + H(Y|X)

    Args:
        joint_p: Joint probability distribution matrix (|X| × |Y|).
        base: Logarithm base.

    Returns:
        Conditional entropy H(Y|X).
    """
    joint_p = np.asarray(joint_p, dtype=np.float64)
    p_x = joint_p.sum(axis=1)
    h_xy = joint_entropy(joint_p, base)
    h_x = shannon_entropy(p_x, base)
    return h_xy - h_x


def cross_entropy(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
    base: float = np.e,
) -> float:
    """Cross entropy between distributions p and q.

    Reference: Cover & Thomas (2006), Section 2.3

    Formula:
        H(p, q) = -Σᵢ p(xᵢ) · log(q(xᵢ))

    Properties:
        - H(p, q) ≥ H(p) (Gibbs' inequality)
        - H(p, q) = H(p) + D_KL(p ‖ q)
        - Used as loss function in neural network classification

    Args:
        p: True distribution.
        q: Model distribution.
        base: Logarithm base.

    Returns:
        Cross entropy.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-15, 1.0)
    return float(-np.sum(p * np.log(q) / np.log(base)))


# =============================================================================
# Divergences
# =============================================================================


def kl_divergence(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
    base: float = np.e,
) -> float:
    """Kullback-Leibler divergence D_KL(P ‖ Q).

    Reference: Kullback & Leibler (1951), "On Information and Sufficiency"

    Formula:
        D_KL(P ‖ Q) = Σᵢ p(xᵢ) · log(p(xᵢ)/q(xᵢ))
                     = H(p, q) - H(p)

    Properties:
        - D_KL(P ‖ Q) ≥ 0 (Gibbs' inequality / information inequality)
        - D_KL(P ‖ Q) = 0 iff P = Q
        - NOT symmetric: D_KL(P‖Q) ≠ D_KL(Q‖P)
        - NOT a metric (violates triangle inequality)

    Interpretation:
        Expected extra number of bits/nats needed to encode data from P
        using a code optimized for Q instead of P.

    Args:
        p: True distribution P.
        q: Approximating distribution Q.
        base: Logarithm base. e = nats.

    Returns:
        KL divergence (always ≥ 0).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    q_safe = np.clip(q[mask], 1e-15, None)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe) / np.log(base)))


def jensen_shannon_divergence(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Jensen-Shannon divergence JSD(P ‖ Q).

    Reference: Lin (1991), "Divergence measures based on the Shannon entropy"

    Formula:
        M = ½(P + Q)
        JSD(P ‖ Q) = ½ D_KL(P ‖ M) + ½ D_KL(Q ‖ M)

    Properties:
        - 0 ≤ JSD ≤ log(2) (when base = 2, max = 1 bit)
        - SYMMETRIC: JSD(P ‖ Q) = JSD(Q ‖ P)
        - √JSD is a proper metric (satisfies triangle inequality)
        - Always finite (unlike KL divergence)

    Args:
        p: First distribution.
        q: Second distribution.
        base: Logarithm base.

    Returns:
        Jensen-Shannon divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


# =============================================================================
# Mutual Information
# =============================================================================


def mutual_information(
    joint_p: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Mutual information I(X; Y) from joint distribution.

    Reference: Cover & Thomas (2006), Section 2.4

    Formula:
        I(X;Y) = Σᵢ Σⱼ p(xᵢ,yⱼ) · log(p(xᵢ,yⱼ) / (p(xᵢ)p(yⱼ)))
               = H(X) + H(Y) - H(X,Y)
               = H(X) - H(X|Y)
               = H(Y) - H(Y|X)
               = D_KL(p(x,y) ‖ p(x)p(y))

    Properties:
        - I(X;Y) ≥ 0
        - I(X;Y) = 0 iff X, Y are independent
        - I(X;Y) = I(Y;X) (symmetric)
        - I(X;Y) ≤ min(H(X), H(Y))

    Application in Kalachakra:
        CRITICAL TEST — Measure whether each planetary feature
        (Nakshatra, Dasha state, aspect pattern, Yoga) carries any
        statistically significant information about target events.
        If I(feature; event) ≈ 0, the feature is noise.

    Args:
        joint_p: Joint probability distribution (|X| × |Y|).
        base: Logarithm base.

    Returns:
        Mutual information I(X;Y).
    """
    joint_p = np.asarray(joint_p, dtype=np.float64)
    p_x = joint_p.sum(axis=1)
    p_y = joint_p.sum(axis=0)
    h_x = shannon_entropy(p_x, base)
    h_y = shannon_entropy(p_y, base)
    h_xy = joint_entropy(joint_p, base)
    return h_x + h_y - h_xy


def normalized_mutual_information(
    joint_p: NDArray[np.floating],
    method: str = "arithmetic",
) -> float:
    """Normalized Mutual Information (NMI).

    Formula:
        NMI_arithmetic(X;Y) = 2 · I(X;Y) / (H(X) + H(Y))
        NMI_geometric(X;Y) = I(X;Y) / √(H(X) · H(Y))
        NMI_max(X;Y) = I(X;Y) / max(H(X), H(Y))
        NMI_min(X;Y) = I(X;Y) / min(H(X), H(Y))

    NMI ∈ [0, 1] where 1 = perfect correlation, 0 = independence.

    Args:
        joint_p: Joint probability distribution.
        method: Normalization method ("arithmetic", "geometric", "max", "min").

    Returns:
        Normalized mutual information ∈ [0, 1].
    """
    joint_p = np.asarray(joint_p, dtype=np.float64)
    mi = mutual_information(joint_p)
    p_x = joint_p.sum(axis=1)
    p_y = joint_p.sum(axis=0)
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)

    if h_x == 0 and h_y == 0:
        return 1.0

    if method == "arithmetic":
        return 2 * mi / (h_x + h_y + 1e-15)
    elif method == "geometric":
        return mi / (np.sqrt(h_x * h_y) + 1e-15)
    elif method == "max":
        return mi / (max(h_x, h_y) + 1e-15)
    elif method == "min":
        return mi / (min(h_x, h_y) + 1e-15)
    else:
        raise ValueError(f"Unknown method: {method}")


def information_gain(
    parent_p: NDArray[np.floating],
    children_ps: list[NDArray[np.floating]],
    children_weights: NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """Information gain (reduction in entropy from a split).

    Reference: Quinlan, "Induction of Decision Trees" (1986)

    Formula:
        IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) · H(Sᵥ)

    where S is the parent set, A is the split attribute,
    and Sᵥ are the child sets after splitting on A.

    Used in decision trees (ID3, C4.5, CART) for feature selection.

    Args:
        parent_p: Distribution of parent set.
        children_ps: List of distributions for each child.
        children_weights: Weight (proportion) of each child.
        base: Logarithm base.

    Returns:
        Information gain.
    """
    h_parent = shannon_entropy(parent_p, base)
    h_children = sum(
        w * shannon_entropy(cp, base)
        for w, cp in zip(children_weights, children_ps)
    )
    return h_parent - h_children


# =============================================================================
# Continuous MI Estimation
# =============================================================================


def mutual_information_knn(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    k: int = 3,
) -> float:
    """Estimate mutual information for continuous variables using KNN.

    Reference: Kraskov, Stögbauer & Grassberger (KSG estimator),
               "Estimating Mutual Information" (2004),
               Physical Review E 69, 066138

    Algorithm (KSG Estimator 1):
        For each point i:
            1. Find k-th nearest neighbor distance ε(i) in joint space (X,Y)
            2. Count n_x(i) = #{j : |xᵢ - xⱼ| < ε(i)}
            3. Count n_y(i) = #{j : |yᵢ - yⱼ| < ε(i)}

        I(X;Y) ≈ ψ(k) - 〈ψ(n_x + 1) + ψ(n_y + 1)〉 + ψ(N)

    where ψ is the digamma function.

    This is a non-parametric estimator that doesn't require binning,
    making it suitable for continuous planetary position data.

    Args:
        X: Continuous variable, shape (N,) or (N, D_x).
        Y: Continuous variable, shape (N,) or (N, D_y).
        k: Number of nearest neighbors.

    Returns:
        Estimated mutual information in nats.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64).T).T
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64).T).T

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    N = X.shape[0]
    assert N == Y.shape[0], "X and Y must have same number of samples"

    # Joint space
    XY = np.hstack([X, Y])

    # For each point, find k-th nearest neighbor distance in Chebyshev metric
    from scipy.spatial import KDTree

    tree_xy = KDTree(XY)
    tree_x = KDTree(X)
    tree_y = KDTree(Y)

    # Query k+1 neighbors (including self)
    dists_xy, _ = tree_xy.query(XY, k=k + 1, p=np.inf)
    eps = dists_xy[:, -1]  # k-th neighbor distance

    # Count neighbors within eps in marginal spaces
    n_x = np.array([
        len(tree_x.query_ball_point(X[i], eps[i], p=np.inf)) - 1
        for i in range(N)
    ])
    n_y = np.array([
        len(tree_y.query_ball_point(Y[i], eps[i], p=np.inf)) - 1
        for i in range(N)
    ])

    # KSG estimator
    mi = float(
        digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
    )

    return max(0.0, mi)  # MI is non-negative


# =============================================================================
# Fisher Information
# =============================================================================


def fisher_information(
    log_likelihood_grad: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Estimate Fisher Information Matrix from gradient samples.

    Reference: Cover & Thomas (2006), Section 12.1

    Formula:
        I(θ) = E[∇log p(x|θ) · ∇log p(x|θ)ᵀ]
             ≈ (1/N) Σᵢ ∇log p(xᵢ|θ) · ∇log p(xᵢ|θ)ᵀ

    Properties:
        - I(θ) is positive semi-definite
        - Cramér-Rao bound: Var(θ̂) ≥ I(θ)⁻¹
        - Measures curvature of log-likelihood at θ
        - Large I(θ) → parameter well-determined by data

    Application in Kalachakra:
        Measure how sensitive predictions are to each model parameter.
        High Fisher information for a parameter = that parameter is
        well-constrained by the data.

    Args:
        log_likelihood_grad: Matrix of score function values,
            shape (n_samples, n_params). Each row is ∇log p(xᵢ|θ).

    Returns:
        Fisher Information Matrix of shape (n_params, n_params).
    """
    grads = np.asarray(log_likelihood_grad, dtype=np.float64)
    N = grads.shape[0]
    return (grads.T @ grads) / N


# =============================================================================
# Minimum Description Length
# =============================================================================


def mdl_score(
    log_likelihood: float,
    n_params: int,
    n_samples: int,
) -> float:
    """Minimum Description Length (two-part MDL) score.

    Reference: Rissanen, "Modeling by shortest data description" (1978)
               Grünwald, "The Minimum Description Length Principle" (2007)

    Formula:
        MDL = -log L(θ̂|data) + (k/2) · log(n)

    where:
        L(θ̂|data) = maximized likelihood
        k = number of free parameters
        n = number of data points

    This is equivalent to the Bayesian Information Criterion (BIC).

    Interpretation:
        Total description length = code length for data given model
                                  + code length for the model itself.
        The best model minimizes the total description length.

    Application in Kalachakra:
        Compare models of different complexity. A 50-parameter Vedic
        feature model must beat a 5-parameter baseline by enough to
        justify the extra complexity. MDL prevents overfitting.

    Args:
        log_likelihood: Maximized log-likelihood.
        n_params: Number of free parameters k.
        n_samples: Number of data points n.

    Returns:
        MDL score (lower is better).
    """
    return -log_likelihood + 0.5 * n_params * np.log(n_samples)
