"""
Domain 1: Mathematical Foundations — Probability Theory

Research References:
    - Papoulis & Pillai, "Probability, Random Variables and Stochastic
      Processes" (4th ed.)
    - Bishop, "Pattern Recognition and Machine Learning" (2006), Ch. 2, 8, 13
    - Murphy, "Machine Learning: A Probabilistic Perspective" (2012)
    - Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications
      in Speech Recognition" (1989), Proceedings of the IEEE 77(2)

Provides probability distributions, sampling, Markov chains, HMMs,
and mixture models used for Bayesian inference throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.special import logsumexp

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Distribution Wrappers (type-safe, with exact formulas)
# =============================================================================


@dataclass(frozen=True, slots=True)
class GaussianDistribution:
    """Univariate Gaussian (Normal) distribution.

    Reference: Bishop PRML, Section 2.3

    PDF:
        p(x|μ,σ²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))

    Maximum Likelihood Estimates:
        μ_ML = (1/N) Σᵢ xᵢ
        σ²_ML = (1/N) Σᵢ (xᵢ - μ_ML)²

    Entropy:
        H[X] = ½ ln(2πeσ²)
    """

    mu: float = 0.0
    sigma: float = 1.0

    def pdf(self, x: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Probability density function."""
        x = np.asarray(x)
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - self.mu) / self.sigma) ** 2
        )

    def log_pdf(self, x: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Log probability density function."""
        x = np.asarray(x)
        return (
            -0.5 * np.log(2 * np.pi)
            - np.log(self.sigma)
            - 0.5 * ((x - self.mu) / self.sigma) ** 2
        )

    def sample(self, n: int, seed: int = 42) -> NDArray[np.floating]:
        """Generate n samples."""
        rng = np.random.default_rng(seed)
        return rng.normal(self.mu, self.sigma, n)

    def entropy(self) -> float:
        """Differential entropy H[X] = ½ ln(2πeσ²)."""
        return 0.5 * np.log(2 * np.pi * np.e * self.sigma**2)

    @staticmethod
    def fit(data: NDArray[np.floating]) -> GaussianDistribution:
        """Maximum likelihood estimation."""
        return GaussianDistribution(mu=float(np.mean(data)), sigma=float(np.std(data)))


@dataclass(frozen=True, slots=True)
class MultivariateGaussian:
    """Multivariate Gaussian distribution.

    Reference: Bishop PRML, Section 2.3

    PDF:
        p(x|μ,Σ) = (2π)^(-D/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀ Σ⁻¹(x-μ))

    Mahalanobis distance:
        Δ² = (x-μ)ᵀ Σ⁻¹ (x-μ)

    Application in Kalachakra:
        Model joint distributions of planetary positions.
        CMA-ES samples from this distribution.
    """

    mu: NDArray[np.floating]
    sigma: NDArray[np.floating]  # Covariance matrix Σ

    def pdf(self, x: NDArray[np.floating]) -> float:
        """Multivariate Gaussian PDF."""
        return float(sp_stats.multivariate_normal.pdf(x, mean=self.mu, cov=self.sigma))

    def log_pdf(self, x: NDArray[np.floating]) -> float:
        """Log PDF."""
        return float(sp_stats.multivariate_normal.logpdf(x, mean=self.mu, cov=self.sigma))

    def sample(self, n: int, seed: int = 42) -> NDArray[np.floating]:
        """Sample n points from the distribution."""
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mu, self.sigma, n)

    def mahalanobis_distance(self, x: NDArray[np.floating]) -> float:
        """Mahalanobis distance: Δ = √((x-μ)ᵀ Σ⁻¹ (x-μ))."""
        diff = x - self.mu
        inv_sigma = np.linalg.inv(self.sigma)
        return float(np.sqrt(diff @ inv_sigma @ diff))


# =============================================================================
# Markov Chains
# =============================================================================


@dataclass
class MarkovChain:
    """Discrete-time Markov chain.

    Reference: Papoulis, Ch. 15

    Markov property:
        P(Xₜ₊₁|Xₜ, Xₜ₋₁, ..., X₀) = P(Xₜ₊₁|Xₜ)

    Transition matrix P:
        P_ij = P(Xₜ₊₁ = j | Xₜ = i)
        Each row sums to 1: Σⱼ P_ij = 1

    Chapman-Kolmogorov equation:
        P^(n) = Pⁿ  (n-step transition matrix)

    Stationary distribution π:
        πP = π  (left eigenvector with eigenvalue 1)
        Σᵢ πᵢ = 1

    Application in Kalachakra:
        Model Dasha state transitions — the Vimshottari Dasha system
        can be represented as a Markov chain over planetary period states.
    """

    transition_matrix: NDArray[np.floating]
    state_names: list[str] | None = None

    def __post_init__(self) -> None:
        P = self.transition_matrix
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"Transition matrix must be square, got {P.shape}")
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"Rows must sum to 1, got sums: {row_sums}")

    @property
    def n_states(self) -> int:
        return self.transition_matrix.shape[0]

    def n_step_transition(self, n: int) -> NDArray[np.floating]:
        """Compute n-step transition matrix P^n (Chapman-Kolmogorov)."""
        return np.linalg.matrix_power(self.transition_matrix, n)

    def stationary_distribution(self) -> NDArray[np.floating]:
        """Compute stationary distribution π satisfying πP = π.

        Method: Find left eigenvector of P with eigenvalue 1.
                Equivalently, find right eigenvector of Pᵀ with eigenvalue 1.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        # Find eigenvector corresponding to eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()  # Normalize to probability distribution
        return pi

    def simulate(self, initial_state: int, n_steps: int, seed: int = 42) -> list[int]:
        """Simulate a trajectory from the Markov chain."""
        rng = np.random.default_rng(seed)
        states = [initial_state]
        current = initial_state

        for _ in range(n_steps):
            current = rng.choice(self.n_states, p=self.transition_matrix[current])
            states.append(current)

        return states

    def is_ergodic(self) -> bool:
        """Check if chain is ergodic (irreducible + aperiodic)."""
        # Check irreducibility: (I + P)^(n-1) should have all positive entries
        n = self.n_states
        test = np.linalg.matrix_power(np.eye(n) + self.transition_matrix, n - 1)
        return bool(np.all(test > 0))


# =============================================================================
# Hidden Markov Models
# =============================================================================


@dataclass
class HiddenMarkovModel:
    """Discrete Hidden Markov Model (HMM).

    Reference: Rabiner, "A Tutorial on Hidden Markov Models" (1989)
               Proceedings of the IEEE, 77(2), 257-286

    Model specification λ = (A, B, π):
        A: State transition matrix — A_ij = P(sₜ₊₁=j | sₜ=i)
        B: Emission matrix — B_ij = P(oₜ=j | sₜ=i)
        π: Initial state distribution — πᵢ = P(s₁=i)

    Three fundamental problems:
        1. EVALUATION (Forward algorithm):
           Given λ and O, compute P(O|λ)

        2. DECODING (Viterbi algorithm):
           Given λ and O, find optimal state sequence S*

        3. LEARNING (Baum-Welch / EM):
           Given O, find λ* = argmax_λ P(O|λ)

    Application in Kalachakra:
        Model hidden planetary influences as latent states,
        with observable life events as emissions.
    """

    A: NDArray[np.floating]  # Transition matrix (N_states × N_states)
    B: NDArray[np.floating]  # Emission matrix (N_states × N_observations)
    pi: NDArray[np.floating]  # Initial distribution (N_states,)

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_observations(self) -> int:
        return self.B.shape[1]

    def forward(self, observations: list[int]) -> tuple[NDArray[np.floating], float]:
        """Forward algorithm — compute P(O|λ).

        Reference: Rabiner (1989), Eq. 18-21

        Initialization:
            α₁(i) = πᵢ · bᵢ(o₁)

        Induction:
            αₜ₊₁(j) = [Σᵢ αₜ(i) · aᵢⱼ] · bⱼ(oₜ₊₁)

        Termination:
            P(O|λ) = Σᵢ αₜ(i)

        Computed in log-space for numerical stability.

        Args:
            observations: Sequence of observation indices.

        Returns:
            Tuple of (alpha matrix, log probability).
        """
        T = len(observations)
        N = self.n_states

        # Log-space forward variables
        log_alpha = np.full((T, N), -np.inf)

        # Initialization: α₁(i) = πᵢ · bᵢ(o₁)
        log_alpha[0] = np.log(self.pi + 1e-300) + np.log(self.B[:, observations[0]] + 1e-300)

        # Induction
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-300))
                    + np.log(self.B[j, observations[t]] + 1e-300)
                )

        # Termination
        log_prob = float(logsumexp(log_alpha[T - 1]))

        return log_alpha, log_prob

    def viterbi(self, observations: list[int]) -> tuple[list[int], float]:
        """Viterbi algorithm — find most likely state sequence.

        Reference: Rabiner (1989), Eq. 32-35

        Initialization:
            δ₁(i) = πᵢ · bᵢ(o₁)
            ψ₁(i) = 0

        Recursion:
            δₜ(j) = max_i [δₜ₋₁(i) · aᵢⱼ] · bⱼ(oₜ)
            ψₜ(j) = argmax_i [δₜ₋₁(i) · aᵢⱼ]

        Termination:
            P* = max_i δₜ(i)
            s*_T = argmax_i δₜ(i)

        Backtracking:
            s*_t = ψₜ₊₁(s*ₜ₊₁)

        Args:
            observations: Sequence of observation indices.

        Returns:
            Tuple of (optimal state sequence, log probability).
        """
        T = len(observations)
        N = self.n_states

        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        log_delta[0] = np.log(self.pi + 1e-300) + np.log(self.B[:, observations[0]] + 1e-300)

        # Recursion
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for j in range(N):
                candidates = log_delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(candidates))
                log_delta[t, j] = candidates[psi[t, j]] + np.log(
                    self.B[j, observations[t]] + 1e-300
                )

        # Termination
        best_last = int(np.argmax(log_delta[T - 1]))
        log_prob = float(log_delta[T - 1, best_last])

        # Backtracking
        path = [0] * T
        path[T - 1] = best_last
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, log_prob

    def baum_welch(
        self,
        observations: list[int],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Baum-Welch (EM) algorithm — learn HMM parameters.

        Reference: Rabiner (1989), Eq. 40-42

        E-step:
            Compute forward (α) and backward (β) variables.
            γₜ(i) = P(sₜ=i | O, λ) — posterior state probability
            ξₜ(i,j) = P(sₜ=i, sₜ₊₁=j | O, λ) — posterior transition prob

        M-step:
            π̂ᵢ = γ₁(i)
            âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
            b̂ⱼ(k) = Σ_{t:oₜ=k} γₜ(j) / Σₜ γₜ(j)

        Returns:
            Updated (A, B, pi).
        """
        T = len(observations)
        N = self.n_states
        M = self.n_observations

        A = self.A.copy()
        B = self.B.copy()
        pi = self.pi.copy()

        prev_log_prob = -np.inf

        for iteration in range(max_iter):
            # --- Forward pass ---
            alpha = np.zeros((T, N))
            alpha[0] = pi * B[:, observations[0]]
            scale = np.zeros(T)
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0] + 1e-300

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, observations[t]]
                scale[t] = alpha[t].sum()
                alpha[t] /= scale[t] + 1e-300

            log_prob = float(np.sum(np.log(scale + 1e-300)))

            # --- Backward pass ---
            beta = np.zeros((T, N))
            beta[T - 1] = 1.0

            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[:, observations[t + 1]] * beta[t + 1])
                beta[t] /= scale[t + 1] + 1e-300

            # --- Compute γ and ξ ---
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((T - 1, N, N))
            for t in range(T - 1):
                numerator = (
                    np.outer(alpha[t], B[:, observations[t + 1]] * beta[t + 1]) * A
                )
                xi[t] = numerator / (numerator.sum() + 1e-300)

            # --- M-step ---
            pi = gamma[0]
            A = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            for k in range(M):
                mask = np.array(observations) == k
                B[:, k] = gamma[mask].sum(axis=0) / (gamma.sum(axis=0) + 1e-300)

            # Convergence check
            if abs(log_prob - prev_log_prob) < tol:
                log.debug(f"Baum-Welch converged at iteration {iteration + 1}")
                break
            prev_log_prob = log_prob

        self.__dict__["A"] = A
        self.__dict__["B"] = B
        self.__dict__["pi"] = pi

        return A, B, pi


# =============================================================================
# Gaussian Mixture Models
# =============================================================================


@dataclass
class GaussianMixtureModel:
    """Gaussian Mixture Model with EM fitting.

    Reference: Bishop PRML, Section 9.2

    PDF:
        p(x) = Σₖ₌₁ᴷ πₖ · N(x|μₖ, Σₖ)

    where πₖ are mixing coefficients (Σπₖ = 1).

    EM Algorithm:
        E-step: Compute responsibilities
            rₙₖ = πₖ N(xₙ|μₖ,Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ,Σⱼ)

        M-step: Update parameters
            Nₖ = Σₙ rₙₖ
            μₖ = (1/Nₖ) Σₙ rₙₖ xₙ
            Σₖ = (1/Nₖ) Σₙ rₙₖ (xₙ-μₖ)(xₙ-μₖ)ᵀ
            πₖ = Nₖ / N

    Application in Kalachakra:
        Cluster planetary configurations into distinct regime types.
    """

    n_components: int = 3
    max_iter: int = 100
    tol: float = 1e-6
    seed: int = 42

    # Fitted parameters
    weights_: NDArray[np.floating] | None = field(default=None, repr=False)
    means_: NDArray[np.floating] | None = field(default=None, repr=False)
    covariances_: NDArray[np.floating] | None = field(default=None, repr=False)

    def fit(self, X: NDArray[np.floating]) -> GaussianMixtureModel:
        """Fit GMM using Expectation-Maximization.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            self (fitted).
        """
        rng = np.random.default_rng(self.seed)
        N, D = X.shape
        K = self.n_components

        # Initialize with K-means++ style
        indices = rng.choice(N, K, replace=False)
        means = X[indices].copy()
        covariances = np.array([np.eye(D) for _ in range(K)])
        weights = np.ones(K) / K

        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # --- E-step: compute responsibilities ---
            resp = np.zeros((N, K))
            for k in range(K):
                resp[:, k] = weights[k] * sp_stats.multivariate_normal.pdf(
                    X, mean=means[k], cov=covariances[k] + 1e-6 * np.eye(D)
                )
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp /= resp_sum + 1e-300

            # --- M-step ---
            Nk = resp.sum(axis=0)
            for k in range(K):
                means[k] = (resp[:, k] @ X) / (Nk[k] + 1e-300)
                diff = X - means[k]
                covariances[k] = (
                    (resp[:, k][:, None] * diff).T @ diff / (Nk[k] + 1e-300)
                    + 1e-6 * np.eye(D)  # Regularization
                )
            weights = Nk / N

            # Log-likelihood
            log_likelihood = float(np.sum(np.log(resp_sum + 1e-300)))

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                log.debug(f"GMM EM converged at iteration {iteration + 1}")
                break
            prev_log_likelihood = log_likelihood

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.intp]:
        """Assign each sample to most likely component."""
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute posterior responsibilities for each component."""
        if self.weights_ is None or self.means_ is None or self.covariances_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        N = X.shape[0]
        D = X.shape[1]
        K = self.n_components
        resp = np.zeros((N, K))

        for k in range(K):
            resp[:, k] = self.weights_[k] * sp_stats.multivariate_normal.pdf(
                X, mean=self.means_[k],
                cov=self.covariances_[k] + 1e-6 * np.eye(D),
            )

        resp /= resp.sum(axis=1, keepdims=True) + 1e-300
        return resp
