"""
Domain 1: Mathematical Foundations — Optimization

Research References:
    - Kingma & Ba, "Adam: A Method for Stochastic Optimization" (ICLR 2015)
      arXiv:1412.6980
    - Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in
      Evolution Strategies" (2001), Evolutionary Computation 9(2)
    - Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772)
    - Boyd & Vandenberghe, "Convex Optimization" (2004)
    - Nocedal & Wright, "Numerical Optimization" (2006)
    - Kirkpatrick et al., "Optimization by Simulated Annealing" (1983)

Provides gradient-based and derivative-free optimizers for hyperparameter
tuning, model training, and combinatorial optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kalachakra.core.logging import get_logger

log = get_logger(__name__)

Vec = NDArray[np.floating]
ObjFunc = Callable[[Vec], float]
GradFunc = Callable[[Vec], Vec]


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    x: Vec                                  # Best solution found
    fun: float                              # Objective value at x
    n_iterations: int = 0
    n_evaluations: int = 0
    converged: bool = False
    method: str = ""
    history: list[float] = field(default_factory=list)  # Objective value history
    message: str = ""


# =============================================================================
# Gradient-Based Optimizers
# =============================================================================


def sgd(
    f: ObjFunc,
    grad_f: GradFunc,
    x0: Vec,
    lr: float = 0.01,
    momentum: float = 0.0,
    nesterov: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Stochastic Gradient Descent with optional momentum.

    Reference: Polyak, "Some methods of speeding up the convergence of
               iteration methods" (1964)
               Nesterov, "A method for solving a convex programming problem
               with convergence rate O(1/k²)" (1983)

    Standard SGD:
        θₜ = θₜ₋₁ - α · ∇f(θₜ₋₁)

    With Momentum (Polyak, 1964):
        vₜ = μ · vₜ₋₁ + ∇f(θₜ₋₁)
        θₜ = θₜ₋₁ - α · vₜ

    With Nesterov Momentum (Nesterov, 1983):
        vₜ = μ · vₜ₋₁ + ∇f(θₜ₋₁ - α · μ · vₜ₋₁)
        θₜ = θₜ₋₁ - α · vₜ

    Args:
        f: Objective function f(x) → ℝ.
        grad_f: Gradient function ∇f(x) → ℝⁿ.
        x0: Initial point.
        lr: Learning rate α.
        momentum: Momentum coefficient μ ∈ [0, 1).
        nesterov: If True, use Nesterov accelerated gradient.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on gradient norm.

    Returns:
        OptimizationResult.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    v = np.zeros_like(x)
    history: list[float] = []

    for i in range(max_iter):
        if nesterov:
            g = grad_f(x - lr * momentum * v)
        else:
            g = grad_f(x)

        v = momentum * v + g
        x = x - lr * v

        fval = f(x)
        history.append(fval)

        if np.linalg.norm(g) < tol:
            return OptimizationResult(
                x=x, fun=fval, n_iterations=i + 1,
                n_evaluations=i + 1, converged=True,
                method="sgd" + ("_nesterov" if nesterov else ""),
                history=history,
            )

    return OptimizationResult(
        x=x, fun=f(x), n_iterations=max_iter,
        n_evaluations=max_iter, converged=False,
        method="sgd", history=history,
    )


def adam(
    f: ObjFunc,
    grad_f: GradFunc,
    x0: Vec,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Adam optimizer (Adaptive Moment Estimation).

    Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization"
               ICLR 2015, arXiv:1412.6980

    Algorithm (Algorithm 1 from the paper):
        Input: α (step size), β₁, β₂ (decay rates), ε (numerical stability)
        Initialize: m₀ = 0, v₀ = 0, t = 0

        For each iteration t:
            1. gₜ = ∇_θ f(θₜ₋₁)                    (compute gradient)
            2. mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ      (1st moment estimate)
            3. vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²      (2nd moment estimate)
            4. m̂ₜ = mₜ / (1 - β₁ᵗ)                  (bias-corrected 1st moment)
            5. v̂ₜ = vₜ / (1 - β₂ᵗ)                  (bias-corrected 2nd moment)
            6. θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)      (parameter update)

    Bias Correction Rationale:
        Since m₀ = v₀ = 0, the moments are biased toward zero especially
        in early steps. The correction factors (1 - βᵗ) exactly counteract
        this initialization bias.

    Default hyperparameters (from paper):
        α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸

    Args:
        f: Objective function.
        grad_f: Gradient function.
        x0: Initial parameters θ₀.
        lr: Step size α.
        beta1: Exponential decay rate for 1st moment β₁.
        beta2: Exponential decay rate for 2nd moment β₂.
        epsilon: Numerical stability constant ε.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on gradient norm.

    Returns:
        OptimizationResult.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)  # First moment (mean of gradients)
    v = np.zeros_like(x)  # Second moment (mean of squared gradients)
    history: list[float] = []

    for t in range(1, max_iter + 1):
        # Step 1: Compute gradient
        g = grad_f(x)

        # Step 2: Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Step 3: Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * g**2

        # Step 4: Bias-corrected first moment
        m_hat = m / (1 - beta1**t)

        # Step 5: Bias-corrected second moment
        v_hat = v / (1 - beta2**t)

        # Step 6: Parameter update
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        fval = f(x)
        history.append(fval)

        if np.linalg.norm(g) < tol:
            return OptimizationResult(
                x=x, fun=fval, n_iterations=t,
                n_evaluations=t, converged=True,
                method="adam", history=history,
            )

    return OptimizationResult(
        x=x, fun=f(x), n_iterations=max_iter,
        n_evaluations=max_iter, converged=False,
        method="adam", history=history,
    )


def adamw(
    f: ObjFunc,
    grad_f: GradFunc,
    x0: Vec,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizationResult:
    """AdamW optimizer (Adam with decoupled Weight Decay).

    Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
               ICLR 2019, arXiv:1711.05101

    Key insight: L2 regularization ≠ weight decay for adaptive methods.
    AdamW decouples weight decay from the gradient-based update:

        1-5. Same as Adam (compute m̂ₜ, v̂ₜ)
        6. θₜ = (1 - α·λ) · θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)

    where λ is the weight decay coefficient. This is equivalent to
    adding a penalty term λ‖θ‖² directly to the update, rather than
    to the gradient (which would be scaled by the adaptive learning rate).

    Args:
        f: Objective function.
        grad_f: Gradient function.
        x0: Initial parameters.
        lr: Learning rate α.
        beta1, beta2: Moment decay rates.
        epsilon: Numerical stability.
        weight_decay: Weight decay λ.
        max_iter: Max iterations.
        tol: Convergence tolerance.

    Returns:
        OptimizationResult.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history: list[float] = []

    for t in range(1, max_iter + 1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Decoupled weight decay
        x = (1 - lr * weight_decay) * x - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        fval = f(x)
        history.append(fval)

        if np.linalg.norm(g) < tol:
            return OptimizationResult(
                x=x, fun=fval, n_iterations=t,
                n_evaluations=t, converged=True,
                method="adamw", history=history,
            )

    return OptimizationResult(
        x=x, fun=f(x), n_iterations=max_iter, n_evaluations=max_iter,
        converged=False, method="adamw", history=history,
    )


# =============================================================================
# Derivative-Free Optimizers
# =============================================================================


def simulated_annealing(
    f: ObjFunc,
    x0: Vec,
    bounds: list[tuple[float, float]],
    T_init: float = 1.0,
    T_min: float = 1e-8,
    cooling_rate: float = 0.995,
    step_size: float = 0.1,
    max_iter: int = 10_000,
    seed: int = 42,
) -> OptimizationResult:
    """Simulated Annealing optimization.

    Reference: Kirkpatrick, Gelatt & Vecchi, "Optimization by Simulated
               Annealing", Science 220(4598), 671-680 (1983)

    Inspired by the annealing process in metallurgy. At each step:

    Algorithm:
        1. Generate candidate: x' = x + N(0, step_size)
        2. Compute ΔE = f(x') - f(x)
        3. If ΔE < 0: accept (downhill move)
        4. If ΔE ≥ 0: accept with probability P = exp(-ΔE / T)
           (Metropolis criterion — allows uphill moves to escape local minima)
        5. Cool: T ← T · cooling_rate

    The acceptance probability P = exp(-ΔE/T) is the Boltzmann distribution.
    As T → 0, the algorithm becomes greedy (only accepts improvements).

    Args:
        f: Objective function to minimize.
        x0: Initial point.
        bounds: Search bounds for each dimension [(lo, hi), ...].
        T_init: Initial temperature.
        T_min: Minimum temperature (stopping criterion).
        cooling_rate: Geometric cooling factor ∈ (0, 1).
        step_size: Standard deviation of Gaussian perturbation.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    bounds_arr = np.array(bounds)

    best_x = x.copy()
    best_f = f(x)
    current_f = best_f
    T = T_init
    history: list[float] = [best_f]
    n_evals = 1

    for i in range(max_iter):
        if T < T_min:
            break

        # Generate candidate (Gaussian perturbation, clipped to bounds)
        candidate = x + rng.normal(0, step_size, size=len(x))
        candidate = np.clip(candidate, bounds_arr[:, 0], bounds_arr[:, 1])

        candidate_f = f(candidate)
        n_evals += 1
        delta_e = candidate_f - current_f

        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            x = candidate
            current_f = candidate_f

            if current_f < best_f:
                best_x = x.copy()
                best_f = current_f

        # Geometric cooling
        T *= cooling_rate
        history.append(best_f)

    log.debug(f"SA: {n_evals} evaluations, T_final={T:.2e}, best_f={best_f:.6e}")
    return OptimizationResult(
        x=best_x, fun=best_f, n_iterations=i + 1,
        n_evaluations=n_evals, converged=T < T_min,
        method="simulated_annealing", history=history,
    )


def particle_swarm(
    f: ObjFunc,
    bounds: list[tuple[float, float]],
    n_particles: int = 50,
    w: float = 0.7298,
    c1: float = 1.49618,
    c2: float = 1.49618,
    max_iter: int = 500,
    seed: int = 42,
) -> OptimizationResult:
    """Particle Swarm Optimization (PSO).

    Reference: Kennedy & Eberhart, "Particle Swarm Optimization" (1995)
               Clerc & Kennedy, "The particle swarm — explosion, stability,
               and convergence in a multidimensional complex space" (2002)

    Each particle i has position xᵢ and velocity vᵢ. Update rules:

        vᵢ(t+1) = w·vᵢ(t) + c₁·r₁·(pᵢ - xᵢ(t)) + c₂·r₂·(g - xᵢ(t))
        xᵢ(t+1) = xᵢ(t) + vᵢ(t+1)

    where:
        w  = inertia weight (controls exploration vs exploitation)
        c₁ = cognitive coefficient (attraction to personal best pᵢ)
        c₂ = social coefficient (attraction to global best g)
        r₁, r₂ ~ U(0,1) (random factors)

    Default coefficients from Clerc & Kennedy (2002):
        w = 0.7298, c₁ = c₂ = 1.49618
        (These guarantee convergence via constriction coefficient χ)

    Args:
        f: Objective function to minimize.
        bounds: Search bounds [(lo, hi), ...].
        n_particles: Swarm size.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    bounds_arr = np.array(bounds)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]

    # Initialize particles uniformly in search space
    positions = rng.uniform(lo, hi, size=(n_particles, dim))
    velocities = rng.uniform(-(hi - lo), (hi - lo), size=(n_particles, dim)) * 0.1

    # Evaluate initial fitness
    fitness = np.array([f(p) for p in positions])
    n_evals = n_particles

    # Personal bests
    p_best_pos = positions.copy()
    p_best_fit = fitness.copy()

    # Global best
    g_best_idx = np.argmin(fitness)
    g_best_pos = positions[g_best_idx].copy()
    g_best_fit = fitness[g_best_idx]

    history: list[float] = [g_best_fit]

    for iteration in range(max_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        # Update velocities
        cognitive = c1 * r1 * (p_best_pos - positions)
        social = c2 * r2 * (g_best_pos - positions)
        velocities = w * velocities + cognitive + social

        # Update positions (clip to bounds)
        positions = np.clip(positions + velocities, lo, hi)

        # Evaluate fitness
        fitness = np.array([f(p) for p in positions])
        n_evals += n_particles

        # Update personal bests
        improved = fitness < p_best_fit
        p_best_pos[improved] = positions[improved]
        p_best_fit[improved] = fitness[improved]

        # Update global best
        gen_best_idx = np.argmin(p_best_fit)
        if p_best_fit[gen_best_idx] < g_best_fit:
            g_best_pos = p_best_pos[gen_best_idx].copy()
            g_best_fit = p_best_fit[gen_best_idx]

        history.append(g_best_fit)

    log.debug(f"PSO: {n_evals} evaluations, best_f={g_best_fit:.6e}")
    return OptimizationResult(
        x=g_best_pos, fun=g_best_fit, n_iterations=max_iter,
        n_evaluations=n_evals, converged=True,
        method="particle_swarm", history=history,
    )


def cma_es(
    f: ObjFunc,
    x0: Vec,
    sigma0: float = 0.5,
    pop_size: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-11,
    seed: int = 42,
) -> OptimizationResult:
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Reference: Hansen & Ostermeier, "Completely Derandomized Self-Adaptation
               in Evolution Strategies" (2001), Evolutionary Computation 9(2)
               Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772)

    CMA-ES adapts a multivariate normal search distribution:
        x_k^(g+1) ~ m^(g) + σ^(g) · N(0, C^(g))

    Core update equations:

    1. SELECTION & RECOMBINATION (mean update):
        m^(g+1) = Σᵢ₌₁^μ wᵢ · x_{i:λ}^(g+1)

    2. EVOLUTION PATH (cumulation for rank-one update):
        p_c^(g+1) = (1 - c_c) · p_c^(g) +
                    √(c_c(2 - c_c) · μ_eff) · (m^(g+1) - m^(g)) / σ^(g)

    3. COVARIANCE MATRIX ADAPTATION:
        C^(g+1) = (1 - c₁ - c_μ) · C^(g)                    (old covariance)
                + c₁ · p_c · p_cᵀ                             (rank-one update)
                + c_μ · Σᵢ wᵢ · yᵢ · yᵢᵀ                     (rank-μ update)

        where yᵢ = (x_{i:λ} - m^(g)) / σ^(g)

    4. STEP-SIZE CONTROL (CSA — Cumulative Step-size Adaptation):
        p_σ^(g+1) = (1 - c_σ) · p_σ^(g) +
                    √(c_σ(2 - c_σ) · μ_eff) · C^(-½) · (m^(g+1) - m^(g)) / σ^(g)

        σ^(g+1) = σ^(g) · exp(c_σ/d_σ · (‖p_σ‖/E[‖N(0,I)‖] - 1))

    Key hyperparameters (all derived from dimension n):
        λ = 4 + ⌊3 ln(n)⌋           (population size)
        μ = ⌊λ/2⌋                    (number of parents)
        c_c ≈ 4/n                     (cumulation for C)
        c_σ ≈ (μ_eff + 2)/(n + μ_eff + 5)  (cumulation for σ)
        c₁ ≈ 2/n²                    (learning rate for rank-one)
        c_μ ≈ μ_eff/n²               (learning rate for rank-μ)
        d_σ ≈ 1 + c_σ                (damping for σ)

    Args:
        f: Objective function to minimize.
        x0: Initial mean m₀.
        sigma0: Initial step size σ₀.
        pop_size: Population size λ. None = 4 + ⌊3 ln(n)⌋.
        max_iter: Maximum generations.
        tol: Convergence tolerance (on σ · max(diag(C))).
        seed: Random seed.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=np.float64)
    n = len(x0)

    # --- Strategy Parameters (all from Hansen 2016 tutorial) ---
    lam = pop_size if pop_size is not None else 4 + int(3 * np.log(n))
    mu = lam // 2

    # Recombination weights (log-linear)
    weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights_raw / weights_raw.sum()
    mu_eff = 1.0 / np.sum(weights**2)  # Variance-effective selection mass

    # Cumulation parameters
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)

    # Learning rates for covariance matrix
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu_raw = min(
        1 - c1,
        2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff),
    )

    # Expected length of N(0, I) vector
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    # --- Initialize state ---
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(n)                       # Covariance matrix
    p_sigma = np.zeros(n)               # Evolution path for σ
    p_c = np.zeros(n)                   # Evolution path for C

    best_x = x0.copy()
    best_f = f(x0)
    n_evals = 1
    history: list[float] = [best_f]

    for gen in range(max_iter):
        # --- SAMPLE λ offspring ---
        # Eigendecomposition of C for sampling: C = B D² Bᵀ
        eigenvalues, B = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        D = np.sqrt(eigenvalues)
        invsqrtC = B @ np.diag(1.0 / D) @ B.T

        arz = rng.standard_normal((lam, n))  # Standard normal samples
        arx = mean + sigma * (arz @ np.diag(D) @ B.T)  # Transformed samples

        # --- EVALUATE & RANK ---
        fitness = np.array([f(x) for x in arx])
        n_evals += lam

        # Sort by fitness (ascending = best first)
        order = np.argsort(fitness)

        # Update best
        if fitness[order[0]] < best_f:
            best_f = fitness[order[0]]
            best_x = arx[order[0]].copy()

        history.append(best_f)

        # --- RECOMBINATION: update mean ---
        old_mean = mean.copy()
        selected = arx[order[:mu]]
        mean = weights @ selected

        # --- CUMULATION: evolution paths ---
        mean_diff = (mean - old_mean) / sigma

        # σ path (in isotropic coordinate system)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
            c_sigma * (2 - c_sigma) * mu_eff
        ) * (invsqrtC @ mean_diff)

        # C path (with heaviside function for stalling detection)
        h_sigma = float(
            np.linalg.norm(p_sigma)
            / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            < (1.4 + 2 / (n + 1)) * chi_n
        )

        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(
            c_c * (2 - c_c) * mu_eff
        ) * mean_diff

        # --- COVARIANCE MATRIX ADAPTATION ---
        # Rank-one update
        rank_one = np.outer(p_c, p_c)

        # Rank-μ update
        artmp = (selected - old_mean) / sigma
        rank_mu = sum(
            w_i * np.outer(y_i, y_i) for w_i, y_i in zip(weights, artmp)
        )

        C = (
            (1 - c1 - c_mu_raw + (1 - h_sigma) * c1 * c_c * (2 - c_c)) * C
            + c1 * rank_one
            + c_mu_raw * rank_mu
        )

        # Enforce symmetry
        C = (C + C.T) / 2

        # --- STEP-SIZE CONTROL (CSA) ---
        sigma = sigma * np.exp(
            (c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)
        )

        # --- CONVERGENCE CHECK ---
        if sigma * np.max(D) < tol:
            log.debug(f"CMA-ES converged at generation {gen + 1}")
            return OptimizationResult(
                x=best_x, fun=best_f, n_iterations=gen + 1,
                n_evaluations=n_evals, converged=True,
                method="cma_es", history=history,
            )

    log.debug(f"CMA-ES: {n_evals} evaluations, best_f={best_f:.6e}")
    return OptimizationResult(
        x=best_x, fun=best_f, n_iterations=max_iter,
        n_evaluations=n_evals, converged=False,
        method="cma_es", history=history,
    )


def differential_evolution(
    f: ObjFunc,
    bounds: list[tuple[float, float]],
    pop_size: int = 50,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = "best/1/bin",
    max_iter: int = 1000,
    tol: float = 1e-10,
    seed: int = 42,
) -> OptimizationResult:
    """Differential Evolution (DE) optimizer.

    Reference: Storn & Price, "Differential Evolution — A Simple and
               Efficient Heuristic for Global Optimization" (1997)
               J. Global Optimization 11, 341-359

    Algorithm (DE/best/1/bin):
        For each target vector xᵢ in population:

        1. MUTATION: Create donor vector
           vᵢ = x_best + F · (xᵣ₁ - xᵣ₂)
           where r1 ≠ r2 ≠ i, F ∈ (0, 2] is the mutation scale factor

        2. CROSSOVER (binomial): Create trial vector
           uᵢⱼ = vᵢⱼ  if rand() < CR or j = j_rand
                  xᵢⱼ  otherwise
           where CR ∈ [0, 1] is crossover rate, j_rand ensures at least
           one dimension is mutated

        3. SELECTION: Greedy replacement
           xᵢ^(g+1) = uᵢ  if f(uᵢ) ≤ f(xᵢ)
                       xᵢ  otherwise

    Args:
        f: Objective function.
        bounds: Search bounds [(lo, hi), ...].
        pop_size: Population size (≥ 4).
        F: Mutation scale factor ∈ (0, 2].
        CR: Crossover probability ∈ [0, 1].
        strategy: Mutation strategy.
        max_iter: Maximum generations.
        tol: Convergence tolerance.
        seed: Random seed.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    bounds_arr = np.array(bounds)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]

    # Initialize population
    pop = rng.uniform(lo, hi, (pop_size, dim))
    fitness = np.array([f(x) for x in pop])
    n_evals = pop_size

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    history: list[float] = [best_f]

    for gen in range(max_iter):
        for i in range(pop_size):
            # Select 2 random distinct individuals (≠ i)
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2 = rng.choice(candidates, 2, replace=False)

            # Mutation: DE/best/1
            donor = best_x + F * (pop[r1] - pop[r2])
            donor = np.clip(donor, lo, hi)

            # Binomial crossover
            j_rand = rng.integers(0, dim)
            mask = rng.random(dim) < CR
            mask[j_rand] = True  # Ensure at least one dimension from donor
            trial = np.where(mask, donor, pop[i])

            # Selection
            trial_f = f(trial)
            n_evals += 1

            if trial_f <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_f

                if trial_f < best_f:
                    best_x = trial.copy()
                    best_f = trial_f

        history.append(best_f)

        # Convergence check
        if np.std(fitness) < tol:
            break

    log.debug(f"DE: {n_evals} evaluations, best_f={best_f:.6e}")
    return OptimizationResult(
        x=best_x, fun=best_f, n_iterations=gen + 1,
        n_evaluations=n_evals, converged=np.std(fitness) < tol,
        method="differential_evolution", history=history,
    )


def genetic_algorithm(
    f: ObjFunc,
    bounds: list[tuple[float, float]],
    pop_size: int = 100,
    mutation_rate: float = 0.01,
    crossover_rate: float = 0.8,
    elite_fraction: float = 0.1,
    max_iter: int = 500,
    seed: int = 42,
) -> OptimizationResult:
    """Real-coded Genetic Algorithm (GA).

    Reference: Holland, "Adaptation in Natural and Artificial Systems" (1975)
               Goldberg, "Genetic Algorithms in Search, Optimization, and
               Machine Learning" (1989)

    Algorithm:
        1. Initialize random population
        2. Evaluate fitness
        3. SELECTION: Tournament selection (k=3)
        4. CROSSOVER: BLX-α blend crossover
           child = parent1 + α·(parent2 - parent1), α ~ U(-0.5, 1.5)
        5. MUTATION: Gaussian perturbation
           x' = x + N(0, σ), σ decays with generation
        6. ELITISM: Top fraction survives unchanged
        7. Repeat 2-6

    Args:
        f: Objective function (minimization).
        bounds: Search bounds.
        pop_size: Population size.
        mutation_rate: Probability of mutating each gene.
        crossover_rate: Probability of crossover.
        elite_fraction: Fraction of best individuals preserved.
        max_iter: Maximum generations.
        seed: Random seed.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    bounds_arr = np.array(bounds)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]

    # Initialize
    pop = rng.uniform(lo, hi, (pop_size, dim))
    fitness = np.array([f(x) for x in pop])
    n_evals = pop_size
    n_elite = max(1, int(pop_size * elite_fraction))

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    history: list[float] = [best_f]

    for gen in range(max_iter):
        order = np.argsort(fitness)
        new_pop = [pop[order[i]].copy() for i in range(n_elite)]

        while len(new_pop) < pop_size:
            # Tournament selection (k=3)
            parents = []
            for _ in range(2):
                tournament = rng.choice(pop_size, 3, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                parents.append(pop[winner])

            # BLX-α crossover
            if rng.random() < crossover_rate:
                alpha = rng.uniform(-0.5, 1.5, dim)
                child = parents[0] + alpha * (parents[1] - parents[0])
            else:
                child = parents[0].copy()

            # Gaussian mutation (adaptive σ)
            sigma = (hi - lo) * 0.1 * (1 - gen / max_iter)
            mask = rng.random(dim) < mutation_rate
            child[mask] += rng.normal(0, sigma[mask])

            child = np.clip(child, lo, hi)
            new_pop.append(child)

        pop = np.array(new_pop[:pop_size])
        fitness = np.array([f(x) for x in pop])
        n_evals += pop_size

        gen_best = np.argmin(fitness)
        if fitness[gen_best] < best_f:
            best_f = fitness[gen_best]
            best_x = pop[gen_best].copy()

        history.append(best_f)

    return OptimizationResult(
        x=best_x, fun=best_f, n_iterations=max_iter,
        n_evaluations=n_evals, converged=True,
        method="genetic_algorithm", history=history,
    )


# =============================================================================
# Newton's Method (2nd-order gradient)
# =============================================================================


def newton_method(
    f: ObjFunc,
    grad_f: GradFunc,
    hess_f: Callable[[Vec], NDArray[np.floating]],
    x0: Vec,
    max_iter: int = 100,
    tol: float = 1e-10,
    line_search: bool = True,
) -> OptimizationResult:
    """Newton's method for optimization.

    Reference: Nocedal & Wright, "Numerical Optimization" (2006), Ch. 6

    Update rule:
        xₖ₊₁ = xₖ - [∇²f(xₖ)]⁻¹ · ∇f(xₖ)

    Newton decrement:
        λ(x) = [∇f(x)ᵀ · [∇²f(x)]⁻¹ · ∇f(x)]^(1/2)

    Convergence: Quadratic near optimum (if Hessian is Lipschitz continuous).

    With Armijo backtracking line search:
        α = max{β^k : f(x + β^k d) ≤ f(x) + c₁ · β^k · ∇f(x)ᵀd}

    Args:
        f: Objective function.
        grad_f: Gradient function ∇f.
        hess_f: Hessian function ∇²f.
        x0: Initial point.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (on gradient norm).
        line_search: If True, use Armijo backtracking.

    Returns:
        OptimizationResult.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    history: list[float] = []

    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)

        if np.linalg.norm(g) < tol:
            return OptimizationResult(
                x=x, fun=f(x), n_iterations=k + 1,
                n_evaluations=k + 1, converged=True,
                method="newton", history=history,
            )

        # Newton direction: d = -H⁻¹g
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Hessian singular — fall back to gradient descent
            d = -g

        # Armijo backtracking line search
        if line_search:
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            fval = f(x)
            for _ in range(50):
                if f(x + alpha * d) <= fval + c1 * alpha * (g @ d):
                    break
                alpha *= rho
        else:
            alpha = 1.0

        x = x + alpha * d
        fval = f(x)
        history.append(fval)

    return OptimizationResult(
        x=x, fun=f(x), n_iterations=max_iter,
        n_evaluations=max_iter, converged=False,
        method="newton", history=history,
    )
