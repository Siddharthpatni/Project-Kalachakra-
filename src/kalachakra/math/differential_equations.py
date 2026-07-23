"""
Domain 1: Mathematical Foundations — Differential Equations

Research References:
    - Press et al., "Numerical Recipes" (3rd ed.), Ch. 17
    - Dormand & Prince, "A family of embedded Runge-Kutta formulae" (1980)
    - Hairer, Nørsett & Wanner, "Solving Ordinary Differential Equations I" (1993)
    - Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)

Provides ODE solvers for orbital mechanics (Kepler's equation, N-body),
dynamical systems analysis, and Neural ODE foundations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kalachakra.core.logging import get_logger

log = get_logger(__name__)

# Type aliases
State = NDArray[np.floating]  # State vector y ∈ ℝⁿ
ODEFunc = Callable[[float, State], State]  # f(t, y) → dy/dt


@dataclass
class ODESolution:
    """Solution of an ODE initial value problem.

    Contains the trajectory y(t) at all computed time points.
    """

    t: NDArray[np.floating]            # Time points
    y: NDArray[np.floating]            # States at each time point, shape (n_steps, n_dims)
    n_evaluations: int = 0             # Number of f(t,y) evaluations
    n_steps: int = 0                   # Number of accepted steps
    n_rejected: int = 0                # Number of rejected steps (adaptive only)
    method: str = ""                   # Solver method name
    success: bool = True
    message: str = ""
    step_sizes: list[float] = field(default_factory=list)  # Accepted step sizes


# =============================================================================
# Fixed-Step Solvers
# =============================================================================


def euler(
    f: ODEFunc,
    y0: State,
    t_span: tuple[float, float],
    h: float,
) -> ODESolution:
    """Forward Euler method (1st order).

    Reference: Hairer et al., "Solving ODEs I", Ch. II.1

    Formula:
        yₙ₊₁ = yₙ + h · f(tₙ, yₙ)

    Local truncation error: O(h²)
    Global error: O(h)

    ⚠️ Unstable for stiff problems. Use only for reference/testing.

    Args:
        f: Right-hand side function f(t, y) → dy/dt.
        y0: Initial state vector.
        t_span: Integration interval (t_start, t_end).
        h: Fixed step size.

    Returns:
        ODESolution with trajectory.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / h))

    ts = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, len(y0)))

    ts[0] = t_start
    ys[0] = y0

    n_evals = 0
    for i in range(n_steps):
        t = ts[i]
        y = ys[i]
        ys[i + 1] = y + h * f(t, y)
        ts[i + 1] = t + h
        n_evals += 1

    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        method="euler",
    )


def rk4(
    f: ODEFunc,
    y0: State,
    t_span: tuple[float, float],
    h: float,
) -> ODESolution:
    """Classical 4th-order Runge-Kutta method.

    Reference: Press et al., "Numerical Recipes", Ch. 17.1

    Formula (the "classical" RK4):
        k₁ = f(tₙ, yₙ)
        k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
        k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
        k₄ = f(tₙ + h, yₙ + h·k₃)

        yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

    This is a weighted average of four slope estimates:
        - k₁: slope at interval start
        - k₂: slope at midpoint using k₁
        - k₃: slope at midpoint using k₂
        - k₄: slope at interval end using k₃

    Local truncation error: O(h⁵)
    Global error: O(h⁴)

    Application in Kalachakra:
        Solve Kepler's equation of motion, planetary orbit propagation,
        and N-body gravitational dynamics.

    Args:
        f: Right-hand side function dy/dt = f(t, y).
        y0: Initial state vector y(t₀).
        t_span: Integration interval (t_start, t_end).
        h: Fixed step size.

    Returns:
        ODESolution with trajectory at each time step.

    Example:
        >>> # Simple harmonic oscillator: y'' + y = 0
        >>> # State: [position, velocity]
        >>> def harmonic(t, y):
        ...     return np.array([y[1], -y[0]])
        >>> sol = rk4(harmonic, np.array([1.0, 0.0]), (0, 2*np.pi), 0.01)
        >>> np.allclose(sol.y[-1, 0], 1.0, atol=1e-6)
        True
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / h))

    ts = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, len(y0)))

    ts[0] = t_start
    ys[0] = y0

    n_evals = 0
    for i in range(n_steps):
        t = ts[i]
        y = ys[i]

        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)

        ys[i + 1] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ts[i + 1] = t + h
        n_evals += 4

    log.debug(f"RK4: {n_steps} steps, {n_evals} evaluations")
    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        method="rk4",
    )


# =============================================================================
# Adaptive-Step Solvers
# =============================================================================

# Dormand-Prince 5(4) Butcher tableau coefficients
# Reference: Dormand & Prince (1980), "A family of embedded Runge-Kutta formulae"
# These are the EXACT coefficients from the paper.

_DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])

_DP_A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
    [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
])

# 5th-order weights (used to advance the solution)
_DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])

# 4th-order weights (used for error estimation)
_DP_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

# Error coefficients: E = B5 - B4
_DP_E = _DP_B5 - _DP_B4


def dormand_prince(
    f: ODEFunc,
    y0: State,
    t_span: tuple[float, float],
    rtol: float = 1e-8,
    atol: float = 1e-8,
    h_init: float | None = None,
    h_min: float = 1e-12,
    h_max: float | None = None,
    max_steps: int = 100_000,
) -> ODESolution:
    """Dormand-Prince 5(4) adaptive step-size ODE solver (RK45).

    Reference: Dormand & Prince (1980), "A family of embedded Runge-Kutta formulae"
               J. Comp. Appl. Math., Vol. 6, p. 19-26

    This is a 7-stage, 5th-order method with an embedded 4th-order solution.
    Only requires 6 function evaluations per step due to the FSAL
    (First Same As Last) property: k₇ of step n = k₁ of step n+1.

    Algorithm:
        1. Compute 7 stages: kᵢ = f(tₙ + cᵢh, yₙ + h Σⱼ aᵢⱼ kⱼ)
        2. Compute 5th-order solution: y₅ = yₙ + h Σᵢ b₅ᵢ kᵢ
        3. Compute 4th-order solution: y₄ = yₙ + h Σᵢ b₄ᵢ kᵢ
        4. Error estimate: err = ‖y₅ - y₄‖ / (atol + rtol·max(|yₙ|, |y₅|))
        5. If err ≤ 1: accept step, advance with y₅
        6. Adjust step size: h_new = h · min(5, max(0.2, 0.9 · err^(-1/5)))

    Local truncation error: O(h⁶)
    Global error: O(h⁵)

    Application in Kalachakra:
        High-accuracy planetary orbit integration, solving Kepler's equation
        of motion with adaptive precision matching Swiss Ephemeris accuracy.

    Args:
        f: Right-hand side function dy/dt = f(t, y).
        y0: Initial state vector.
        t_span: Integration interval (t_start, t_end).
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        h_init: Initial step size. None = automatic selection.
        h_min: Minimum allowed step size.
        h_max: Maximum allowed step size. None = (t_end - t_start).
        max_steps: Maximum number of steps before giving up.

    Returns:
        ODESolution with adaptive-step trajectory.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span
    direction = 1.0 if t_end > t_start else -1.0

    if h_max is None:
        h_max = abs(t_end - t_start)
    if h_init is None:
        h_init = _select_initial_step(f, t_start, y0, 5, rtol, atol, direction)

    h = direction * min(abs(h_init), h_max)

    # Storage (dynamic)
    ts_list: list[float] = [t_start]
    ys_list: list[State] = [y0.copy()]
    step_sizes: list[float] = []

    t = t_start
    y = y0.copy()
    n_evals = 0
    n_steps = 0
    n_rejected = 0

    # FSAL: first evaluation
    k1 = f(t, y)
    n_evals += 1

    while direction * (t - t_end) < 0:
        if n_steps >= max_steps:
            log.warning(f"Dormand-Prince: max_steps ({max_steps}) exceeded at t={t:.6e}")
            break

        # Don't step past t_end
        if direction * (t + h - t_end) > 0:
            h = t_end - t

        abs_h = abs(h)

        # Compute stages k₂ through k₇
        k = np.zeros((7, len(y)))
        k[0] = k1

        for i in range(1, 7):
            stage_y = y + h * np.dot(_DP_A[i, :i], k[:i])
            k[i] = f(t + _DP_C[i] * h, stage_y)
        n_evals += 6

        # 5th-order solution (advance with this)
        y_new = y + h * np.dot(_DP_B5, k)

        # Error estimate: difference between 5th and 4th order
        error_vec = h * np.dot(_DP_E, k)

        # Compute scaled error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((error_vec / scale) ** 2))

        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            k1 = k[6]  # FSAL: reuse last stage as first stage of next step

            ts_list.append(t)
            ys_list.append(y.copy())
            step_sizes.append(abs_h)
            n_steps += 1
        else:
            # Reject step
            n_rejected += 1

        # Step size control (PI controller)
        # h_new = h · S · err^(-1/q) where q=5 (order), S=0.9 (safety)
        if err_norm == 0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err_norm ** (-1.0 / 5.0)))

        h = h * factor
        h = direction * min(abs(h), h_max)
        h = direction * max(abs(h), h_min)

    ts = np.array(ts_list)
    ys = np.array(ys_list)

    log.debug(
        f"Dormand-Prince: {n_steps} accepted, {n_rejected} rejected, "
        f"{n_evals} evaluations"
    )

    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        n_rejected=n_rejected, method="dormand_prince_5(4)",
        step_sizes=step_sizes,
        success=direction * (t - t_end) >= 0,
        message="Integration completed" if direction * (t - t_end) >= 0 else "Max steps exceeded",
    )


def _select_initial_step(
    f: ODEFunc,
    t0: float,
    y0: State,
    order: int,
    rtol: float,
    atol: float,
    direction: float,
) -> float:
    """Automatically select initial step size.

    Reference: Hairer et al., "Solving ODEs I", p. 169

    Algorithm:
        1. Compute d₀ = ‖y₀‖ / sc, d₁ = ‖f(t₀, y₀)‖ / sc
        2. h₀ = 0.01 · d₀/d₁ (or 1e-6 if d₀ or d₁ ≈ 0)
        3. Compute one Euler step: y₁ = y₀ + h₀·f₀
        4. d₂ = ‖f(t₀+h₀, y₁) - f₀‖ / (h₀·sc)
        5. h₁ = (0.01 / max(d₁, d₂))^(1/(order+1))
        6. Return min(100·h₀, h₁)
    """
    sc = atol + np.abs(y0) * rtol
    f0 = f(t0, y0)

    d0 = np.sqrt(np.mean((y0 / sc) ** 2))
    d1 = np.sqrt(np.mean((f0 / sc) ** 2))

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = f(t0 + h0 * direction, y1)

    d2 = np.sqrt(np.mean(((f1 - f0) / sc) ** 2)) / h0

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return min(100 * h0, h1)


# =============================================================================
# Symplectic Integrators (for Hamiltonian systems / orbital mechanics)
# =============================================================================


def stormer_verlet(
    acceleration_fn: Callable[[State], State],
    q0: State,
    p0: State,
    t_span: tuple[float, float],
    h: float,
) -> ODESolution:
    """Störmer-Verlet (Leapfrog) symplectic integrator.

    Reference: Hairer, Lubich & Wanner, "Geometric Numerical Integration" (2006)

    For Hamiltonian systems H(q, p) = T(p) + V(q):
        dq/dt = ∂H/∂p = p/m
        dp/dt = -∂H/∂q = F(q)  (force = -∇V)

    Störmer-Verlet Algorithm:
        pₙ₊½ = pₙ + (h/2) · F(qₙ)         (half-step kick)
        qₙ₊₁ = qₙ + h · pₙ₊½ / m          (full drift)
        pₙ₊₁ = pₙ₊½ + (h/2) · F(qₙ₊₁)     (half-step kick)

    Properties:
        - Symplectic: preserves phase space volume (Liouville's theorem)
        - Time-reversible
        - Energy error bounded (no secular drift) — critical for N-body
        - 2nd-order accurate: O(h²)

    Application in Kalachakra:
        N-body gravitational simulation of planetary orbits. Symplectic
        integrators prevent artificial energy drift over millions of years,
        unlike non-symplectic methods like RK4.

    Args:
        acceleration_fn: F(q) → acceleration vector (force/mass = -∇V/m).
        q0: Initial position vector.
        p0: Initial momentum (velocity) vector.
        t_span: Integration interval (t_start, t_end).
        h: Fixed step size.

    Returns:
        ODESolution where y[:, :n] = positions, y[:, n:] = momenta.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / h))
    n_dim = len(q0)

    ts = np.zeros(n_steps + 1)
    qs = np.zeros((n_steps + 1, n_dim))
    ps = np.zeros((n_steps + 1, n_dim))

    ts[0] = t_start
    qs[0] = q0
    ps[0] = p0

    q = q0.copy()
    p = p0.copy()
    n_evals = 0

    for i in range(n_steps):
        # Half-step kick
        a = acceleration_fn(q)
        p_half = p + 0.5 * h * a
        n_evals += 1

        # Full drift
        q = q + h * p_half

        # Half-step kick
        a = acceleration_fn(q)
        p = p_half + 0.5 * h * a
        n_evals += 1

        ts[i + 1] = ts[i] + h
        qs[i + 1] = q
        ps[i + 1] = p

    # Combine q and p into state vector
    ys = np.hstack([qs, ps])

    log.debug(f"Störmer-Verlet: {n_steps} steps, {n_evals} force evaluations")
    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        method="stormer_verlet",
    )


def yoshida4(
    acceleration_fn: Callable[[State], State],
    q0: State,
    p0: State,
    t_span: tuple[float, float],
    h: float,
) -> ODESolution:
    """Yoshida 4th-order symplectic integrator.

    Reference: Yoshida, H. "Construction of higher order symplectic integrators"
               Physics Letters A, 150(5-7), 262-268 (1990)

    Composes three Störmer-Verlet steps with specially chosen substep sizes:
        w₁ = w₃ = 1 / (2 - 2^(1/3))  ≈ 1.3512
        w₂ = -2^(1/3) / (2 - 2^(1/3))  ≈ -1.7024
        w₀ = (not used, derived from w₁+w₂+w₃ = 1)

    The integrator performs:
        Verlet(w₁·h) → Verlet(w₂·h) → Verlet(w₃·h)

    Properties:
        - Symplectic: preserves Hamiltonian structure
        - 4th-order: O(h⁴) global error
        - No energy drift over long integrations

    Application in Kalachakra:
        Higher-accuracy orbital mechanics than Störmer-Verlet while
        maintaining symplecticity for million-year integrations.

    Args:
        acceleration_fn: F(q) → acceleration = -∇V/m.
        q0: Initial position.
        p0: Initial momentum.
        t_span: Integration interval.
        h: Step size.

    Returns:
        ODESolution with positions and momenta.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / h))
    n_dim = len(q0)

    # Yoshida 4th-order coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w2 = -cbrt2 / (2.0 - cbrt2)
    w3 = w1
    weights = [w1, w2, w3]

    ts = np.zeros(n_steps + 1)
    qs = np.zeros((n_steps + 1, n_dim))
    ps = np.zeros((n_steps + 1, n_dim))

    ts[0] = t_start
    qs[0] = q0
    ps[0] = p0

    q = q0.copy()
    p = p0.copy()
    n_evals = 0

    for i in range(n_steps):
        for w in weights:
            sub_h = w * h

            # Half-step kick
            a = acceleration_fn(q)
            p = p + 0.5 * sub_h * a
            n_evals += 1

            # Full drift
            q = q + sub_h * p

            # Half-step kick
            a = acceleration_fn(q)
            p = p + 0.5 * sub_h * a
            n_evals += 1

        ts[i + 1] = ts[i] + h
        qs[i + 1] = q
        ps[i + 1] = p

    ys = np.hstack([qs, ps])

    log.debug(f"Yoshida4: {n_steps} steps, {n_evals} force evaluations")
    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        method="yoshida4",
    )


# =============================================================================
# Stiff ODE Solvers
# =============================================================================


def backward_euler(
    f: ODEFunc,
    y0: State,
    t_span: tuple[float, float],
    h: float,
    jac: Callable[[float, State], NDArray[np.floating]] | None = None,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 50,
) -> ODESolution:
    """Backward (Implicit) Euler method for stiff ODEs.

    Reference: Hairer & Wanner, "Solving Ordinary Differential Equations II" (1996)

    Formula:
        yₙ₊₁ = yₙ + h · f(tₙ₊₁, yₙ₊₁)

    This is an implicit method requiring solution of a nonlinear system
    at each step. Solved via Newton's method:

        g(y) = y - yₙ - h·f(tₙ₊₁, y) = 0
        J_g = I - h·J_f
        y^(k+1) = y^(k) - J_g⁻¹ g(y^(k))

    Properties:
        - A-stable: stable for all λh with Re(λ) < 0
        - L-stable: |R(∞)| = 0
        - 1st order accurate: O(h)
        - Suitable for stiff systems

    Args:
        f: RHS function dy/dt = f(t, y).
        y0: Initial state.
        t_span: Integration interval.
        h: Step size.
        jac: Jacobian ∂f/∂y. If None, uses finite differences.
        newton_tol: Convergence tolerance for Newton iteration.
        newton_max_iter: Max Newton iterations per step.

    Returns:
        ODESolution.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / h))
    n_dim = len(y0)

    ts = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, n_dim))
    ts[0] = t_start
    ys[0] = y0

    n_evals = 0

    for i in range(n_steps):
        t_new = ts[i] + h
        y_guess = ys[i] + h * f(ts[i], ys[i])  # Forward Euler as initial guess
        n_evals += 1

        # Newton iteration to solve: g(y) = y - yₙ - h·f(t_{n+1}, y) = 0
        y_k = y_guess.copy()
        for _ in range(newton_max_iter):
            f_val = f(t_new, y_k)
            n_evals += 1
            g = y_k - ys[i] - h * f_val

            if np.linalg.norm(g) < newton_tol:
                break

            # Jacobian (finite differences if not provided)
            if jac is not None:
                J_f = jac(t_new, y_k)
            else:
                J_f = _finite_diff_jacobian(f, t_new, y_k)
                n_evals += n_dim

            J_g = np.eye(n_dim) - h * J_f
            delta = np.linalg.solve(J_g, -g)
            y_k = y_k + delta

        ts[i + 1] = t_new
        ys[i + 1] = y_k

    return ODESolution(
        t=ts, y=ys, n_evaluations=n_evals, n_steps=n_steps,
        method="backward_euler",
    )


def _finite_diff_jacobian(
    f: ODEFunc,
    t: float,
    y: State,
    eps: float = 1e-8,
) -> NDArray[np.floating]:
    """Compute Jacobian ∂f/∂y via central finite differences.

    Formula:
        J_ij ≈ [f_i(y + εeⱼ) - f_i(y - εeⱼ)] / (2ε)

    Args:
        f: Function f(t, y).
        t: Current time.
        y: Current state.
        eps: Perturbation size.

    Returns:
        Jacobian matrix of shape (n, n).
    """
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(t, y)

    for j in range(n):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += eps
        y_minus[j] -= eps
        J[:, j] = (f(t, y_plus) - f(t, y_minus)) / (2 * eps)

    return J
