"""
Domain 1: Mathematical Foundations — Calculus

Research References:
    - Nocedal & Wright, "Numerical Optimization" (2006), Ch. 8 (finite diffs)
    - Griewank & Walther, "Evaluating Derivatives" (2008) (autodiff theory)
    - Davis & Rabinowitz, "Methods of Numerical Integration" (2007)
    - Press et al., "Numerical Recipes" (3rd ed.), Ch. 4 (integration)

Provides numerical differentiation (Jacobians, Hessians, gradients),
numerical integration (quadrature), and auto-differentiation utilities.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import integrate as sp_integrate

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Numerical Differentiation
# =============================================================================


def gradient(
    f: Callable[[NDArray[np.floating]], float],
    x: NDArray[np.floating],
    method: str = "central",
    eps: float = 1e-7,
) -> NDArray[np.floating]:
    """Compute the gradient ∇f(x) numerically.

    Reference: Nocedal & Wright (2006), Section 8.1

    Methods:
        Forward differences (1st order, O(h)):
            ∂f/∂xᵢ ≈ [f(x + h·eᵢ) - f(x)] / h

        Central differences (2nd order, O(h²)):
            ∂f/∂xᵢ ≈ [f(x + h·eᵢ) - f(x - h·eᵢ)] / (2h)

        Complex step (machine precision, O(ε)):
            ∂f/∂xᵢ ≈ Im[f(x + ih·eᵢ)] / h
            (requires f to accept complex arguments)

    Args:
        f: Scalar function f: ℝⁿ → ℝ.
        x: Point at which to compute gradient.
        method: "forward", "central", or "complex".
        eps: Step size h.

    Returns:
        Gradient vector ∇f(x) of shape (n,).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    grad = np.zeros(n)

    if method == "forward":
        f0 = f(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (f(x_plus) - f0) / eps

    elif method == "central":
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    elif method == "complex":
        # Complex step derivative — no cancellation error
        for i in range(n):
            x_complex = x.astype(np.complex128)
            x_complex[i] += 1j * eps
            grad[i] = np.imag(f(x_complex)) / eps

    else:
        raise ValueError(f"Unknown method: {method}")

    return grad


def jacobian(
    f: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    x: NDArray[np.floating],
    eps: float = 1e-7,
) -> NDArray[np.floating]:
    """Compute the Jacobian matrix J(x).

    Reference: Nocedal & Wright (2006), Section 8.1

    Formula (central differences):
        J_ij = ∂fᵢ/∂xⱼ ≈ [fᵢ(x + h·eⱼ) - fᵢ(x - h·eⱼ)] / (2h)

    The Jacobian J ∈ ℝ^{m×n} maps:
        f: ℝⁿ → ℝᵐ
        J_ij = ∂fᵢ/∂xⱼ

    Application in Kalachakra:
        Compute sensitivity of planetary positions to initial conditions
        (used in orbit propagation uncertainty quantification).

    Args:
        f: Vector function f: ℝⁿ → ℝᵐ.
        x: Point at which to compute Jacobian.
        eps: Step size.

    Returns:
        Jacobian matrix of shape (m, n).
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = f(x)
    m = len(f0)
    n = len(x)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return J


def hessian(
    f: Callable[[NDArray[np.floating]], float],
    x: NDArray[np.floating],
    eps: float = 1e-5,
) -> NDArray[np.floating]:
    """Compute the Hessian matrix H(x) = ∇²f(x).

    Reference: Nocedal & Wright (2006), Section 8.1

    Formula (central differences):
        Diagonal:
            H_ii = [f(x+h·eᵢ) - 2f(x) + f(x-h·eᵢ)] / h²

        Off-diagonal:
            H_ij = [f(x+h·eᵢ+h·eⱼ) - f(x+h·eᵢ-h·eⱼ)
                   - f(x-h·eᵢ+h·eⱼ) + f(x-h·eᵢ-h·eⱼ)] / (4h²)

    The Hessian is a symmetric n×n matrix of second partial derivatives.
    At a critical point:
        H positive definite → local minimum
        H negative definite → local maximum
        H indefinite → saddle point

    Args:
        f: Scalar function f: ℝⁿ → ℝ.
        x: Point at which to compute Hessian.
        eps: Step size h.

    Returns:
        Hessian matrix of shape (n, n).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)

    for i in range(n):
        # Diagonal elements
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        H[i, i] = (f(x_plus) - 2 * f0 + f(x_minus)) / eps**2

        # Off-diagonal elements (symmetric)
        for j in range(i + 1, n):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm[i] += eps
            xpm[j] -= eps
            xmp[i] -= eps
            xmp[j] += eps
            xmm[i] -= eps
            xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
            H[j, i] = H[i, j]

    return H


def directional_derivative(
    f: Callable[[NDArray[np.floating]], float],
    x: NDArray[np.floating],
    direction: NDArray[np.floating],
    eps: float = 1e-7,
) -> float:
    """Compute the directional derivative D_v f(x).

    Formula:
        D_v f(x) = lim_{h→0} [f(x + hv) - f(x)] / h
                 = ∇f(x) · v̂

    where v̂ = v/‖v‖ is the unit direction vector.

    Args:
        f: Scalar function.
        x: Point.
        direction: Direction vector v (will be normalized).
        eps: Step size.

    Returns:
        Directional derivative (scalar).
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    v_hat = v / np.linalg.norm(v)

    return float((f(x + eps * v_hat) - f(x - eps * v_hat)) / (2 * eps))


# =============================================================================
# Numerical Integration
# =============================================================================


def integrate_1d(
    f: Callable[[float], float],
    a: float,
    b: float,
    method: str = "adaptive",
    n_points: int = 100,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """Numerical integration ∫ₐᵇ f(x) dx.

    Methods:
        - "trapezoidal": Trapezoidal rule — O(h²)
            ∫ₐᵇ f(x)dx ≈ h/2 · [f(a) + 2Σf(xᵢ) + f(b)]

        - "simpson": Simpson's 1/3 rule — O(h⁴)
            ∫ₐᵇ f(x)dx ≈ h/3 · [f(a) + 4Σf(x_{odd}) + 2Σf(x_{even}) + f(b)]

        - "gauss": Gaussian quadrature — exact for polynomials up to degree 2n-1
            ∫₋₁¹ f(x)dx ≈ Σᵢ wᵢ f(xᵢ)
            where (xᵢ, wᵢ) are Gauss-Legendre nodes and weights

        - "adaptive": Adaptive quadrature (QUADPACK, Piessens et al. 1983)
            Recursively subdivides intervals where error is large.

    Args:
        f: Integrand function.
        a: Lower limit.
        b: Upper limit.
        method: Integration method.
        n_points: Number of quadrature points.
        tol: Error tolerance (for adaptive method).

    Returns:
        Tuple of (integral value, error estimate).
    """
    if method == "trapezoidal":
        x = np.linspace(a, b, n_points)
        y = np.array([f(xi) for xi in x])
        result = float(np.trapz(y, x))
        # Error estimate: O(h²) ≈ (b-a)³/(12n²) · max|f''|
        return result, abs(result) * 1e-8  # Rough estimate

    elif method == "simpson":
        if n_points % 2 == 0:
            n_points += 1
        x = np.linspace(a, b, n_points)
        y = np.array([f(xi) for xi in x])
        result = float(sp_integrate.simpson(y, x=x))
        return result, abs(result) * 1e-12

    elif method == "gauss":
        # Gauss-Legendre quadrature
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        # Transform from [-1, 1] to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        w = 0.5 * (b - a) * weights
        result = float(np.sum(w * np.array([f(xi) for xi in x])))
        return result, abs(result) * 1e-14

    elif method == "adaptive":
        result, error = sp_integrate.quad(f, a, b, epsabs=tol, epsrel=tol)
        return float(result), float(error)

    else:
        raise ValueError(f"Unknown method: {method}")


def integrate_2d(
    f: Callable[[float, float], float],
    x_range: tuple[float, float],
    y_range: tuple[float, float] | Callable[[float], tuple[float, float]],
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Double integral ∫∫ f(x, y) dy dx.

    If y_range is a function of x, computes:
        ∫_{x₀}^{x₁} ∫_{g(x)}^{h(x)} f(x, y) dy dx

    Uses QUADPACK adaptive quadrature (dblquad).

    Args:
        f: Integrand f(y, x) — note: scipy convention is f(y, x).
        x_range: (x_min, x_max).
        y_range: (y_min, y_max) or function x → (y_min(x), y_max(x)).
        tol: Error tolerance.

    Returns:
        Tuple of (integral value, error estimate).
    """
    if callable(y_range):
        gfun = lambda x: y_range(x)[0]
        hfun = lambda x: y_range(x)[1]
    else:
        gfun = lambda x: y_range[0]
        hfun = lambda x: y_range[1]

    result, error = sp_integrate.dblquad(
        f, x_range[0], x_range[1], gfun, hfun,
        epsabs=tol, epsrel=tol,
    )
    return float(result), float(error)


def monte_carlo_integrate(
    f: Callable[[NDArray[np.floating]], float],
    bounds: list[tuple[float, float]],
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo integration in arbitrary dimensions.

    Reference: Press et al., "Numerical Recipes", Ch. 7

    Formula:
        I = V · (1/N) Σᵢ f(xᵢ)

    where V = Π(bⱼ - aⱼ) is the volume of the integration region,
    and xᵢ are uniformly distributed random points.

    Error estimate (CLT):
        σ_I ≈ V · σ_f / √N

    Converges as O(1/√N) regardless of dimension — advantageous for
    high-dimensional integrals where quadrature rules become intractable.

    Args:
        f: Integrand f: ℝᵈ → ℝ.
        bounds: Integration bounds [(a₁,b₁), ..., (aₐ,bₐ)].
        n_samples: Number of random samples.
        seed: Random seed.

    Returns:
        Tuple of (integral estimate, standard error).
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    bounds_arr = np.array(bounds)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]

    # Volume of the integration region
    volume = float(np.prod(hi - lo))

    # Sample uniformly in the hypercube
    samples = rng.uniform(lo, hi, (n_samples, d))
    values = np.array([f(s) for s in samples])

    # Estimate
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    integral = volume * mean
    se = volume * std / np.sqrt(n_samples)

    return integral, se
