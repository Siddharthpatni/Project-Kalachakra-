"""
Domain 1: Mathematical Foundations — Linear Algebra

Research References:
    - Strang, G. "Introduction to Linear Algebra" (5th ed.)
    - Golub & Van Loan, "Matrix Computations" (4th ed.)
    - Trefethen & Bau, "Numerical Linear Algebra" (1997)

Provides matrix decompositions, spectral analysis, and tensor operations
used throughout the astronomical and ML pipeline.

All implementations wrap NumPy/SciPy with domain-specific validation,
detailed docstrings containing exact mathematical formulas, and
type-safe interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg

from kalachakra.core.logging import get_logger

log = get_logger(__name__)

# Type alias for real-valued arrays
RealArray = NDArray[np.floating]


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class SVDResult:
    """Result of Singular Value Decomposition A = U Σ Vᵀ.

    Reference: Golub & Van Loan, "Matrix Computations", Ch. 2.5

    Mathematical Definition:
        For any m×n matrix A of rank r:
            A = U Σ Vᵀ

        where:
            U  ∈ ℝ^{m×m}  — orthogonal, columns are left singular vectors
                             (eigenvectors of AAᵀ)
            Σ  ∈ ℝ^{m×n}  — diagonal, σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0
                             (square roots of eigenvalues of AᵀA)
            Vᵀ ∈ ℝ^{n×n}  — orthogonal, rows are right singular vectors
                             (eigenvectors of AᵀA)
    """

    U: RealArray
    sigma: RealArray  # 1-D array of singular values
    Vt: RealArray
    rank: int
    condition_number: float

    def reconstruct(self, k: int | None = None) -> RealArray:
        """Reconstruct matrix from top-k singular values (low-rank approximation).

        By the Eckart–Young–Mirsky theorem, the best rank-k approximation
        (in Frobenius or spectral norm) is:
            A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ

        Args:
            k: Number of singular values to use. None = use all (exact).

        Returns:
            Reconstructed matrix of shape (m, n).
        """
        if k is None:
            k = len(self.sigma)
        k = min(k, len(self.sigma))
        return (self.U[:, :k] * self.sigma[:k]) @ self.Vt[:k, :]

    def explained_variance_ratio(self) -> RealArray:
        """Fraction of total variance explained by each singular value.

        Formula:
            ratio_i = σᵢ² / Σⱼ σⱼ²

        Returns:
            Array of explained variance ratios summing to 1.0.
        """
        s_squared = self.sigma**2
        total = s_squared.sum()
        if total == 0:
            return np.zeros_like(s_squared)
        return s_squared / total


@dataclass(frozen=True, slots=True)
class QRResult:
    """Result of QR Decomposition A = QR.

    Reference: Trefethen & Bau, "Numerical Linear Algebra", Lecture 7-8

    Mathematical Definition:
        For m×n matrix A (m ≥ n):
            A = QR

        where:
            Q ∈ ℝ^{m×m}  — orthogonal (QᵀQ = I)
            R ∈ ℝ^{m×n}  — upper triangular

    Methods:
        - Householder reflections: H = I - 2vvᵀ/(vᵀv)
          Numerically stable, O(2mn² - 2n³/3) flops
        - Gram-Schmidt: Conceptually simple but numerically unstable
        - Modified Gram-Schmidt: Better stability than classical
        - Givens rotations: For sparse matrices
    """

    Q: RealArray
    R: RealArray


@dataclass(frozen=True, slots=True)
class LUResult:
    """Result of LU Decomposition PA = LU.

    Reference: Strang, "Introduction to Linear Algebra", Ch. 2

    Mathematical Definition:
        For square n×n matrix A:
            PA = LU

        where:
            P ∈ ℝ^{n×n}  — permutation matrix (for partial pivoting)
            L ∈ ℝ^{n×n}  — lower triangular with 1s on diagonal
            U ∈ ℝ^{n×n}  — upper triangular

    Partial pivoting ensures numerical stability by selecting the largest
    absolute value in each column as the pivot element.
    """

    P: RealArray  # Permutation matrix
    L: RealArray  # Lower triangular
    U: RealArray  # Upper triangular
    pivots: NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class EigenResult:
    """Result of Eigendecomposition A = VΛV⁻¹.

    Reference: Strang, "Introduction to Linear Algebra", Ch. 6

    Mathematical Definition:
        For square matrix A:
            Av = λv

        where λ are eigenvalues and v are eigenvectors.

        For symmetric A (Spectral Theorem):
            A = QΛQᵀ

        where Q is orthogonal (eigenvectors) and Λ is diagonal (eigenvalues).
    """

    eigenvalues: NDArray[np.complexfloating] | RealArray
    eigenvectors: NDArray[np.complexfloating] | RealArray
    is_symmetric: bool


# =============================================================================
# Decomposition Functions
# =============================================================================


def svd(
    A: RealArray,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> SVDResult:
    """Compute the Singular Value Decomposition of matrix A.

    Decomposes A into A = U Σ Vᵀ using the Golub-Kahan bidiagonalization
    algorithm (via LAPACK's dgesdd).

    Formula:
        A = U Σ Vᵀ

        Step 1: Compute AᵀA, find eigenvalues λᵢ
        Step 2: σᵢ = √λᵢ (singular values)
        Step 3: vᵢ = eigenvectors of AᵀA (right singular vectors)
        Step 4: uᵢ = (1/σᵢ) A vᵢ (left singular vectors)

    Args:
        A: Input matrix of shape (m, n).
        full_matrices: If True, U is m×m and Vt is n×n.
                       If False, U is m×k and Vt is k×n where k=min(m,n).
        compute_uv: If True, compute U and Vt. If False, only singular values.

    Returns:
        SVDResult containing U, sigma, Vt, rank, and condition number.

    Example:
        >>> A = np.array([[1, 2], [3, 4], [5, 6]])
        >>> result = svd(A)
        >>> np.allclose(A, result.reconstruct())
        True
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_matrix(A, "A")

    U, sigma, Vt = np.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)

    # Numerical rank: count singular values above machine epsilon threshold
    tol = max(A.shape) * sigma[0] * np.finfo(A.dtype).eps if len(sigma) > 0 else 0.0
    rank = int(np.sum(sigma > tol))

    # Condition number: κ(A) = σ_max / σ_min
    cond = float(sigma[0] / sigma[-1]) if sigma[-1] > 0 else float("inf")

    log.debug(f"SVD: shape={A.shape}, rank={rank}, κ(A)={cond:.2e}")

    return SVDResult(U=U, sigma=sigma, Vt=Vt, rank=rank, condition_number=cond)


def qr_decomposition(
    A: RealArray,
    mode: Literal["reduced", "complete"] = "reduced",
    pivoting: bool = False,
) -> QRResult:
    """Compute the QR Decomposition of matrix A.

    Uses Householder reflections (LAPACK's dgeqrf):
        H_k = I - 2 vₖvₖᵀ / (vₖᵀvₖ)

    The matrix Q is the product of Householder reflectors:
        Q = H₁ H₂ ... H_n

    Complexity: O(2mn² − 2n³/3) flops.

    Args:
        A: Input matrix of shape (m, n) with m ≥ n.
        mode: "reduced" returns economy-size Q,R. "complete" returns full.
        pivoting: If True, use column pivoting for rank-revealing QR.

    Returns:
        QRResult containing orthogonal Q and upper triangular R.
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_matrix(A, "A")

    if pivoting:
        Q, R, _ = sp_linalg.qr(A, pivoting=True, mode="economic" if mode == "reduced" else "full")
    else:
        Q, R = np.linalg.qr(A, mode=mode)

    log.debug(f"QR: shape={A.shape}, mode={mode}, pivoting={pivoting}")
    return QRResult(Q=Q, R=R)


def lu_decomposition(A: RealArray) -> LUResult:
    """Compute the LU Decomposition with partial pivoting: PA = LU.

    Algorithm (Gaussian elimination with partial pivoting):
        For each column k:
            1. Find pivot: p = argmax|A[k:, k]|
            2. Swap rows k and p
            3. Compute multipliers: L[i, k] = A[i, k] / A[k, k]
            4. Eliminate below: A[i, j] -= L[i, k] * A[k, j]

    Complexity: O(2n³/3) flops.

    Args:
        A: Square input matrix of shape (n, n).

    Returns:
        LUResult containing P (permutation), L (lower), U (upper), pivots.

    Raises:
        ValueError: If A is not square.
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_square(A, "A")

    P, L, U = sp_linalg.lu(A)
    pivots = np.argmax(P, axis=1)

    log.debug(f"LU: shape={A.shape}")
    return LUResult(P=P, L=L, U=U, pivots=pivots)


def eigendecomposition(
    A: RealArray,
    symmetric: bool | None = None,
) -> EigenResult:
    """Compute the Eigendecomposition of matrix A.

    For general A:
        Av = λv  →  eigenvalues λ, eigenvectors v

    For symmetric A (Spectral Theorem):
        A = QΛQᵀ
        where Q is orthogonal, Λ is real diagonal

    Uses:
        - dsyev (symmetric): QR algorithm with implicit shifts
        - dgeev (general): Hessenberg reduction → QR iteration

    Args:
        A: Square input matrix of shape (n, n).
        symmetric: If True, use symmetric solver (faster, real eigenvalues).
                   If None, auto-detect symmetry.

    Returns:
        EigenResult containing eigenvalues, eigenvectors, and symmetry flag.
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_square(A, "A")

    if symmetric is None:
        symmetric = bool(np.allclose(A, A.T, rtol=1e-10))

    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Sort by descending eigenvalue magnitude
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    log.debug(
        f"Eigen: shape={A.shape}, symmetric={symmetric}, "
        f"λ_max={np.max(np.abs(eigenvalues)):.4e}"
    )
    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        is_symmetric=symmetric,
    )


# =============================================================================
# Matrix Properties & Operations
# =============================================================================


def condition_number(
    A: RealArray,
    p: int | float | str = 2,
) -> float:
    """Compute the condition number of matrix A.

    Formula:
        κ_p(A) = ‖A‖_p · ‖A⁻¹‖_p

    For p=2 (spectral norm):
        κ₂(A) = σ_max / σ_min

    A large condition number indicates the matrix is nearly singular
    and solutions to Ax=b will be sensitive to perturbations.

    Args:
        A: Input matrix.
        p: Norm order. Common values: 1, 2, inf, 'fro'.

    Returns:
        Condition number (≥ 1 for non-singular, inf for singular).
    """
    A = np.asarray(A, dtype=np.float64)
    return float(np.linalg.cond(A, p=p))


def matrix_rank(
    A: RealArray,
    tol: float | None = None,
) -> int:
    """Compute the numerical rank of matrix A via SVD.

    Formula:
        rank(A) = #{σᵢ : σᵢ > tol}
        where tol = max(m,n) · σ₁ · ε_machine

    Args:
        A: Input matrix.
        tol: Tolerance for singular value threshold. None = auto.

    Returns:
        Numerical rank.
    """
    A = np.asarray(A, dtype=np.float64)
    return int(np.linalg.matrix_rank(A, tol=tol))


def pseudo_inverse(
    A: RealArray,
    rcond: float = 1e-15,
) -> RealArray:
    """Compute the Moore-Penrose pseudo-inverse A⁺.

    Formula (via SVD):
        A = UΣVᵀ  →  A⁺ = VΣ⁺Uᵀ

        where Σ⁺ has entries 1/σᵢ for σᵢ > rcond·σ_max, else 0.

    Properties:
        - AA⁺A = A
        - A⁺AA⁺ = A⁺
        - (AA⁺)ᵀ = AA⁺
        - (A⁺A)ᵀ = A⁺A

    Args:
        A: Input matrix.
        rcond: Cutoff for small singular values.

    Returns:
        Pseudo-inverse matrix of shape (n, m) for input (m, n).
    """
    A = np.asarray(A, dtype=np.float64)
    return np.linalg.pinv(A, rcond=rcond)


def kronecker_product(A: RealArray, B: RealArray) -> RealArray:
    """Compute the Kronecker (tensor) product A ⊗ B.

    Formula:
        If A is m×n and B is p×q, then A ⊗ B is mp×nq:

        A ⊗ B = [a₁₁B  a₁₂B  ...  a₁ₙB]
                [a₂₁B  a₂₂B  ...  a₂ₙB]
                [  ⋮     ⋮    ⋱    ⋮  ]
                [aₘ₁B  aₘ₂B  ...  aₘₙB]

    Properties:
        - (A⊗B)(C⊗D) = (AC)⊗(BD)  (mixed-product property)
        - (A⊗B)ᵀ = Aᵀ⊗Bᵀ
        - tr(A⊗B) = tr(A)·tr(B)

    Application in Kalachakra:
        Used for constructing joint state spaces from planetary
        feature tensors (e.g., planet × house × aspect).

    Args:
        A: First matrix.
        B: Second matrix.

    Returns:
        Kronecker product of shape (mp, nq).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return np.kron(A, B)


# =============================================================================
# Additional Operations
# =============================================================================


def solve_linear_system(
    A: RealArray,
    b: RealArray,
    method: Literal["lu", "qr", "svd", "cholesky"] = "lu",
) -> RealArray:
    """Solve the linear system Ax = b.

    Methods:
        - LU: PA = LU → solve Ly = Pb, then Ux = y. O(2n³/3 + 2n²)
        - QR: A = QR → Rx = Qᵀb. More stable but O(2mn²)
        - SVD: A = UΣVᵀ → x = VΣ⁺Uᵀb. Most stable, handles rank-deficient
        - Cholesky: A = LLᵀ → solve Ly = b, Lᵀx = y. For SPD matrices, O(n³/3)

    Args:
        A: Coefficient matrix (n×n for square, m×n for least squares).
        b: Right-hand side vector or matrix.
        method: Solution method.

    Returns:
        Solution vector x.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if method == "lu":
        return np.linalg.solve(A, b)
    elif method == "qr":
        Q, R = np.linalg.qr(A)
        return sp_linalg.solve_triangular(R, Q.T @ b)
    elif method == "svd":
        return pseudo_inverse(A) @ b
    elif method == "cholesky":
        L = np.linalg.cholesky(A)
        y = sp_linalg.solve_triangular(L, b, lower=True)
        return sp_linalg.solve_triangular(L.T, y, lower=False)
    else:
        raise ValueError(f"Unknown method: {method}")


def sparse_to_dense(indices: RealArray, values: RealArray, shape: tuple[int, ...]) -> RealArray:
    """Convert sparse representation (COO format) to dense matrix.

    Args:
        indices: Row and column indices, shape (2, nnz).
        values: Non-zero values, shape (nnz,).
        shape: Output matrix shape.

    Returns:
        Dense matrix.
    """
    result = np.zeros(shape, dtype=np.float64)
    rows, cols = indices[0].astype(int), indices[1].astype(int)
    result[rows, cols] = values
    return result


def gram_matrix(X: RealArray) -> RealArray:
    """Compute the Gram matrix G = XᵀX.

    The Gram matrix contains all pairwise inner products:
        G_ij = ⟨xᵢ, xⱼ⟩

    Properties:
        - Symmetric positive semi-definite
        - G_ii = ‖xᵢ‖²
        - rank(G) = rank(X)

    Application in Kalachakra:
        Compute similarity between planetary feature vectors.

    Args:
        X: Data matrix of shape (m, n) where columns are vectors.

    Returns:
        Gram matrix of shape (n, n).
    """
    X = np.asarray(X, dtype=np.float64)
    return X.T @ X


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_matrix(A: RealArray, name: str) -> None:
    """Validate that A is a 2-D matrix."""
    if A.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {A.shape}")


def _validate_square(A: RealArray, name: str) -> None:
    """Validate that A is a square matrix."""
    _validate_matrix(A, name)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
