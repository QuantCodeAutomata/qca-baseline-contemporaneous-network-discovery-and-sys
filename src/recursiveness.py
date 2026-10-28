"""
Recursiveness diagnostics: ICA-Cholesky distance and backward-mass measures.

Implements diagnostics from the paper's appendix to evaluate whether the
standardized data are consistent with a recursive acyclic contemporaneous structure.
"""

import logging
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

logger = logging.getLogger(__name__)


def cholesky_factor(
    data: np.ndarray,
    causal_order: list[int],
) -> np.ndarray:
    """Compute lower Cholesky factor of the covariance matrix in causal order.

    Reorders columns according to causal_order, computes sample covariance,
    then returns the lower-triangular Cholesky factor D such that D @ D.T = Sigma.

    Parameters
    ----------
    data         : (T, n) data matrix
    causal_order : list of variable indices defining the ordering

    Returns
    -------
    (n, n) lower-triangular Cholesky factor
    """
    data_ordered = data[:, causal_order]
    cov = np.cov(data_ordered.T, ddof=1)
    # Ensure positive definiteness
    cov = (cov + cov.T) / 2
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 1e-10:
        cov += (abs(min_eig) + 1e-8) * np.eye(cov.shape[0])
    D = np.linalg.cholesky(cov)
    return D


def run_fastica(
    data: np.ndarray,
    n_init: int = 10,
    seed: int = 1121,
    tol: float = 1e-5,
    max_iter: int = 4000,
) -> list[np.ndarray]:
    """Run FastICA with multiple random initializations.

    Settings match the paper's appendix: log-cosh contrast, symmetric decorrelation.

    Parameters
    ----------
    data    : (T, n) data matrix
    n_init  : number of random initializations
    seed    : base random seed (incremented per initialization)
    tol     : convergence tolerance
    max_iter: maximum iterations

    Returns
    -------
    list of n_init (n, n) mixing matrices
    """
    n = data.shape[1]
    mixing_matrices = []

    for init in range(n_init):
        try:
            ica = FastICA(
                n_components=n,
                fun="logcosh",
                algorithm="parallel",
                tol=tol,
                max_iter=max_iter,
                random_state=seed + init,
                whiten="unit-variance",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica.fit(data)
            mixing_matrices.append(ica.mixing_)
        except Exception as exc:
            logger.warning("FastICA init %d failed: %s", init, exc)
            mixing_matrices.append(None)

    return [m for m in mixing_matrices if m is not None]


def align_ica_to_cholesky(
    ica_mixing: np.ndarray,
    chol_factor: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find closest signed-permutation alignment: min_P ||ica_mixing @ P - chol_factor||_F.

    Solves the linear assignment problem on |ica_mixing.T @ chol_factor|,
    then optimizes sign flips to further minimize the Frobenius distance.

    Parameters
    ----------
    ica_mixing  : (n, n) ICA mixing matrix
    chol_factor : (n, n) Cholesky factor

    Returns
    -------
    (aligned_matrix, frobenius_distance)
    """
    # Cost matrix for assignment: maximize |col alignment|
    cost = -(np.abs(ica_mixing.T @ chol_factor))
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build permutation matrix
    n = ica_mixing.shape[0]
    P = np.zeros((n, n))
    for r, c in zip(row_ind, col_ind):
        P[r, c] = 1.0

    # Permuted mixing
    aligned = ica_mixing @ P

    # Optimize signs: for each column, choose sign to minimize residual
    for c in range(n):
        diff_pos = np.linalg.norm(aligned[:, c] - chol_factor[:, c])
        diff_neg = np.linalg.norm(-aligned[:, c] - chol_factor[:, c])
        if diff_neg < diff_pos:
            aligned[:, c] *= -1

    dist = float(np.linalg.norm(aligned - chol_factor, "fro"))
    return aligned, dist


def random_rotation_reference(
    ica_mixing: np.ndarray,
    chol_factor: np.ndarray,
    n_rotations: int = 10000,
    seed: int = 1121,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rotation-based reference distributions for ICA-Cholesky distance.

    Generates random orthogonal matrices via QR decomposition of Gaussian matrices.

    Parameters
    ----------
    ica_mixing  : (n, n) ICA mixing matrix
    chol_factor : (n, n) Cholesky factor
    n_rotations : number of random rotations
    seed        : random seed

    Returns
    -------
    (dist_rotate_ica, dist_rotate_chol) each of shape (n_rotations,)
    """
    rng = np.random.RandomState(seed)
    n = ica_mixing.shape[0]

    dist_ica  = np.empty(n_rotations)
    dist_chol = np.empty(n_rotations)

    for r in range(n_rotations):
        G = rng.randn(n, n)
        Q, _ = np.linalg.qr(G)
        # Ensure orthogonal (handle sign)
        dist_ica[r]  = float(np.linalg.norm(ica_mixing @ Q - chol_factor,  "fro"))
        dist_chol[r] = float(np.linalg.norm(chol_factor  @ Q - ica_mixing, "fro"))

    return dist_ica, dist_chol


def ilmonen_distance(
    ica_mixing: np.ndarray,
    chol_factor: np.ndarray,
) -> float:
    """Compute Ilmonen et al. minimum-distance index.

    Normalizes columns of each matrix to unit norm, then finds minimum
    matching distance via linear assignment.

    Parameters
    ----------
    ica_mixing  : (n, n) ICA mixing matrix
    chol_factor : (n, n) Cholesky factor

    Returns
    -------
    Scalar minimum matching distance
    """
    n = ica_mixing.shape[0]

    # Normalize columns
    def normalize_cols(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return M / norms

    A = normalize_cols(ica_mixing.copy())
    B = normalize_cols(chol_factor.copy())

    # Cost: 1 - |correlation| between columns
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1.0 - abs(float(A[:, i] @ B[:, j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


# ── Backward-mass statistic ─────────────────────────────────────────────────────

def backward_mass(
    W: np.ndarray,
    causal_order: list[int],
) -> float:
    """Compute backward-mass statistic R(W, r).

    R(W, r) = [sum_{i!=j, r(i) < r(j)} W_{ij}^2] / [sum_{i!=j} W_{ij}^2]

    where r(i) is the position of variable i in causal_order.

    A lower backward mass means more off-diagonal mass lies in the 'forward'
    direction relative to the estimated causal order.

    Parameters
    ----------
    W            : (n, n) unmixing weight matrix
    causal_order : list of variable indices in causal order

    Returns
    -------
    Backward-mass ratio (float in [0, 1])
    """
    n = W.shape[0]
    # rank[i] = position of variable i in causal_order
    rank = np.empty(n, dtype=int)
    for pos, var in enumerate(causal_order):
        rank[var] = pos

    numerator   = 0.0
    denominator = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val_sq = float(W[i, j] ** 2)
            denominator += val_sq
            # "backward": r(i) < r(j) means i comes before j in order,
            # so W[i,j] represents a backward edge
            if rank[i] < rank[j]:
                numerator += val_sq

    if denominator < 1e-14:
        return np.nan
    return numerator / denominator


def random_order_backward_mass(
    W: np.ndarray,
    n_random: int = 5000,
    seed: int = 1121,
) -> np.ndarray:
    """Compute backward-mass for n_random random orderings of variables.

    Parameters
    ----------
    W        : (n, n) unmixing weight matrix
    n_random : number of random orderings
    seed     : random seed

    Returns
    -------
    Array of shape (n_random,) with backward-mass values
    """
    rng = np.random.RandomState(seed)
    n = W.shape[0]
    bm_values = np.empty(n_random)

    for r in range(n_random):
        order = list(rng.permutation(n))
        bm_values[r] = backward_mass(W, order)

    return bm_values


def acyclic_reference_backward_mass(
    n_vars: int = 52,
    n_systems: int = 20,
    T: int = 1990,
    seed: int = 1121,
) -> np.ndarray:
    """Generate n_systems acyclic reference systems and compute their backward-mass.

    Each system uses a sparse lower-triangular true coefficient matrix with
    Student-t innovations. FastICA is fitted and backward-mass computed
    under the true ordering.

    Parameters
    ----------
    n_vars    : number of variables
    n_systems : number of reference systems to generate
    T         : sample size per system
    seed      : random seed

    Returns
    -------
    Array of shape (n_systems,) with backward-mass values
    """
    rng = np.random.RandomState(seed)
    bm_values = []

    for sys_idx in range(n_systems):
        # Generate sparse lower-triangular coefficient matrix
        B = np.zeros((n_vars, n_vars))
        for i in range(1, n_vars):
            for j in range(i):
                if rng.rand() < 0.15:  # ~15% sparsity
                    B[i, j] = rng.uniform(0.1, 0.4) * rng.choice([-1, 1])

        # Generate t-distributed shocks
        nu = 5.0
        eps = rng.standard_t(nu, size=(T, n_vars))

        # Generate data: X = (I - B)^{-1} eps (recursive ordering)
        I_minus_B = np.eye(n_vars) - B
        try:
            data = np.linalg.solve(I_minus_B.T, eps.T).T
        except np.linalg.LinAlgError:
            data = eps.copy()

        # Fit FastICA
        try:
            ica = FastICA(
                n_components=min(n_vars, 10),  # reduce for speed
                fun="logcosh",
                algorithm="parallel",
                tol=1e-4,
                max_iter=1000,
                random_state=seed + sys_idx,
                whiten="unit-variance",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica.fit(data)
            W_ica = ica.components_  # unmixing matrix (n_components x n_vars)
            # Pad to square if needed
            if W_ica.shape[0] < n_vars:
                W_sq = np.zeros((n_vars, n_vars))
                W_sq[:W_ica.shape[0], :] = W_ica
                W_ica = W_sq
        except Exception:
            W_ica = np.eye(n_vars)

        # True causal order: variables ordered as 0, 1, ..., n_vars-1
        true_order = list(range(n_vars))
        bm = backward_mass(W_ica, true_order)
        if np.isfinite(bm):
            bm_values.append(bm)

    return np.array(bm_values)
