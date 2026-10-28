"""
DirectLiNGAM estimation and bootstrap inference utilities.

Implements:
- DirectLiNGAM estimation with pairwise likelihood-ratio criterion
- Moving-block bootstrap for standard error computation
- Wald tests, BH/BY multiple testing correction
- Predecessor OLS regressions
"""

import logging
import numpy as np
import pandas as pd
import lingam
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

GLOBAL_SEED: int = 1121
BOOT_SEED: int = 731


# ── DirectLiNGAM estimation ─────────────────────────────────────────────────────

def estimate_direct_lingam(
    Z: np.ndarray,
    seed: int = GLOBAL_SEED,
) -> dict:
    """Estimate DirectLiNGAM with pairwise likelihood-ratio criterion.

    Uses lingam.DirectLiNGAM(measure='pwling') as specified by the paper
    (Hyvarinen and Smith 2013 pairwise likelihood-ratio independence criterion).

    Parameters
    ----------
    Z    : (T x n) data matrix
    seed : random seed

    Returns
    -------
    dict with keys:
        'model'            : fitted lingam.DirectLiNGAM object
        'order'            : causal ordering (list of original variable indices)
        'adjacency_matrix' : (n x n) coefficient matrix in original variable order
                             B[i,j] = contemporaneous coefficient of j → i
    """
    np.random.seed(seed)
    model = lingam.DirectLiNGAM(measure="pwling")
    model.fit(Z)
    return {
        "model": model,
        "order": list(model.causal_order_),
        "adjacency_matrix": model.adjacency_matrix_,
    }


# ── OLS predecessor regressions ─────────────────────────────────────────────────

def fit_predecessor_regressions(
    Z: np.ndarray,
    causal_order: list[int],
) -> np.ndarray:
    """Fit OLS predecessor regressions implied by a given causal ordering.

    Uses Gram-matrix approach: compute G = Z.T @ Z once, then solve small
    sub-systems for each variable in the causal order. This is ~5-10× faster
    than calling lstsq separately for each variable.

    B[i, j] = effect of j on i (i is regressed on its predecessors j).

    Parameters
    ----------
    Z            : (T x n) data matrix (should be centered)
    causal_order : list of variable indices defining causal ordering

    Returns
    -------
    (n x n) coefficient matrix in ORIGINAL variable ordering.
    B[i, j] is non-zero only if j precedes i in causal_order.
    """
    n = Z.shape[1]
    B0 = np.zeros((n, n))
    order_arr = np.array(causal_order, dtype=int)

    # Compute Gram matrix once: O(T * n^2)
    G = Z.T @ Z  # (n x n)

    for pos in range(1, n):
        target = order_arr[pos]
        preds  = order_arr[:pos]  # length pos

        G_pp = G[np.ix_(preds, preds)]  # (pos x pos)
        G_py = G[preds, target]          # (pos,)

        # Solve G_pp @ beta = G_py  (small pos×pos system)
        try:
            coefs = np.linalg.solve(G_pp, G_py)
        except np.linalg.LinAlgError:
            coefs, _, _, _ = np.linalg.lstsq(G_pp, G_py, rcond=None)

        B0[target, preds] = coefs

    return B0


# ── Moving-block bootstrap ──────────────────────────────────────────────────────

def moving_block_bootstrap_indices(
    n: int,
    n_boot: int,
    block_length: int = 10,
    seed: int = BOOT_SEED,
) -> list[np.ndarray]:
    """Generate moving-block bootstrap row indices.

    Concatenates randomly sampled contiguous blocks of `block_length` rows
    until >= n observations, then truncates to exactly n.

    Parameters
    ----------
    n            : target number of observations
    n_boot       : number of bootstrap resamples
    block_length : length of each block
    seed         : random seed

    Returns
    -------
    list of n_boot arrays, each of length n
    """
    rng = np.random.RandomState(seed)
    max_start = n - block_length + 1
    all_indices = []

    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = rng.randint(0, max_start)
            block = list(range(start, start + block_length))
            indices.extend(block)
        all_indices.append(np.array(indices[:n], dtype=int))

    return all_indices


def compute_bootstrap_se(
    Z: np.ndarray,
    causal_order: list[int],
    n_boot: int = 1000,
    block_length: int = 10,
    seed: int = BOOT_SEED,
) -> np.ndarray:
    """Compute bootstrap standard errors for all order-allowed coefficients.

    For each bootstrap resample:
    1. Sample rows of Z using moving blocks.
    2. Center each column of the resampled data (but do NOT rescale).
    3. Fit predecessor regressions with the FIXED causal order.
    4. Record the full coefficient matrix.

    SE[i, j] = std of B0[i, j] across bootstrap replicates.

    Parameters
    ----------
    Z            : (T x n) data matrix
    causal_order : fixed causal ordering from full-sample DirectLiNGAM
    n_boot       : number of bootstrap resamples
    block_length : block length for moving-block bootstrap
    seed         : random seed

    Returns
    -------
    (n x n) SE matrix; NaN at positions that are not order-allowed.
    """
    n = Z.shape[1]
    boot_indices = moving_block_bootstrap_indices(len(Z), n_boot, block_length, seed)

    # Identify order-allowed positions
    order_pos = {v: i for i, v in enumerate(causal_order)}
    allowed_positions = []
    for i in range(n):
        for j in range(n):
            if i != j and order_pos.get(j, -1) < order_pos.get(i, -1):
                allowed_positions.append((i, j))

    boot_coefs = np.full((n_boot, n, n), np.nan)

    for b, idx in enumerate(boot_indices):
        Z_boot = Z[idx].copy()
        # Center each column
        Z_boot -= Z_boot.mean(axis=0)
        B0_b = fit_predecessor_regressions(Z_boot, causal_order)
        boot_coefs[b] = B0_b

    SE = np.full((n, n), np.nan)
    for (i, j) in allowed_positions:
        coef_draws = boot_coefs[:, i, j]
        SE[i, j] = float(np.std(coef_draws, ddof=1))

    return SE


# ── Wald tests and BH/BY selection ─────────────────────────────────────────────

def wald_bh_selection(
    B0: np.ndarray,
    SE: np.ndarray,
    causal_order: list[int],
    q_bh: float = 0.05,
    materiality_threshold: float = 0.05,
) -> dict:
    """Wald tests + BH/BY correction + materiality screening.

    For each order-allowed (i, j):
        W_ij = B0[i,j]^2 / SE[i,j]^2  (chi-square(1) Wald statistic)
        p_ij = 1 - chi2(1).cdf(W_ij)

    Applies:
    - BH correction at q_bh across all 1326 tests
    - BY correction at q_bh (sensitivity)
    - Post-BH materiality: |B0[i,j]| >= materiality_threshold

    Parameters
    ----------
    B0                    : (n x n) full coefficient matrix (original order)
    SE                    : (n x n) bootstrap SE matrix
    causal_order          : recovered causal ordering
    q_bh                  : BH/BY FDR level
    materiality_threshold : post-selection coefficient size threshold

    Returns
    -------
    dict with keys: W, pvalues, bh_rejected, by_rejected,
                    B0_selected_bh, B0_selected_by,
                    n_order_allowed, n_unadj_sig, n_bh, n_bh_mat, n_by_mat,
                    order_allowed_pairs
    """
    n = B0.shape[0]
    order_pos = {v: i for i, v in enumerate(causal_order)}

    # Collect all order-allowed pairs
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and order_pos.get(j, -1) < order_pos.get(i, -1):
                pairs.append((i, j))

    n_order_allowed = len(pairs)
    assert n_order_allowed == n * (n - 1) // 2, (
        f"Expected {n*(n-1)//2} order-allowed pairs, got {n_order_allowed}"
    )

    W_vals, p_vals, ij_list = [], [], []
    for (i, j) in pairs:
        se = SE[i, j]
        b  = B0[i, j]
        if np.isnan(se) or se <= 0:
            W_ij = 0.0
            p_ij = 1.0
        else:
            W_ij = float(b ** 2 / se ** 2)
            p_ij = float(stats.chi2.sf(W_ij, df=1))
        W_vals.append(W_ij)
        p_vals.append(p_ij)
        ij_list.append((i, j))

    W_arr = np.array(W_vals)
    p_arr = np.array(p_vals)

    # BH correction
    _, p_bh, _, _ = multipletests(p_arr, alpha=q_bh, method="fdr_bh")
    bh_reject_mask = p_bh < q_bh

    # BY correction
    _, p_by, _, _ = multipletests(p_arr, alpha=q_bh, method="fdr_by")
    by_reject_mask = p_by < q_bh

    # Build full matrices
    W_mat       = np.full((n, n), np.nan)
    pval_mat    = np.full((n, n), np.nan)
    bh_mat      = np.zeros((n, n), dtype=bool)
    by_mat      = np.zeros((n, n), dtype=bool)
    B0_sel_bh   = np.zeros((n, n))
    B0_sel_by   = np.zeros((n, n))

    for k, (i, j) in enumerate(ij_list):
        W_mat[i, j]    = W_arr[k]
        pval_mat[i, j] = p_arr[k]
        bh_mat[i, j]   = bool(bh_reject_mask[k])
        by_mat[i, j]   = bool(by_reject_mask[k])
        if bh_reject_mask[k] and abs(B0[i, j]) >= materiality_threshold:
            B0_sel_bh[i, j] = B0[i, j]
        if by_reject_mask[k] and abs(B0[i, j]) >= materiality_threshold:
            B0_sel_by[i, j] = B0[i, j]

    n_unadj_sig = int(np.sum(p_arr < 0.05))
    n_bh        = int(np.sum(bh_reject_mask))
    n_bh_mat    = int(np.sum(B0_sel_bh != 0))
    n_by_mat    = int(np.sum(B0_sel_by != 0))

    print(f"[SELECTION] Order-allowed coefficients: {n_order_allowed}")
    print(f"[SELECTION] Unadjusted p<0.05: {n_unadj_sig} (target ~332)")
    print(f"[SELECTION] BH-selected: {n_bh} (target ~219)")
    print(f"[SELECTION] BH+materiality retained: {n_bh_mat} (target ~214)")
    print(f"[SELECTION] BY+materiality retained: {n_by_mat} (target ~169)")

    return {
        "W": W_mat,
        "pvalues": pval_mat,
        "bh_rejected": bh_mat,
        "by_rejected": by_mat,
        "B0_selected_bh": B0_sel_bh,
        "B0_selected_by": B0_sel_by,
        "n_order_allowed": n_order_allowed,
        "n_unadj_sig": n_unadj_sig,
        "n_bh": n_bh,
        "n_bh_mat": n_bh_mat,
        "n_by_mat": n_by_mat,
        "order_allowed_pairs": ij_list,
    }


def descriptive_edge_count(
    B0: np.ndarray,
    threshold: float = 0.05,
) -> int:
    """Count edges with |B0_ij| >= threshold (descriptive display only).

    Parameters
    ----------
    B0        : coefficient matrix
    threshold : display threshold

    Returns
    -------
    int count of descriptive edges
    """
    return int(np.sum(np.abs(B0) >= threshold))


def recover_shocks(
    Z: np.ndarray,
    B0: np.ndarray,
) -> np.ndarray:
    """Recover structural shocks: eps_hat = (I - B0) @ Z.T transposed.

    Parameters
    ----------
    Z  : (T x n) standardized data matrix
    B0 : (n x n) contemporaneous coefficient matrix

    Returns
    -------
    (T x n) recovered shock matrix
    """
    n = B0.shape[0]
    transformation = np.eye(n) - B0
    return (transformation @ Z.T).T
