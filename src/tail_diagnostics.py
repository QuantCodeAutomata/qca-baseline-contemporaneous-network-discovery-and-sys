"""
Tail dependence diagnostics: CoLift and DirLift.

Implements symmetric lower-tail lift (CoLift) and directional shock-conditioned
lower-tail lift (DirLift) for evaluating selected directed links.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tail_indicators(
    Z: np.ndarray,
    quantile: float = 0.05,
) -> np.ndarray:
    """Compute lower-tail indicators D_i,t = 1{Z_{i,t} <= q_i(quantile)}.

    Parameters
    ----------
    Z        : (T, n) data matrix
    quantile : lower tail quantile (default 0.05 = 5th percentile)

    Returns
    -------
    Binary indicator matrix (T, n) dtype=int
    """
    quantiles = np.quantile(Z, quantile, axis=0)  # (n,)
    D = (Z <= quantiles[None, :]).astype(int)
    return D


def compute_shock_indicators(
    shocks: np.ndarray,
    quantile: float = 0.05,
) -> np.ndarray:
    """Compute shock lower-tail indicators S_i,t = 1{eps_i,t <= q_i^eps(quantile)}.

    Parameters
    ----------
    shocks   : (T, n) recovered shock matrix
    quantile : lower tail quantile

    Returns
    -------
    Binary indicator matrix (T, n)
    """
    return compute_tail_indicators(shocks, quantile)


def compute_colift(
    D: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Symmetric lower-tail lift: CoLift_{ij} = P(D_i=1, D_j=1) / [P(D_i=1) * P(D_j=1)].

    Parameters
    ----------
    D : (T, n) binary indicator matrix
    i : index of first variable
    j : index of second variable

    Returns
    -------
    CoLift statistic (float); NaN if any marginal probability is zero.
    """
    p_i  = float(D[:, i].mean())
    p_j  = float(D[:, j].mean())
    p_ij = float((D[:, i] * D[:, j]).mean())
    if p_i * p_j <= 0:
        return np.nan
    return p_ij / (p_i * p_j)


def compute_dirlift(
    D: np.ndarray,
    S: np.ndarray,
    source: int,
    target: int,
) -> float:
    """Directional shock-conditioned lower-tail lift.

    DirLift_{source→target} = P(D_target=1 | S_source=1) / P(D_target=1)

    Parameters
    ----------
    D      : (T, n) return tail indicator matrix
    S      : (T, n) shock tail indicator matrix
    source : index of source variable (conditioning variable)
    target : index of target variable

    Returns
    -------
    DirLift statistic (float); NaN if marginal or conditioning prob is zero.
    """
    S_src = S[:, source].astype(bool)
    D_tgt = D[:, target]

    n_source_tail = float(S_src.sum())
    if n_source_tail <= 0:
        return np.nan

    cond_prob = float(D_tgt[S_src].mean())
    marg_prob = float(D_tgt.mean())
    if marg_prob <= 0:
        return np.nan

    return cond_prob / marg_prob


def compute_all_pair_lifts(
    Z: np.ndarray,
    shocks: np.ndarray,
    selected_edges: list[tuple[int, int]],
    quantile: float = 0.05,
) -> pd.DataFrame:
    """Compute CoLift and DirLift for all selected edges.

    Parameters
    ----------
    Z              : (T, n) standardized return matrix
    shocks         : (T, n) recovered structural shock matrix
    selected_edges : list of (source_idx, target_idx) in original variable order
    quantile       : lower-tail quantile

    Returns
    -------
    DataFrame with columns: source, target, colift, dirlift_forward,
                            dirlift_reverse, directional_gap
    """
    D = compute_tail_indicators(Z, quantile)
    S = compute_shock_indicators(shocks, quantile)

    rows = []
    for (src, tgt) in selected_edges:
        colift     = compute_colift(D, src, tgt)
        dirlift_fw = compute_dirlift(D, S, src, tgt)
        dirlift_rv = compute_dirlift(D, S, tgt, src)
        gap        = (dirlift_fw - dirlift_rv) if (
            np.isfinite(dirlift_fw) and np.isfinite(dirlift_rv)
        ) else np.nan
        rows.append({
            "source": src, "target": tgt,
            "colift": colift,
            "dirlift_forward": dirlift_fw,
            "dirlift_reverse": dirlift_rv,
            "directional_gap": gap,
        })

    return pd.DataFrame(rows)


def compute_colift_matrix(
    Z: np.ndarray,
    quantile: float = 0.05,
) -> np.ndarray:
    """Compute full pairwise CoLift matrix.

    Parameters
    ----------
    Z        : (T, n) data matrix
    quantile : lower tail quantile

    Returns
    -------
    (n, n) CoLift matrix (symmetric, diagonal = 1/q by construction)
    """
    D = compute_tail_indicators(Z, quantile)
    n = D.shape[1]
    CM = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            CM[i, j] = compute_colift(D, i, j)
    return CM


def permutation_diagnostic(
    colift_matrix: np.ndarray,
    selected_edges: list[tuple[int, int]],
    n_permutations: int = 1000,
    seed: int = 1121,
) -> dict:
    """Node-label permutation diagnostic for CoLift alignment.

    Keeps graph topology fixed, relabels graph endpoints on the pre-computed
    CoLift matrix, and computes permutation p-value for observed alignment.

    Parameters
    ----------
    colift_matrix   : (n, n) pre-computed CoLift matrix
    selected_edges  : list of (source, target) tuples
    n_permutations  : number of label permutations
    seed            : random seed

    Returns
    -------
    dict with: 'observed_mean_colift', 'permutation_dist', 'p_value'
    """
    rng = np.random.RandomState(seed)
    n = colift_matrix.shape[0]

    # Observed: mean CoLift over selected edges
    obs_vals = [colift_matrix[tgt, src] for (src, tgt) in selected_edges
                if np.isfinite(colift_matrix[tgt, src])]
    observed_mean = float(np.mean(obs_vals)) if obs_vals else np.nan

    # Topology: same edge structure, relabel nodes
    perm_means = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        vals = [colift_matrix[perm[tgt], perm[src]] for (src, tgt) in selected_edges
                if np.isfinite(colift_matrix[perm[tgt], perm[src]])]
        perm_means.append(float(np.mean(vals)) if vals else np.nan)

    perm_arr = np.array([v for v in perm_means if np.isfinite(v)])
    n_above  = int(np.sum(perm_arr >= observed_mean))
    p_value  = (1 + n_above) / (len(perm_arr) + 1)

    return {
        "observed_mean_colift": observed_mean,
        "permutation_dist": perm_arr,
        "p_value": p_value,
    }


def block_bootstrap_gap_interval(
    Z_test: np.ndarray,
    shocks_test: np.ndarray,
    selected_edges: list[tuple[int, int]],
    train_thresholds_Z: np.ndarray,
    train_thresholds_shock: np.ndarray,
    block_length: int = 10,
    n_boot: int = 1000,
    seed: int = 1121,
) -> tuple[float, float]:
    """Block-resampled interval for holdout forward-minus-reverse directional gap.

    Uses training-defined thresholds (not re-estimated on test data) to maintain
    strict information separation.

    Parameters
    ----------
    Z_test               : (T_test, n) test period standardized returns
    shocks_test          : (T_test, n) test period recovered shocks
    selected_edges       : list of (source, target) from training selection
    train_thresholds_Z   : (n,) training 5th-percentile thresholds for Z
    train_thresholds_shock : (n,) training 5th-percentile thresholds for shocks
    block_length         : bootstrap block length
    n_boot               : number of bootstrap resamples
    seed                 : random seed

    Returns
    -------
    (5th percentile, 95th percentile) of bootstrap gap distribution
    """
    T, n = Z_test.shape
    rng = np.random.RandomState(seed)
    max_start = T - block_length + 1

    def gap_on_sample(idx: np.ndarray) -> float:
        Z_b  = Z_test[idx]
        S_b  = shocks_test[idx]
        D_b  = (Z_b <= train_thresholds_Z[None, :]).astype(int)
        Sh_b = (S_b <= train_thresholds_shock[None, :]).astype(int)
        fwd_lifts, rev_lifts = [], []
        for (src, tgt) in selected_edges:
            fwd = compute_dirlift(D_b, Sh_b, src, tgt)
            rev = compute_dirlift(D_b, Sh_b, tgt, src)
            if np.isfinite(fwd) and np.isfinite(rev):
                fwd_lifts.append(fwd)
                rev_lifts.append(rev)
        if not fwd_lifts:
            return np.nan
        return float(np.mean(fwd_lifts) - np.mean(rev_lifts))

    # Full-sample gap on test
    all_idx = np.arange(T)

    boot_gaps = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < T:
            start = rng.randint(0, max_start)
            indices.extend(range(start, start + block_length))
        idx_b = np.array(indices[:T], dtype=int)
        g = gap_on_sample(idx_b)
        if np.isfinite(g):
            boot_gaps.append(g)

    boot_arr = np.array(boot_gaps)
    return float(np.percentile(boot_arr, 5)), float(np.percentile(boot_arr, 95))
