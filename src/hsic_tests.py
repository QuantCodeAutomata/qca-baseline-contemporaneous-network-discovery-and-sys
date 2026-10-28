"""
Exact Gaussian-kernel HSIC with circular-shift reference distribution.

Implements the paper's pairwise independence test for recovered structural shocks,
using all admissible circular shifts (min wrap-around distance >= 20).
"""

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ── Kernel utilities ────────────────────────────────────────────────────────────

def _median_bandwidth(x: np.ndarray) -> float:
    """Compute median pairwise distance bandwidth for 1D array."""
    n = len(x)
    if n <= 1:
        return 1.0
    diffs = np.abs(x[:, None] - x[None, :])
    # Upper triangle
    upper = diffs[np.triu_indices(n, k=1)]
    med = float(np.median(upper))
    return med if med > 0 else 1.0


def gaussian_kernel_matrix(
    x: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Compute Gaussian kernel matrix K where K_ij = exp(-||x_i - x_j||^2 / (2h^2)).

    Parameters
    ----------
    x         : 1D array of shape (T,)
    bandwidth : kernel bandwidth h; if None, uses median pairwise distance

    Returns
    -------
    (T x T) kernel matrix
    """
    x = x.ravel()
    if bandwidth is None:
        bandwidth = _median_bandwidth(x)
    sq_dists = (x[:, None] - x[None, :]) ** 2
    K = np.exp(-sq_dists / (2.0 * bandwidth ** 2))
    return K


def centering_matrix(n: int) -> np.ndarray:
    """Return centering matrix H = I - (1/n) * 11'."""
    return np.eye(n) - np.ones((n, n)) / n


def compute_hsic(
    x: np.ndarray,
    y: np.ndarray,
    bw_x: Optional[float] = None,
    bw_y: Optional[float] = None,
) -> float:
    """Compute HSIC = trace(K H L H) / (n-1)^2.

    Uses the identity trace(KH @ LH) = (KH * LH.T).sum() to avoid O(T^3)
    matrix multiply; centering via row-mean subtraction instead of H matrix.

    Parameters
    ----------
    x   : 1D array of shape (T,)
    y   : 1D array of shape (T,)
    bw_x: bandwidth for x kernel (None = median heuristic)
    bw_y: bandwidth for y kernel (None = median heuristic)

    Returns
    -------
    HSIC statistic (float)
    """
    x = x.ravel()
    y = y.ravel()
    n = len(x)
    assert len(y) == n, "x and y must have the same length"

    K = gaussian_kernel_matrix(x, bw_x)
    L = gaussian_kernel_matrix(y, bw_y)

    # Efficient centering: K @ H = K - row_mean_K  (H = I - 11'/n)
    KH = K - K.mean(axis=1, keepdims=True)   # O(T^2)
    LH = L - L.mean(axis=1, keepdims=True)   # O(T^2)

    # trace(KH @ LH) = (KH * LH.T).sum()   — O(T^2) not O(T^3)
    hsic_val = float(np.einsum("ij,ji->", KH, LH)) / float((n - 1) ** 2)
    return hsic_val


# ── Circular shift p-value (FFT-accelerated) ────────────────────────────────────

def circular_shift_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    min_shift_distance: int = 20,
    bw_x: Optional[float] = None,
    bw_y: Optional[float] = None,
) -> tuple[float, float, int]:
    """HSIC p-value via circular shift reference distribution.

    Enumerates all admissible shifts s where min(s, T-s) >= min_shift_distance.
    For T=1990 and min_shift_distance=20: s in [20, 1970] → 1951 shifts.

    **FFT acceleration**: Instead of computing a T×T kernel matrix for every
    admissible shift (O(T^2 × N_shifts) per pair), we precompute the kernel
    matrices K and L once and use a 2D FFT cross-correlation to evaluate
    trace(KH @ L_s^H) for ALL shifts simultaneously in O(T^2 log T).

    Key identity:
        trace(KH @ LH_s) = term1(s) − term2(s)
    where
        term1(s) = (KH * L_s.T).sum()
                 = 2D cross-corr(KH, L.T) evaluated at diagonal index (T−s, T−s)
        term2(s) = row_mean_L_s · col_sum_KH
                 = 1D cross-corr(row_mean_L, col_sum_KH) at index s

    Both are computed via FFT in O(T^2 log T + T log T) per pair.

    P-value = (1 + #{shifted HSIC >= observed HSIC}) / (N_shifts + 1)

    Parameters
    ----------
    x                  : 1D array (T,)
    y                  : 1D array (T,)
    min_shift_distance : minimum wrap-around circular distance
    bw_x               : pre-computed bandwidth for x
    bw_y               : pre-computed bandwidth for y

    Returns
    -------
    (observed_hsic, p_value, n_admissible_shifts)
    """
    x = x.ravel()
    y = y.ravel()
    T = len(x)

    if bw_x is None:
        bw_x = _median_bandwidth(x)
    if bw_y is None:
        bw_y = _median_bandwidth(y)

    # ── Precompute kernel matrices once ──────────────────────────────────────
    K = gaussian_kernel_matrix(x, bw_x)   # (T, T)
    L = gaussian_kernel_matrix(y, bw_y)   # (T, T)

    # Efficient centering: A @ H = A − row_mean_A  (H is idempotent)
    KH = K - K.mean(axis=1, keepdims=True)  # O(T^2)

    # Auxiliary 1D vectors for term2
    col_sum_KH = KH.sum(axis=0)            # (T,)  — sum_i KH[i,j]
    row_mean_L = L.mean(axis=1)            # (T,)  — precomputed for all shifts

    # Admissible shifts
    admissible_shifts = np.array(
        [s for s in range(1, T) if min(s, T - s) >= min_shift_distance],
        dtype=np.intp,
    )
    n_shifts = int(admissible_shifts.size)

    # ── Term1: 2D FFT cross-correlation ──────────────────────────────────────
    # term1(s) = sum_{i,j} KH[i,j] * L.T[(i−s)%T, (j−s)%T]
    #          = (KH ⊛ L.T)[T−s, T−s]   (2D circular cross-corr)
    # FFT formula: corr_2d = irfft2(conj(rfft2(KH)) * rfft2(L.T))
    KH_f = np.fft.rfft2(KH)
    LT_f = np.fft.rfft2(L.T)
    corr_2d = np.fft.irfft2(np.conj(KH_f) * LT_f, s=(T, T))  # (T, T) real

    idx = (T - admissible_shifts) % T          # diagonal indices for each shift
    term1_shifts = corr_2d[idx, idx]           # (n_shifts,)
    term1_obs    = float(corr_2d[0, 0])        # s = 0

    # ── Term2: 1D FFT cross-correlation ──────────────────────────────────────
    # term2(s) = sum_u row_mean_L[u] * col_sum_KH[(u+s)%T]
    #          = (row_mean_L ⊛ col_sum_KH)[s]
    rl_f = np.fft.rfft(row_mean_L)
    ck_f = np.fft.rfft(col_sum_KH)
    corr_1d = np.fft.irfft(np.conj(rl_f) * ck_f, n=T)  # (T,) real

    term2_shifts = corr_1d[admissible_shifts]   # (n_shifts,)
    term2_obs    = float(corr_1d[0])            # s = 0

    # ── HSIC values ──────────────────────────────────────────────────────────
    scale = 1.0 / float((T - 1) ** 2)
    hsic_obs    = (term1_obs - term2_obs) * scale
    hsic_shifts = (term1_shifts - term2_shifts) * scale

    n_above = int(np.sum(hsic_shifts >= hsic_obs))
    p_value = (1.0 + n_above) / (n_shifts + 1)
    return float(hsic_obs), float(p_value), n_shifts


# ── Full pairwise HSIC testing ──────────────────────────────────────────────────

def test_all_pairs_hsic(
    shocks: np.ndarray,
    min_shift_distance: int = 20,
    q_bh: float = 0.05,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Test all n*(n-1)/2 pairwise shock dependencies using HSIC with circular shifts.

    Parameters
    ----------
    shocks             : (T, n) array of recovered structural shocks
    min_shift_distance : minimum circular shift distance (default 20)
    q_bh               : BH/BY FDR level
    n_jobs             : number of parallel jobs (1 = serial)

    Returns
    -------
    DataFrame with columns: i, j, bw_i, bw_j, hsic_observed, n_shifts,
                            n_above, pvalue_raw, pvalue_bh, pvalue_by,
                            reject_raw, reject_bh, reject_by
    """
    T, n = shocks.shape
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)

    logger.info("Testing %d pairs on shocks (%d, %d)", n_pairs, T, n)

    # Pre-compute bandwidths for all variables
    bandwidths = [_median_bandwidth(shocks[:, k]) for k in range(n)]

    def test_pair(i: int, j: int) -> tuple:
        xi = shocks[:, i]
        xj = shocks[:, j]
        bw_i = bandwidths[i]
        bw_j = bandwidths[j]
        hsic_obs, pval, n_shifts = circular_shift_pvalue(
            xi, xj, min_shift_distance, bw_x=bw_i, bw_y=bw_j
        )
        n_above = int(round(pval * (n_shifts + 1) - 1))
        return i, j, bw_i, bw_j, hsic_obs, n_shifts, n_above, pval

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(test_pair)(i, j) for i, j in pairs
            )
        except ImportError:
            logger.warning("joblib not available, running serially")
            results = [test_pair(i, j) for i, j in pairs]
    else:
        results = [test_pair(i, j) for i, j in pairs]

    rows = []
    pvals_raw = []
    for r in results:
        rows.append({
            "i": r[0], "j": r[1],
            "bw_i": r[2], "bw_j": r[3],
            "hsic_observed": r[4],
            "n_shifts": r[5],
            "n_above": r[6],
            "pvalue_raw": r[7],
        })
        pvals_raw.append(r[7])

    pvals_arr = np.array(pvals_raw)

    # BH correction
    _, p_bh, _, _ = multipletests(pvals_arr, alpha=q_bh, method="fdr_bh")
    # BY correction
    _, p_by, _, _ = multipletests(pvals_arr, alpha=q_bh, method="fdr_by")

    for k, row in enumerate(rows):
        row["pvalue_bh"] = float(p_bh[k])
        row["pvalue_by"] = float(p_by[k])
        row["reject_raw"] = bool(pvals_arr[k] < 0.05)
        row["reject_bh"]  = bool(p_bh[k] < q_bh)
        row["reject_by"]  = bool(p_by[k] < q_bh)

    df = pd.DataFrame(rows)

    n_raw_rej = int(df["reject_raw"].sum())
    n_bh_rej  = int(df["reject_bh"].sum())
    n_by_rej  = int(df["reject_by"].sum())
    print(f"[HSIC] Tested pairs: {n_pairs}")
    print(f"[HSIC] Unadjusted p<0.05 rejections: {n_raw_rej} (filtered target ~219, unfiltered ~276)")
    print(f"[HSIC] BH rejections: {n_bh_rej} (filtered target ~14)")
    print(f"[HSIC] BY rejections: {n_by_rej} (filtered target ~0)")

    return df


def mean_squared_shock_correlation(shocks: np.ndarray) -> float:
    """Compute mean pairwise Pearson correlation of squared shock series.

    Parameters
    ----------
    shocks : (T, n) shock array

    Returns
    -------
    Mean pairwise correlation of squared shocks (float)
    """
    sq_shocks = shocks ** 2
    n = sq_shocks.shape[1]
    corrs = []
    for i, j in combinations(range(n), 2):
        r = float(np.corrcoef(sq_shocks[:, i], sq_shocks[:, j])[0, 1])
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else np.nan
