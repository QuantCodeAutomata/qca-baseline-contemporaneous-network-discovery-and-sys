"""
Experiment 3: Re-discovery bootstrap for network uncertainty, backbone extraction,
and score-rank reproducibility.

Runs 500 outer bootstrap resamples, re-estimating DirectLiNGAM order and
coefficient selection in each replication, then computes SIS/SVS rank distributions.
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BANK_TICKERS, MARKET_TICKER, MARKET_CAPS, build_baseline_dataset
from src.garch_filter import fit_garch_filter
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    moving_block_bootstrap_indices, wald_bh_selection, compute_bootstrap_se,
)
from src.systemic_scores import (
    compute_total_effect_matrix, compute_systemic_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED   = 1121
OUTER_SEED    = 1121
INNER_SEED    = 731
N_OUTER       = int(os.environ.get("N_OUTER_BOOT", "500"))   # paper: 500
N_INNER       = int(os.environ.get("N_INNER_BOOT", "200"))   # paper: 200
BLOCK_LEN     = 10
MATERIALITY   = 0.05
BH_LEVEL      = 0.05
BACKBONE_FREQ = 0.80
AMBIGUOUS_SUM = 0.10
AMBIGUOUS_MIN = 0.20


def _make_rng(seed: int, replication: int) -> np.random.RandomState:
    """Deterministic sub-stream seed for each outer replication."""
    return np.random.RandomState(seed + replication * 7919)


def run_single_outer_rep(
    Z_baseline: np.ndarray,
    tickers: list[str],
    bank_tickers: list[str],
    boot_idx: np.ndarray,
    inner_seed: int,
    rep_id: int,
) -> dict | None:
    """Run one outer bootstrap replication.

    Returns dict with: order, selected_edge_set, sis_values, svs_values,
                       rank_sis, rank_svs
    Returns None if singular.
    """
    T, n = Z_baseline.shape
    Z_boot = Z_baseline[boot_idx].copy()
    Z_boot -= Z_boot.mean(axis=0)  # center; do NOT rescale

    # Re-estimate DirectLiNGAM order
    try:
        np.random.seed(inner_seed + rep_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lingam
            model = lingam.DirectLiNGAM(measure="pwling")
            model.fit(Z_boot)
        causal_order = list(model.causal_order_)
        B0_b = fit_predecessor_regressions(Z_boot, causal_order)
    except Exception as exc:
        logger.debug("Rep %d: DirectLiNGAM failed: %s", rep_id, exc)
        return None

    # Inner coefficient selection
    try:
        SE_b = compute_bootstrap_se(
            Z_boot, causal_order,
            n_boot=N_INNER,
            block_length=BLOCK_LEN,
            seed=inner_seed + rep_id * 13,
        )
        sel = wald_bh_selection(
            B0_b, SE_b, causal_order,
            q_bh=BH_LEVEL, materiality_threshold=MATERIALITY
        )
        B0_sel = sel["B0_selected_bh"]
    except Exception as exc:
        logger.debug("Rep %d: selection failed: %s", rep_id, exc)
        return None

    # Build selected edge set (as set of original (i,j) index pairs)
    selected_edges = set()
    for i in range(n):
        for j in range(n):
            if B0_sel[i, j] != 0:
                selected_edges.add((i, j))

    # Compute systemic scores
    avail_bank_idx = [tickers.index(t) for t in bank_tickers if t in tickers]
    avail_banks    = [tickers[idx] for idx in avail_bank_idx]
    B0_bank = B0_sel[np.ix_(avail_bank_idx, avail_bank_idx)]

    try:
        C_B = compute_total_effect_matrix(B0_bank, avail_banks)
        scores = compute_systemic_scores(C_B, MARKET_CAPS, avail_banks)
        sis_ranks = {t: float(scores.loc[t, "SIS_rank"]) for t in avail_banks}
        svs_ranks = {t: float(scores.loc[t, "SVS_rank"]) for t in avail_banks}
    except Exception:
        sis_ranks = {t: np.nan for t in avail_banks}
        svs_ranks = {t: np.nan for t in avail_banks}

    return {
        "order": causal_order,
        "selected_edges": selected_edges,
        "sis_ranks": sis_ranks,
        "svs_ranks": svs_ranks,
    }


def run_exp_3(Z_baseline: pd.DataFrame | None = None, dataset: dict | None = None) -> dict:
    """Run the re-discovery bootstrap experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 3: Re-Discovery Bootstrap ===")
    print(f"  N_outer={N_OUTER}, N_inner={N_INNER}, block_length={BLOCK_LEN}")

    # Load data if not provided
    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    if Z_baseline is None:
        z_path = DATA_DIR / "Z_baseline.parquet"
        if z_path.exists():
            Z_baseline = pd.read_parquet(z_path)
        else:
            garch_res = fit_garch_filter(dataset["baseline_returns"], seed=GLOBAL_SEED)
            Z_baseline = garch_res["Z"]

    Z_arr = Z_baseline.values
    tickers = list(Z_baseline.columns)
    n = len(tickers)

    avail_bank_tickers = [t for t in BANK_TICKERS if t in tickers]
    T = Z_arr.shape[0]

    # Pre-generate all outer bootstrap indices for reproducibility
    outer_indices = moving_block_bootstrap_indices(T, N_OUTER, BLOCK_LEN, seed=OUTER_SEED)

    # Full-sample baseline result (for rank correlation reference)
    print("\n[EXP3] Estimating full-sample baseline for reference...")
    from src.lingam_utils import descriptive_edge_count
    import lingam
    np.random.seed(GLOBAL_SEED)
    base_model = lingam.DirectLiNGAM(measure="pwling")
    base_model.fit(Z_arr)
    base_order = list(base_model.causal_order_)
    base_B0 = fit_predecessor_regressions(Z_arr, base_order)

    # Frequency tracking
    freq_matrix = np.zeros((n, n), dtype=int)   # directional frequencies

    # Score rank matrices  (N_OUTER x n_banks)
    n_banks = len(avail_bank_tickers)
    sis_rank_matrix = np.full((N_OUTER, n_banks), np.nan)
    svs_rank_matrix = np.full((N_OUTER, n_banks), np.nan)

    # Run outer replications — parallel with joblib (one process per core)
    n_jobs = int(os.environ.get("N_JOBS_EXP3", "-1"))  # -1 = all cores
    print(f"\n[EXP3] Running {N_OUTER} outer replications (n_jobs={n_jobs})...")

    # Limit BLAS threads per worker to avoid oversubscription
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_single_outer_rep)(
            Z_arr, tickers, avail_bank_tickers,
            outer_indices[rep], INNER_SEED + rep, rep
        )
        for rep in range(N_OUTER)
    )

    n_completed = 0
    n_singular  = 0

    for rep, result in enumerate(parallel_results):
        if result is None:
            n_singular += 1
            continue
        n_completed += 1

        # Accumulate edge frequencies
        for (i, j) in result["selected_edges"]:
            freq_matrix[i, j] += 1

        # Accumulate rank matrices
        for k, t in enumerate(avail_bank_tickers):
            if t in result["sis_ranks"] and np.isfinite(result["sis_ranks"][t]):
                sis_rank_matrix[rep, k] = result["sis_ranks"][t]
            if t in result["svs_ranks"] and np.isfinite(result["svs_ranks"][t]):
                svs_rank_matrix[rep, k] = result["svs_ranks"][t]

    print(f"\n[EXP3] Completed {n_completed}/{N_OUTER} replications ({n_singular} singular)")

    # ── Directional frequency analysis ────────────────────────────────────────
    freq_frac = freq_matrix / max(n_completed, 1)

    # Active pairs: unordered pairs active in >= 10% of resamples
    active_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            total_freq = freq_frac[i, j] + freq_frac[j, i]
            if total_freq >= 0.10:
                active_pairs.append((i, j))
    n_active = len(active_pairs)

    # Ambiguous pairs: total freq >= 10% AND both directions >= 20%
    ambiguous_pairs = []
    for (i, j) in active_pairs:
        if freq_frac[i, j] >= AMBIGUOUS_MIN and freq_frac[j, i] >= AMBIGUOUS_MIN:
            ambiguous_pairs.append((i, j))
    n_ambiguous = len(ambiguous_pairs)
    ambiguous_share = n_ambiguous / n_active if n_active > 0 else np.nan

    # Strict backbone: base edges selected in >= 80% of replications
    # (baseline B0_selected)
    # For backbone, we need the baseline selected edges
    b0_sel_path = DATA_DIR / "exp1_B0_selected.npy"
    if b0_sel_path.exists():
        B0_base_sel = np.load(b0_sel_path)
        baseline_edges = set(
            (i, j) for i in range(n) for j in range(n) if B0_base_sel[i, j] != 0
        )
    else:
        # Use base_B0 with 0.05 threshold as proxy
        baseline_edges = set(
            (i, j) for i in range(n) for j in range(n) if abs(base_B0[i, j]) >= MATERIALITY
        )

    backbone_edges = [
        (i, j) for (i, j) in baseline_edges if freq_frac[i, j] >= BACKBONE_FREQ
    ]
    n_backbone = len(backbone_edges)
    # Bank-to-bank backbone edges
    bank_idx_set = {tickers.index(t) for t in avail_bank_tickers if t in tickers}
    n_backbone_bb = sum(1 for (i, j) in backbone_edges if i in bank_idx_set and j in bank_idx_set)
    n_backbone_mkt = n_backbone - n_backbone_bb

    print(f"\n[EXP3] Network uncertainty summary:")
    print(f"  Active pairs (>= 10%):     {n_active}  (target: ~624)")
    print(f"  Ambiguous pairs:           {n_ambiguous}  (target: ~{int(0.178*624):.0f})")
    print(f"  Ambiguous share:           {ambiguous_share:.3f}  (target: ~0.178)")
    print(f"  Strict backbone edges:     {n_backbone}  (target: ~32)")
    print(f"    Bank-to-bank:            {n_backbone_bb}  (target: ~19)")
    print(f"    Market-originating:      {n_backbone_mkt}  (target: ~13)")

    # ── Rank reproducibility ──────────────────────────────────────────────────
    print("\n[EXP3] Rank reproducibility analysis...")

    # Compute full-sample SIS/SVS ranks
    avail_bank_idx = [tickers.index(t) for t in avail_bank_tickers if t in tickers]
    base_B0_bank = base_B0[np.ix_(avail_bank_idx, avail_bank_idx)]
    try:
        from src.systemic_scores import compute_total_effect_matrix, compute_systemic_scores
        C_B_base = compute_total_effect_matrix(base_B0_bank, avail_bank_tickers)
        scores_base = compute_systemic_scores(C_B_base, MARKET_CAPS, avail_bank_tickers)
        base_sis_ranks = np.array([float(scores_base.loc[t, "SIS_rank"]) for t in avail_bank_tickers])
        base_svs_ranks = np.array([float(scores_base.loc[t, "SVS_rank"]) for t in avail_bank_tickers])
    except Exception:
        base_sis_ranks = np.arange(1, n_banks + 1, dtype=float)
        base_svs_ranks = np.arange(1, n_banks + 1, dtype=float)

    # Rank correlations with baseline
    sis_corrs, svs_corrs = [], []
    for rep in range(N_OUTER):
        sis_r = sis_rank_matrix[rep]
        svs_r = svs_rank_matrix[rep]
        valid_sis = ~np.isnan(sis_r)
        valid_svs = ~np.isnan(svs_r)
        if valid_sis.sum() >= 10:
            r, _ = stats.spearmanr(base_sis_ranks[valid_sis], sis_r[valid_sis])
            if np.isfinite(r):
                sis_corrs.append(r)
        if valid_svs.sum() >= 10:
            r, _ = stats.spearmanr(base_svs_ranks[valid_svs], svs_r[valid_svs])
            if np.isfinite(r):
                svs_corrs.append(r)

    sis_corrs = np.array(sis_corrs)
    svs_corrs = np.array(svs_corrs)

    def safe_pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p)) if len(arr) > 0 else np.nan

    print(f"\n  SIS rank correlation (baseline vs bootstrap):")
    print(f"    Median: {safe_pct(sis_corrs, 50):.3f}  (target: ~0.50)")
    print(f"    5th pct: {safe_pct(sis_corrs, 5):.3f}  (target: ~0.31)")
    print(f"  SVS rank correlation (baseline vs bootstrap):")
    print(f"    Median: {safe_pct(svs_corrs, 50):.3f}  (target: ~0.72)")
    print(f"    5th pct: {safe_pct(svs_corrs, 5):.3f}  (target: ~0.58)")

    svs_more_stable = (
        safe_pct(svs_corrs, 50) > safe_pct(sis_corrs, 50)
        if not np.isnan(safe_pct(sis_corrs, 50)) else False
    )
    print(f"\n  SVS more reproducible than SIS: {svs_more_stable}  (expected: True)")

    # ── Top-5 inclusion frequencies ──────────────────────────────────────────
    sis_top5_freq = {}
    svs_top5_freq = {}
    for k, t in enumerate(avail_bank_tickers):
        sis_col = sis_rank_matrix[:, k]
        svs_col = svs_rank_matrix[:, k]
        valid_sis = sis_col[~np.isnan(sis_col)]
        valid_svs = svs_col[~np.isnan(svs_col)]
        sis_top5_freq[t] = float(np.mean(valid_sis <= 5)) if len(valid_sis) > 0 else np.nan
        svs_top5_freq[t] = float(np.mean(valid_svs <= 5)) if len(valid_svs) > 0 else np.nan

    top5_sis = sorted(sis_top5_freq.items(), key=lambda x: -x[1])[:5]
    top5_svs = sorted(svs_top5_freq.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top-5 SIS inclusion frequencies: {top5_sis}")
    print(f"  Top-5 SVS inclusion frequencies: {top5_svs}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_rank_correlations(sis_corrs, svs_corrs)
    _plot_frequency_matrix(freq_frac, tickers, n=min(n, 20))

    # ── Save results ──────────────────────────────────────────────────────────
    freq_df = pd.DataFrame(freq_frac, index=tickers, columns=tickers)
    freq_df.to_csv(RESULTS_DIR / "exp3_edge_frequencies.csv")

    pd.DataFrame(sis_rank_matrix, columns=avail_bank_tickers).to_csv(
        RESULTS_DIR / "exp3_sis_rank_matrix.csv", index=False
    )
    pd.DataFrame(svs_rank_matrix, columns=avail_bank_tickers).to_csv(
        RESULTS_DIR / "exp3_svs_rank_matrix.csv", index=False
    )

    meta = {
        "exp_id": "exp_3",
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "n_completed": n_completed,
        "n_singular": n_singular,
        "n_active_pairs": n_active,
        "n_ambiguous_pairs": n_ambiguous,
        "ambiguous_share": float(ambiguous_share) if np.isfinite(ambiguous_share) else None,
        "n_backbone_edges": n_backbone,
        "n_backbone_bb": n_backbone_bb,
        "n_backbone_mkt": n_backbone_mkt,
        "sis_median_corr": safe_pct(sis_corrs, 50),
        "sis_p5_corr": safe_pct(sis_corrs, 5),
        "svs_median_corr": safe_pct(svs_corrs, 50),
        "svs_p5_corr": safe_pct(svs_corrs, 5),
        "svs_more_stable_than_sis": bool(svs_more_stable),
    }
    with open(RESULTS_DIR / "exp3_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp3_metadata.json")

    return {
        "freq_matrix": freq_frac,
        "active_pairs": active_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "backbone_edges": backbone_edges,
        "sis_rank_matrix": sis_rank_matrix,
        "svs_rank_matrix": svs_rank_matrix,
        "sis_corrs": sis_corrs,
        "svs_corrs": svs_corrs,
        "meta": meta,
    }


def _plot_rank_correlations(sis_corrs: np.ndarray, svs_corrs: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.2, 1.0, 25)
    if len(sis_corrs) > 0:
        ax.hist(sis_corrs, bins=bins, alpha=0.6, color="steelblue",
                label=f"SIS (median={np.median(sis_corrs):.2f})", edgecolor="white")
    if len(svs_corrs) > 0:
        ax.hist(svs_corrs, bins=bins, alpha=0.6, color="coral",
                label=f"SVS (median={np.median(svs_corrs):.2f})", edgecolor="white")
    ax.axvline(0.5, color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(0.72, color="coral", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Spearman rank correlation with baseline ranking", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Bootstrap Rank Reproducibility: SIS vs SVS\n"
                 "(higher = more stable ranking across resamples)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Rank stability plot saved to results/exp3_rank_stability.png")


def _plot_frequency_matrix(
    freq_frac: np.ndarray,
    tickers: list[str],
    n: int = 20,
) -> None:
    sub = freq_frac[:n, :n]
    sub_labels = tickers[:n]
    fig, ax = plt.subplots(figsize=(12, 10))
    import seaborn as sns
    sns.heatmap(
        sub, ax=ax, cmap="Blues", vmin=0, vmax=0.5,
        xticklabels=sub_labels, yticklabels=sub_labels,
        linewidths=0.1, cbar_kws={"label": "Selection frequency"},
    )
    ax.set_title("Edge Selection Frequency Matrix (first 20 nodes)\n"
                 f"Re-discovery bootstrap ({N_OUTER} replications)", fontsize=12)
    ax.set_xlabel("Source j", fontsize=10)
    ax.set_ylabel("Target i", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_edge_frequency_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Edge frequency matrix saved to results/exp3_edge_frequency_matrix.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_3(dataset=dataset)
    print("\n=== EXP 3 COMPLETE ===")
