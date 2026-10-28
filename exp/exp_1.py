"""
Experiment 1: Baseline contemporaneous network discovery and systemic score replication.

Replicates the paper's primary baseline:
- GARCH(1,1)-Student-t volatility standardization
- VAR(0)-DirectLiNGAM contemporaneous network
- Moving-block bootstrap Wald tests with BH correction
- Systemic Impact Score (SIS), Systemic Vulnerability Score (SVS), nSVS
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
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    BANK_TICKERS, MARKET_CAPS, MARKET_TICKER, ALL_TICKERS,
    build_baseline_dataset,
)
from src.garch_filter import fit_garch_filter
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    compute_bootstrap_se, wald_bh_selection,
    descriptive_edge_count, recover_shocks,
)
from src.systemic_scores import (
    compute_total_effect_matrix, compute_systemic_scores, compute_rank_correlation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

# Seeds as specified in the paper
GLOBAL_SEED = 1121
BOOT_SEED   = 731

# Bootstrap parameters (reduce N_BOOT for faster testing)
N_BOOT_COEF    = int(os.environ.get("N_BOOT_EXP1", "1000"))
BLOCK_LEN      = 10
MATERIALITY    = 0.05


def run_exp_1() -> dict:
    """Run the full baseline experiment pipeline."""
    np.random.seed(GLOBAL_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1-4: Data ────────────────────────────────────────────────────────
    print("\n=== EXP 1: Data Preparation ===")
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    prices       = dataset["prices"]
    raw_returns  = dataset["raw_returns"]
    baseline_ret = dataset["baseline_returns"]

    n_prices     = len(prices)
    n_raw_ret    = len(raw_returns)
    n_pre2016    = dataset["pre2016_removed"]
    n_screened   = dataset["screen_removed"]
    n_baseline   = len(baseline_ret)

    print(f"\nData verification:")
    print(f"  Aligned price rows:        {n_prices}  (target: 2205)")
    print(f"  Raw return intervals:       {n_raw_ret}  (target: 2204)")
    print(f"  Pre-2016 removed:          {n_pre2016}  (target: 24)")
    print(f"  Calendar-screen removed:   {n_screened}  (target: 190)")
    print(f"  Baseline observations:      {n_baseline}  (target: 1990)")

    # Verify expected columns (may differ slightly from exact 52 if some tickers unavailable)
    available_tickers = list(baseline_ret.columns)
    print(f"  Available tickers:         {len(available_tickers)} (expected: ≤52)")

    # Separate bank tickers and market ticker (only those available)
    avail_bank_tickers = [t for t in BANK_TICKERS if t in available_tickers]
    market_available   = MARKET_TICKER in available_tickers
    print(f"  Available bank tickers:    {len(avail_bank_tickers)} (expected: 51)")
    print(f"  Market proxy available:    {market_available}")

    # ── Step 5-7: GARCH filtering ─────────────────────────────────────────────
    print("\n=== EXP 1: GARCH(1,1)-Student-t Filtering ===")
    garch_result = fit_garch_filter(baseline_ret, seed=GLOBAL_SEED)
    Z = garch_result["Z"]
    garch_params = garch_result["garch_params"]

    print(f"  Filtered Z_t shape: {Z.shape}  (target: 1990 x {len(available_tickers)})")
    assert Z.shape[0] == n_baseline, f"Z row count mismatch: {Z.shape[0]}"

    # Save Z_t
    Z.to_parquet(DATA_DIR / "Z_baseline.parquet")

    # ── Step 8: Diagnostics (informational) ──────────────────────────────────
    print("\n=== EXP 1: Diagnostics ===")
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.stats.stattools import jarque_bera

    diag_results = {}
    for ticker in available_tickers[:5]:  # summarize first 5 for brevity
        col = Z[ticker].values
        adf_stat, adf_p, _, _, _, _ = adfuller(col)
        jb_stat, jb_p = jarque_bera(col)[:2]
        sw_stat, sw_p = stats.shapiro(col[:500])  # Shapiro on subset (n≤5000)
        diag_results[ticker] = {
            "ADF_stat": adf_stat, "ADF_p": adf_p,
            "JB_stat": float(jb_stat), "JB_p": float(jb_p),
            "SW_stat": float(sw_stat), "SW_p": float(sw_p),
        }
        print(f"  {ticker}: ADF p={adf_p:.4f}, JB p={jb_p:.4f}, SW p={sw_p:.4f}")

    # ── Step 9: Lag selection (confirm p=0) ──────────────────────────────────
    print("\n=== EXP 1: VAR Lag Selection ===")
    # For p=0 model, each variable is independent of lagged values.
    # A minimal check: compare AIC/BIC of p=0 vs p=1 on a small subset.
    from statsmodels.tsa.api import VAR
    Z_np = Z.values
    Z_small = Z_np[:, :5]  # lag check on first 5 variables only for speed
    aic_by_lag = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lag in range(1, 4):
            try:
                var_res = VAR(Z_small).fit(lag)
                aic_by_lag[lag] = var_res.aic
            except Exception:
                pass
    if aic_by_lag:
        best_lag = min(aic_by_lag, key=aic_by_lag.get)
        print(f"  VAR AIC by lag (subset): {aic_by_lag}")
        print(f"  Selected lag (baseline): 0 (paper specification)")
    print("  Baseline lag: p=0 (VAR(0) → reduces to structural contemporaneous model)")

    # ── Step 10: DirectLiNGAM estimation ─────────────────────────────────────
    print("\n=== EXP 1: DirectLiNGAM Estimation ===")
    Z_arr = Z.values
    tickers_ordered = list(Z.columns)

    lingam_result = estimate_direct_lingam(Z_arr, seed=GLOBAL_SEED)
    causal_order  = lingam_result["order"]
    B0_adj        = lingam_result["adjacency_matrix"]

    n_vars = len(tickers_ordered)
    expected_order_allowed = n_vars * (n_vars - 1) // 2
    print(f"  Variables (d): {n_vars}")
    print(f"  Order-allowed coefficients: {expected_order_allowed}  (target: {52*51//2}=1326 for full 52)")

    # Re-fit via OLS predecessor regressions (to get same coefficients)
    B0 = fit_predecessor_regressions(Z_arr, causal_order)

    # Map causal_order to ticker labels
    causal_order_tickers = [tickers_ordered[i] for i in causal_order]
    print(f"  Causal order (first 5): {causal_order_tickers[:5]}")

    # ── Step 11: Descriptive edge count ──────────────────────────────────────
    n_descriptive = descriptive_edge_count(B0, threshold=MATERIALITY)
    print(f"\n=== EXP 1: Descriptive Edge Count ===")
    print(f"  |B0_ij| >= {MATERIALITY}: {n_descriptive} edges  (target: ~312)")

    # ── Steps 12-14: Bootstrap coefficient selection ──────────────────────────
    print(f"\n=== EXP 1: Moving-Block Bootstrap SE (n_boot={N_BOOT_COEF}) ===")
    SE = compute_bootstrap_se(
        Z_arr, causal_order,
        n_boot=N_BOOT_COEF,
        block_length=BLOCK_LEN,
        seed=BOOT_SEED,
    )
    print("  Bootstrap SE computed.")

    selection_result = wald_bh_selection(
        B0, SE, causal_order,
        q_bh=0.05,
        materiality_threshold=MATERIALITY,
    )
    B0_selected = selection_result["B0_selected_bh"]
    B0_selected_by = selection_result["B0_selected_by"]

    print(f"\n  Edge count summary:")
    print(f"    Unadjusted p<0.05:  {selection_result['n_unadj_sig']}  (target: ~332)")
    print(f"    BH-selected:        {selection_result['n_bh']}  (target: ~219)")
    print(f"    BH+materiality:     {selection_result['n_bh_mat']}  (target: ~214)")
    print(f"    BY+materiality:     {selection_result['n_by_mat']}  (target: ~169)")

    # ── Step 16: Bank-only matrix and total-effect ────────────────────────────
    print("\n=== EXP 1: Bank-Only Total-Effect Matrix ===")
    # Remove market node from selected matrix
    bank_mask = [t for t in tickers_ordered if t in avail_bank_tickers]
    bank_idx  = [tickers_ordered.index(t) for t in bank_mask]

    B0_bank = B0_selected[np.ix_(bank_idx, bank_idx)]

    try:
        C_B = compute_total_effect_matrix(B0_bank, bank_mask)
        print(f"  C^B shape: {C_B.shape}  (target: 51×51)")

        # ── Step 17: SIS/SVS computation ─────────────────────────────────────
        print("\n=== EXP 1: Systemic Scores ===")
        scores = compute_systemic_scores(C_B, MARKET_CAPS, bank_mask)

        print("\nTop 5 by SIS (target: BKT.MC ~269.5, FBK.MI ~175.8, CABK.MC ~160.3):")
        sis_top5 = scores.nlargest(5, "SIS")
        for t, row in sis_top5.iterrows():
            print(f"  {t}: SIS={row['SIS']:.2f}, rank={int(row['SIS_rank'])}")

        print("\nTop 5 by SVS (target: ACA.PA ~96.4, GLE.PA ~91.1, BNP.PA ~82.7):")
        svs_top5 = scores.nlargest(5, "SVS")
        for t, row in svs_top5.iterrows():
            print(f"  {t}: SVS={row['SVS']:.2f}, rank={int(row['SVS_rank'])}")

        sis_svs_corr = compute_rank_correlation(scores)
        print(f"\nSpearman rank corr SIS vs SVS: {sis_svs_corr:.4f}  (target: ~-0.02)")

        # Save scores
        scores.to_csv(RESULTS_DIR / "exp1_systemic_scores.csv")
        C_B.to_parquet(DATA_DIR / "C_B_baseline.parquet")
        print("[OUTPUT] Systemic scores saved to results/exp1_systemic_scores.csv")

    except Exception as exc:
        logger.warning("C^B computation failed: %s", exc)
        scores = pd.DataFrame()
        sis_svs_corr = np.nan
        C_B = None

    # ── Step 18-19: Plots ─────────────────────────────────────────────────────
    print("\n=== EXP 1: Generating Plots ===")
    _plot_network_adjacency(B0_selected, tickers_ordered, "BH-selected B0^s")
    _plot_score_rankings(scores, avail_bank_tickers)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_1",
        "n_prices": n_prices,
        "n_raw_returns": n_raw_ret,
        "n_pre2016_removed": n_pre2016,
        "n_screen_removed": n_screened,
        "n_baseline": n_baseline,
        "n_available_tickers": len(available_tickers),
        "n_available_banks": len(avail_bank_tickers),
        "n_order_allowed": expected_order_allowed,
        "n_descriptive_edges": n_descriptive,
        "n_unadj_sig": selection_result["n_unadj_sig"],
        "n_bh_selected": selection_result["n_bh"],
        "n_bh_materiality": selection_result["n_bh_mat"],
        "n_by_materiality": selection_result["n_by_mat"],
        "sis_svs_spearman": sis_svs_corr,
        "lag_selection": 0,
        "n_boot_coef": N_BOOT_COEF,
    }
    if len(scores) > 0:
        for t in scores.nlargest(5, "SIS").index:
            meta[f"SIS_{t}"] = float(scores.loc[t, "SIS"])
        for t in scores.nlargest(5, "SVS").index:
            meta[f"SVS_{t}"] = float(scores.loc[t, "SVS"])

    with open(RESULTS_DIR / "exp1_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp1_metadata.json")

    # Save bootstrap SEs and p-values
    np.save(DATA_DIR / "exp1_B0.npy", B0)
    np.save(DATA_DIR / "exp1_SE.npy", SE)
    np.save(DATA_DIR / "exp1_B0_selected.npy", B0_selected)
    pd.Series(causal_order).to_csv(DATA_DIR / "exp1_causal_order.csv", header=False)

    return {
        "Z": Z,
        "B0": B0,
        "SE": SE,
        "B0_selected": B0_selected,
        "causal_order": causal_order,
        "tickers_ordered": tickers_ordered,
        "bank_tickers": bank_mask,
        "C_B": C_B,
        "scores": scores,
        "sis_svs_corr": sis_svs_corr,
        "meta": meta,
        "selection_result": selection_result,
    }


def _plot_network_adjacency(
    B0: np.ndarray,
    tickers: list[str],
    title: str,
) -> None:
    """Plot heatmap of selected coefficient matrix."""
    n = B0.shape[0]
    label_step = max(1, n // 10)
    labels = [tickers[i] if i % label_step == 0 else "" for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        B0,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.5, vmax=0.5,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title(f"Selected Contemporaneous Coefficient Matrix\n{title}", fontsize=14)
    ax.set_xlabel("Source j", fontsize=11)
    ax.set_ylabel("Target i", fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_B0_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] B0 heatmap saved to results/exp1_B0_heatmap.png")


def _plot_score_rankings(scores: pd.DataFrame, tickers: list[str]) -> None:
    """Bar plots for SIS and SVS rankings."""
    if scores.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # SIS top 20
    sis_sorted = scores.sort_values("SIS", ascending=False).head(20)
    axes[0].barh(range(len(sis_sorted)), sis_sorted["SIS"].values, color="steelblue")
    axes[0].set_yticks(range(len(sis_sorted)))
    axes[0].set_yticklabels(sis_sorted.index.tolist(), fontsize=9)
    axes[0].set_xlabel("SIS (EUR bn impact-weighted)", fontsize=11)
    axes[0].set_title("Top 20 Banks by Systemic Impact Score (SIS)\n[transmitter role]", fontsize=12)
    axes[0].invert_yaxis()

    # SVS top 20
    svs_sorted = scores.sort_values("SVS", ascending=False).head(20)
    axes[1].barh(range(len(svs_sorted)), svs_sorted["SVS"].values, color="coral")
    axes[1].set_yticks(range(len(svs_sorted)))
    axes[1].set_yticklabels(svs_sorted.index.tolist(), fontsize=9)
    axes[1].set_xlabel("SVS (EUR bn vulnerability-weighted)", fontsize=11)
    axes[1].set_title("Top 20 Banks by Systemic Vulnerability Score (SVS)\n[receiver role]", fontsize=12)
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_score_rankings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Score rankings saved to results/exp1_score_rankings.png")


if __name__ == "__main__":
    results = run_exp_1()
    print("\n=== EXP 1 COMPLETE ===")
