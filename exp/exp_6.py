"""
Experiment 6: Sensitivity and robustness modules across alternative samples,
preprocessing choices, and model specifications.

Implements robustness modules A-H comparing to the baseline:
  A: Unfiltered same-sample returns
  B: Including 2015 intervals (calendar-screened)
  C: 2016+ without calendar screen
  D: All aligned dates unfiltered
  E: Return-magnitude screens at 0.25, 0.35, 0.50
  F: Issuer exclusion (BMPS.MI and TPEIR.AT)
  G: Exclude banks with market cap < EUR 1bn
  H: VAR(1) standardized innovations
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    BANK_TICKERS, MARKET_TICKER, MARKET_CAPS, ALL_TICKERS,
    build_baseline_dataset, compute_log_returns, apply_calendar_screen,
    download_prices,
)
from src.garch_filter import fit_garch_filter
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    compute_bootstrap_se, wald_bh_selection, descriptive_edge_count,
)
from src.systemic_scores import compute_total_effect_matrix, compute_systemic_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
BOOT_SEED   = 731
N_BOOT      = int(os.environ.get("N_BOOT_EXP6", "200"))   # reduced for robustness module speed
MATERIALITY = 0.05


def run_robustness_module(
    Z: pd.DataFrame,
    tickers: list[str],
    bank_tickers: list[str],
    module_name: str,
    n_boot: int = N_BOOT,
) -> dict:
    """Common robustness template: estimate network and scores."""
    print(f"\n  [Module {module_name}] T={len(Z)}, n={len(tickers)}")
    Z_arr = Z.values
    n = len(tickers)

    # DirectLiNGAM
    try:
        np.random.seed(GLOBAL_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lingam
            model = lingam.DirectLiNGAM(measure="pwling")
            model.fit(Z_arr)
        causal_order = list(model.causal_order_)
        B0 = fit_predecessor_regressions(Z_arr, causal_order)
    except Exception as exc:
        logger.warning("DirectLiNGAM failed for module %s: %s", module_name, exc)
        return {"module": module_name, "status": "failed", "error": str(exc)}

    n_order_allowed = n * (n - 1) // 2
    n_descriptive = descriptive_edge_count(B0, threshold=MATERIALITY)

    # Bootstrap selection
    try:
        SE = compute_bootstrap_se(Z_arr, causal_order, n_boot=n_boot,
                                   block_length=10, seed=BOOT_SEED)
        sel = wald_bh_selection(B0, SE, causal_order, q_bh=0.05,
                                 materiality_threshold=MATERIALITY)
        B0_sel = sel["B0_selected_bh"]
        n_unadj = sel["n_unadj_sig"]
        n_bh    = sel["n_bh"]
        n_bh_mat = sel["n_bh_mat"]
        n_by_mat = sel["n_by_mat"]
    except Exception as exc:
        logger.warning("Selection failed for module %s: %s", module_name, exc)
        return {
            "module": module_name, "status": "selection_failed",
            "n_obs": len(Z), "n_vars": n, "n_order_allowed": n_order_allowed,
            "n_descriptive": n_descriptive,
        }

    # Systemic scores (bank-only)
    avail_bank_idx = [tickers.index(t) for t in bank_tickers if t in tickers]
    avail_banks    = [tickers[idx] for idx in avail_bank_idx]
    B0_bank = B0_sel[np.ix_(avail_bank_idx, avail_bank_idx)]
    try:
        C_B = compute_total_effect_matrix(B0_bank, avail_banks)
        scores = compute_systemic_scores(C_B, MARKET_CAPS, avail_banks)
        top5_sis = scores.nlargest(5, "SIS").index.tolist()
        top5_svs = scores.nlargest(5, "SVS").index.tolist()
    except Exception as exc:
        top5_sis = []
        top5_svs = []

    result = {
        "module": module_name,
        "status": "ok",
        "n_obs": len(Z),
        "n_vars": n,
        "n_order_allowed": n_order_allowed,
        "n_descriptive": n_descriptive,
        "n_unadj_sig": n_unadj,
        "n_bh_selected": n_bh,
        "n_bh_materiality": n_bh_mat,
        "n_by_materiality": n_by_mat,
        "top5_sis": top5_sis,
        "top5_svs": top5_svs,
    }
    print(f"    n_obs={len(Z)}, descriptive={n_descriptive}, BH+mat={n_bh_mat}")
    return result


def run_exp_6(dataset: dict | None = None) -> list[dict]:
    """Run all robustness modules."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 6: Sensitivity and Robustness Modules ===")

    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)

    prices       = dataset["prices"]
    raw_returns  = dataset["raw_returns"]
    baseline_ret = dataset["baseline_returns"]

    tickers_all = list(baseline_ret.columns)
    bank_tickers_avail = [t for t in BANK_TICKERS if t in tickers_all]

    results = []

    # ── Baseline reference ────────────────────────────────────────────────────
    print("\n[EXP6] Module BASELINE: GARCH-filtered screened 2016+ sample")
    garch_base = fit_garch_filter(baseline_ret, seed=GLOBAL_SEED)
    Z_base = garch_base["Z"]
    base_res = run_robustness_module(Z_base, tickers_all, bank_tickers_avail, "BASELINE")
    results.append(base_res)

    # ── Module A: Unfiltered same-sample ──────────────────────────────────────
    print("\n[EXP6] Module A: Unfiltered (no GARCH) same-sample returns")
    Z_A = baseline_ret.copy()
    for col in Z_A.columns:
        s = Z_A[col]
        mu, sd = s.mean(), s.std(ddof=1)
        Z_A[col] = (s - mu) / (sd if sd > 0 else 1.0)
    res_A = run_robustness_module(Z_A, tickers_all, bank_tickers_avail, "A_unfiltered")
    results.append(res_A)

    # ── Module B: Calendar-screened including 2015 ────────────────────────────
    print("\n[EXP6] Module B: Calendar screen including 2015 intervals")
    # Remove only pre-Nov-2015 (align with first available return)
    post_nov2015 = raw_returns[raw_returns.index >= pd.Timestamp("2015-11-20")]
    screened_B = apply_calendar_screen(post_nov2015, prices)
    print(f"  Module B sample size: {len(screened_B)}")
    if len(screened_B) > 100:
        garch_B = fit_garch_filter(screened_B, seed=GLOBAL_SEED)
        Z_B = garch_B["Z"]
        res_B = run_robustness_module(Z_B, tickers_all, bank_tickers_avail, "B_incl2015_screened")
    else:
        res_B = {"module": "B_incl2015_screened", "status": "insufficient_data"}
    results.append(res_B)

    # ── Module C: 2016+ without calendar screen ───────────────────────────────
    print("\n[EXP6] Module C: 2016+ without calendar screen")
    post2016_unscreened = raw_returns[raw_returns.index >= pd.Timestamp("2016-01-01")]
    garch_C = fit_garch_filter(post2016_unscreened, seed=GLOBAL_SEED)
    Z_C = garch_C["Z"]
    res_C = run_robustness_module(Z_C, tickers_all, bank_tickers_avail, "C_nocalendar")
    results.append(res_C)

    # ── Module D: All aligned dates, unfiltered ───────────────────────────────
    print("\n[EXP6] Module D: All aligned dates, unfiltered")
    Z_D = raw_returns.copy()
    for col in Z_D.columns:
        s = Z_D[col]
        mu, sd = s.mean(), s.std(ddof=1)
        Z_D[col] = (s - mu) / (sd if sd > 0 else 1.0)
    res_D = run_robustness_module(Z_D, tickers_all, bank_tickers_avail, "D_all_unfiltered")
    results.append(res_D)

    # ── Module E: Return-magnitude screens ────────────────────────────────────
    print("\n[EXP6] Module E: Return-magnitude screens (0.25, 0.35, 0.50)")
    # Rule: exclude any retained interval where any asset's |return| > threshold
    for threshold in [0.25, 0.35, 0.50]:
        abs_max = baseline_ret.abs().max(axis=1)
        screened_E = baseline_ret[abs_max <= threshold]
        print(f"  Threshold={threshold}: retained {len(screened_E)}/{len(baseline_ret)}")
        if len(screened_E) >= 100:
            garch_E = fit_garch_filter(screened_E, seed=GLOBAL_SEED)
            Z_E = garch_E["Z"]
            res_E = run_robustness_module(
                Z_E, tickers_all, bank_tickers_avail,
                f"E_return_screen_{threshold}"
            )
        else:
            res_E = {"module": f"E_return_screen_{threshold}", "status": "insufficient_data"}
        results.append(res_E)

    # ── Module F: Issuer exclusion (BMPS.MI, TPEIR.AT) ───────────────────────
    print("\n[EXP6] Module F: Exclude BMPS.MI and TPEIR.AT")
    exclude_F  = {"BMPS.MI", "TPEIR.AT"}
    tickers_F  = [t for t in tickers_all if t not in exclude_F]
    banks_F    = [t for t in bank_tickers_avail if t not in exclude_F]
    if len(tickers_F) > 5:
        baseline_F = baseline_ret[tickers_F]
        garch_F = fit_garch_filter(baseline_F, seed=GLOBAL_SEED)
        Z_F = garch_F["Z"]
        res_F = run_robustness_module(Z_F, tickers_F, banks_F, "F_excl_bmps_tpeir")
    else:
        res_F = {"module": "F_excl_bmps_tpeir", "status": "skipped"}
    results.append(res_F)

    # ── Module G: Exclude market cap < EUR 1bn ────────────────────────────────
    print("\n[EXP6] Module G: Exclude banks with market cap < EUR 1bn")
    exclude_G  = {t for t in BANK_TICKERS if MARKET_CAPS.get(t, 0) < 1.0}
    tickers_G  = [t for t in tickers_all if t not in exclude_G or t == MARKET_TICKER]
    banks_G    = [t for t in bank_tickers_avail if t not in exclude_G]
    print(f"  Excluding {len(exclude_G)} low-cap banks: {sorted(exclude_G)}")
    if len(tickers_G) > 5:
        baseline_G = baseline_ret[[t for t in tickers_G if t in baseline_ret.columns]]
        tickers_G_avail = list(baseline_G.columns)
        banks_G_avail = [t for t in banks_G if t in tickers_G_avail]
        garch_G = fit_garch_filter(baseline_G, seed=GLOBAL_SEED)
        Z_G = garch_G["Z"]
        res_G = run_robustness_module(Z_G, tickers_G_avail, banks_G_avail, "G_excl_lowcap")
    else:
        res_G = {"module": "G_excl_lowcap", "status": "skipped"}
    results.append(res_G)

    # ── Module H: VAR(1) standardized innovations ─────────────────────────────
    print("\n[EXP6] Module H: VAR(1) standardized innovations")
    try:
        from statsmodels.tsa.api import VAR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_mod = VAR(baseline_ret.values)
            var_res = var_mod.fit(maxlags=1, ic=None, trend="c")
        residuals_H = pd.DataFrame(
            var_res.resid,
            index=baseline_ret.index[1:],
            columns=tickers_all,
        )
        # Standardize residuals
        for col in residuals_H.columns:
            s = residuals_H[col]
            mu, sd = s.mean(), s.std(ddof=1)
            residuals_H[col] = (s - mu) / (sd if sd > 0 else 1.0)
        print(f"  VAR(1) residuals shape: {residuals_H.shape}")
        garch_H = fit_garch_filter(residuals_H, seed=GLOBAL_SEED)
        Z_H = garch_H["Z"]
        res_H = run_robustness_module(Z_H, tickers_all, bank_tickers_avail, "H_var1_innovations")
    except Exception as exc:
        logger.warning("Module H failed: %s", exc)
        res_H = {"module": "H_var1_innovations", "status": "failed", "error": str(exc)}
    results.append(res_H)

    # ── Consolidate and plot ──────────────────────────────────────────────────
    print("\n=== EXP 6: Robustness Summary ===")
    summary_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(RESULTS_DIR / "exp6_robustness_summary.csv", index=False)

    # Save full results
    with open(RESULTS_DIR / "exp6_robustness_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("[OUTPUT] Robustness summary saved to results/exp6_robustness_summary.csv")
    print("[OUTPUT] Full results saved to results/exp6_robustness_full.json")

    _plot_robustness_summary(summary_df)

    return results


def _plot_robustness_summary(summary_df: pd.DataFrame) -> None:
    """Bar chart comparing BH+materiality edge counts across modules."""
    if "n_bh_materiality" not in summary_df.columns:
        return
    valid = summary_df.dropna(subset=["n_bh_materiality"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(valid))
    bars = ax.bar(x, valid["n_bh_materiality"], color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(valid["module"].tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("BH+materiality selected edges", fontsize=11)
    ax.set_title("Robustness: Selected Edge Counts Across Modules\n"
                 "(baseline ~214 edges)", fontsize=12)
    ax.axhline(214, color="red", linestyle="--", linewidth=1.5, label="Baseline target (214)")
    ax.legend()
    for bar, val in zip(bars, valid["n_bh_materiality"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(val)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_robustness_edgecounts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Robustness edge count plot saved to results/exp6_robustness_edgecounts.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_6(dataset)
    print("\n=== EXP 6 COMPLETE ===")
