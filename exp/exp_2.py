"""
Experiment 2: Recovered-shock independence support via exact pairwise HSIC
with circular-shift reference distribution.

Compares:
- Filtered (GARCH-standardized) baseline Z_t
- Unfiltered (raw standardized) returns on the same screened sample
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BANK_TICKERS, MARKET_TICKER, build_baseline_dataset
from src.garch_filter import fit_garch_filter
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    recover_shocks,
)
from src.hsic_tests import test_all_pairs_hsic, mean_squared_shock_correlation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
# Reduce parallelism if running in restricted environment
N_JOBS = int(os.environ.get("HSIC_N_JOBS", "1"))


def build_inputs(dataset: dict) -> dict:
    """Build filtered and unfiltered inputs on the screened baseline sample."""
    baseline_ret = dataset["baseline_returns"]
    available_tickers = list(baseline_ret.columns)

    # ── Filtered input ────────────────────────────────────────────────────────
    print("\n[EXP2] Building filtered (GARCH) input...")
    garch_result = fit_garch_filter(baseline_ret, seed=GLOBAL_SEED)
    Z_filtered = garch_result["Z"]

    # ── Unfiltered input ──────────────────────────────────────────────────────
    print("[EXP2] Building unfiltered input...")
    # Mean-center and unit-variance standardize (no GARCH)
    Z_unfiltered_df = baseline_ret.copy()
    for col in available_tickers:
        s = Z_unfiltered_df[col]
        mu, sd = s.mean(), s.std(ddof=1)
        Z_unfiltered_df[col] = (s - mu) / (sd if sd > 0 else 1.0)

    return {
        "Z_filtered": Z_filtered,
        "Z_unfiltered": Z_unfiltered_df,
        "tickers": available_tickers,
    }


def run_lingam_and_recover_shocks(
    Z: pd.DataFrame,
    label: str,
) -> dict:
    """Estimate DirectLiNGAM and recover unpruned structural shocks."""
    print(f"\n[EXP2] Estimating DirectLiNGAM on {label} input ...")
    Z_arr = Z.values
    lingam_res = estimate_direct_lingam(Z_arr, seed=GLOBAL_SEED)
    causal_order = lingam_res["order"]
    B0 = fit_predecessor_regressions(Z_arr, causal_order)

    # Recover shocks using UNPRUNED B0 (as required by paper)
    shocks = recover_shocks(Z_arr, B0)
    print(f"  Recovered shocks shape: {shocks.shape}")
    return {
        "causal_order": causal_order,
        "B0": B0,
        "shocks": shocks,
    }


def run_hsic_analysis(
    shocks: np.ndarray,
    label: str,
) -> pd.DataFrame:
    """Run pairwise HSIC tests with circular-shift p-values."""
    print(f"\n[EXP2] Running HSIC tests for {label} ({shocks.shape[1]} vars) ...")
    print("  NOTE: 1326 pairs × 1951 shifts = computationally intensive")
    print("  Running with exact circular shifts (min distance=20) ...")

    df = test_all_pairs_hsic(
        shocks,
        min_shift_distance=20,
        q_bh=0.05,
        n_jobs=N_JOBS,
    )
    return df


def run_exp_2(dataset: dict | None = None) -> dict:
    """Run the full HSIC independence experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 2: HSIC Independence Support ===")

    # Load or build dataset
    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)

    inputs = build_inputs(dataset)
    Z_filtered   = inputs["Z_filtered"]
    Z_unfiltered = inputs["Z_unfiltered"]

    results = {}

    for label, Z_df in [("filtered", Z_filtered), ("unfiltered", Z_unfiltered)]:
        lingam_out = run_lingam_and_recover_shocks(Z_df, label)
        shocks = lingam_out["shocks"]

        # HSIC tests
        hsic_df = run_hsic_analysis(shocks, label)

        n_pairs   = len(hsic_df)
        n_raw_rej = int(hsic_df["reject_raw"].sum())
        n_bh_rej  = int(hsic_df["reject_bh"].sum())
        n_by_rej  = int(hsic_df["reject_by"].sum())

        mean_sq_corr = mean_squared_shock_correlation(shocks)

        print(f"\n  [{label.upper()}] HSIC summary:")
        print(f"    Tested pairs:               {n_pairs}")
        print(f"    Unadjusted p<0.05:          {n_raw_rej}  "
              f"(target filtered: ~219, unfiltered: N/A)")
        print(f"    BH rejections (q=0.05):     {n_bh_rej}  "
              f"(target filtered: ~14)")
        print(f"    BY rejections (q=0.05):     {n_by_rej}  "
              f"(target filtered: ~0)")
        print(f"    Mean squared shock corr:    {mean_sq_corr:.4f}  "
              f"(target filtered: ~0.020)")

        # Independence criterion: fail if BH > 0
        independence_support = (n_bh_rej == 0)
        print(f"    BH independence support:    {'PASS' if independence_support else 'FAIL'}")
        print(f"    (Filtered baseline expected to FAIL per paper)")

        # Save
        hsic_df.to_csv(RESULTS_DIR / f"exp2_hsic_{label}.csv", index=False)

        results[label] = {
            "n_pairs": n_pairs,
            "n_raw_rej": n_raw_rej,
            "n_bh_rej": n_bh_rej,
            "n_by_rej": n_by_rej,
            "mean_sq_corr": float(mean_sq_corr),
            "independence_support_bh": independence_support,
            "hsic_df": hsic_df,
        }

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n=== EXP 2: Filtered vs Unfiltered Comparison ===")
    filt = results["filtered"]
    unfl = results["unfiltered"]

    print(f"  BH rejections:  filtered={filt['n_bh_rej']}  vs  unfiltered={unfl['n_bh_rej']}  (target: 14 vs 276)")
    print(f"  Mean sq corr:   filtered={filt['mean_sq_corr']:.4f}  vs  unfiltered={unfl['mean_sq_corr']:.4f}  (target: 0.020 vs 0.045)")
    dependence_reduced = filt["n_bh_rej"] < unfl["n_bh_rej"]
    print(f"  Filtering reduces dependence: {dependence_reduced}  (expected: True)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_pvalue_distributions(results)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_2",
        "filtered": {k: v for k, v in filt.items() if k != "hsic_df"},
        "unfiltered": {k: v for k, v in unfl.items() if k != "hsic_df"},
        "dependence_reduced_by_filtering": bool(dependence_reduced),
    }
    with open(RESULTS_DIR / "exp2_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp2_metadata.json")

    return results


def _plot_pvalue_distributions(results: dict) -> None:
    """Plot p-value histograms for filtered and unfiltered HSIC tests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, color) in zip(axes, [("filtered", "steelblue"), ("unfiltered", "coral")]):
        if label not in results:
            continue
        hsic_df = results[label]["hsic_df"]
        if hsic_df.empty:
            continue
        pvals = hsic_df["pvalue_raw"].values
        ax.hist(pvals, bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05")
        n_bh = results[label]["n_bh_rej"]
        ax.set_title(
            f"HSIC p-values ({label})\nBH rejections: {n_bh} / {len(pvals)}",
            fontsize=12,
        )
        ax.set_xlabel("p-value (circular shift)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_hsic_pvalues.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] HSIC p-value plot saved to results/exp2_hsic_pvalues.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_2(dataset)
    print("\n=== EXP 2 COMPLETE ===")
