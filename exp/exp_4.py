"""
Experiment 4: Tail-dependence diagnostics, temporal holdout evaluation,
and alternative-order comparator.

Evaluates CoLift/DirLift for selected edges in sample and out of sample,
using a training/test split with training-estimated network applied to test data.
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

from src.data_loader import BANK_TICKERS, MARKET_TICKER, MARKET_CAPS, build_baseline_dataset
from src.garch_filter import fit_garch_filter, extend_garch_to_test
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    compute_bootstrap_se, wald_bh_selection,
    moving_block_bootstrap_indices, recover_shocks,
)
from src.tail_diagnostics import (
    compute_tail_indicators, compute_shock_indicators,
    compute_colift, compute_dirlift,
    compute_all_pair_lifts, permutation_diagnostic,
    compute_colift_matrix, block_bootstrap_gap_interval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
BOOT_SEED   = 731
# Training cutoff
TRAIN_END   = "2021-12-31"
N_BOOT_COEF = int(os.environ.get("N_BOOT_EXP4", "1000"))
N_ALT_ORDERS = int(os.environ.get("N_ALT_ORDERS", "200"))
MATERIALITY = 0.05
TAIL_Q      = 0.05


def run_exp_4(dataset: dict | None = None) -> dict:
    """Run the full tail diagnostics and holdout experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 4: Tail Diagnostics and Holdout Evaluation ===")

    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)

    baseline_ret = dataset["baseline_returns"]
    tickers = list(baseline_ret.columns)
    avail_banks = [t for t in BANK_TICKERS if t in tickers]

    # ── Training/test split ──────────────────────────────────────────────────
    train_mask = baseline_ret.index <= pd.Timestamp(TRAIN_END)
    test_mask  = ~train_mask

    train_ret = baseline_ret[train_mask]
    test_ret  = baseline_ret[test_mask]

    T_train = len(train_ret)
    T_test  = len(test_ret)
    print(f"  Training: {T_train} observations  (target: 1267)")
    print(f"  Test:     {T_test} observations   (target: 723)")

    # ── PART A: Full-sample baseline ─────────────────────────────────────────
    print("\n=== EXP 4 Part A: Full-Sample Tail Diagnostics ===")
    z_path = DATA_DIR / "Z_baseline.parquet"
    if z_path.exists():
        Z_full = pd.read_parquet(z_path)
    else:
        garch_res = fit_garch_filter(baseline_ret, seed=GLOBAL_SEED)
        Z_full = garch_res["Z"]

    # Load or recompute B0 selected
    b0_path = DATA_DIR / "exp1_B0_selected.npy"
    causal_path = DATA_DIR / "exp1_causal_order.csv"
    if b0_path.exists() and causal_path.exists():
        B0_full_sel = np.load(b0_path)
        causal_order_full = list(pd.read_csv(causal_path, header=None)[1].astype(int))
    else:
        # Recompute
        lingam_res = estimate_direct_lingam(Z_full.values, seed=GLOBAL_SEED)
        causal_order_full = lingam_res["order"]
        B0_full = fit_predecessor_regressions(Z_full.values, causal_order_full)
        SE_full = compute_bootstrap_se(Z_full.values, causal_order_full,
                                        n_boot=N_BOOT_COEF, block_length=10, seed=BOOT_SEED)
        sel = wald_bh_selection(B0_full, SE_full, causal_order_full,
                                 q_bh=0.05, materiality_threshold=MATERIALITY)
        B0_full_sel = sel["B0_selected_bh"]

    shocks_full = recover_shocks(Z_full.values, B0_full_sel)

    # Get selected edges (full sample)
    n_full = Z_full.shape[1]
    full_edges = [(i, j) for i in range(n_full) for j in range(n_full) if B0_full_sel[i, j] != 0]
    print(f"  Full-sample selected edges: {len(full_edges)}")

    # Tail indicators
    D_full = compute_tail_indicators(Z_full.values, TAIL_Q)
    S_full = compute_shock_indicators(shocks_full, TAIL_Q)

    # CoLift for selected edges vs non-links
    full_colift_matrix = compute_colift_matrix(Z_full.values, TAIL_Q)

    colift_linked  = [full_colift_matrix[i, j] for (i, j) in full_edges if np.isfinite(full_colift_matrix[i, j])]
    # Non-links (genuine non-links, exclude reverse edges too)
    edge_set = set(full_edges) | {(j, i) for (i, j) in full_edges}
    non_links = [(i, j) for i in range(n_full) for j in range(n_full)
                 if i != j and (i, j) not in edge_set]
    non_links_sample = non_links[:min(500, len(non_links))]  # sample for speed
    colift_nonlinked = [full_colift_matrix[i, j] for (i, j) in non_links_sample
                        if np.isfinite(full_colift_matrix[i, j])]

    mean_colift_linked   = float(np.mean(colift_linked))   if colift_linked   else np.nan
    mean_colift_nonlinked = float(np.mean(colift_nonlinked)) if colift_nonlinked else np.nan
    print(f"  Mean CoLift (linked):     {mean_colift_linked:.4f}")
    print(f"  Mean CoLift (non-linked): {mean_colift_nonlinked:.4f}")

    # DirLift full sample
    fwd_lifts_full = []
    rev_lifts_full = []
    for (src, tgt) in full_edges:
        fwd = compute_dirlift(D_full, S_full, src, tgt)
        rev = compute_dirlift(D_full, S_full, tgt, src)
        if np.isfinite(fwd): fwd_lifts_full.append(fwd)
        if np.isfinite(rev): rev_lifts_full.append(rev)
    mean_fwd_full = float(np.mean(fwd_lifts_full)) if fwd_lifts_full else np.nan
    mean_rev_full = float(np.mean(rev_lifts_full)) if rev_lifts_full else np.nan
    print(f"  Full-sample forward DirLift:  {mean_fwd_full:.4f}")
    print(f"  Full-sample reverse DirLift:  {mean_rev_full:.4f}")
    print(f"  Full-sample gap (fwd-rev):    {mean_fwd_full - mean_rev_full:.4f}")

    # Permutation diagnostic
    print("\n  Running node-label permutation diagnostic (1000 permutations)...")
    perm_result = permutation_diagnostic(
        full_colift_matrix, full_edges, n_permutations=1000, seed=GLOBAL_SEED
    )
    print(f"  Permutation p-value: {perm_result['p_value']:.4f}  (smallest possible: {1/1001:.5f})")

    # ── PART B: Temporal holdout ──────────────────────────────────────────────
    print("\n=== EXP 4 Part B: Temporal Holdout ===")

    # Train-only GARCH + DirectLiNGAM
    print("[EXP4] Fitting GARCH on training data...")
    train_garch = fit_garch_filter(train_ret, seed=GLOBAL_SEED)
    Z_train = train_garch["Z"]
    garch_params_train = train_garch["garch_params"]

    print("[EXP4] Estimating DirectLiNGAM on training data...")
    np.random.seed(GLOBAL_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lingam
        train_model = lingam.DirectLiNGAM(measure="pwling")
        train_model.fit(Z_train.values)
    causal_order_train = list(train_model.causal_order_)
    B0_train = fit_predecessor_regressions(Z_train.values, causal_order_train)

    print(f"[EXP4] Bootstrap selection on training data (n_boot={N_BOOT_COEF})...")
    SE_train = compute_bootstrap_se(
        Z_train.values, causal_order_train,
        n_boot=N_BOOT_COEF, block_length=10, seed=BOOT_SEED
    )
    sel_train = wald_bh_selection(
        B0_train, SE_train, causal_order_train, q_bh=0.05, materiality_threshold=MATERIALITY
    )
    B0_train_sel = sel_train["B0_selected_bh"]
    n_train_edges = int(np.sum(B0_train_sel != 0))
    print(f"  Training selected edges: {n_train_edges}  (target: ~150)")

    # Training thresholds (5% quantiles from training data)
    Z_train_arr = Z_train.values
    shocks_train = recover_shocks(Z_train_arr, B0_train_sel)
    train_thresh_Z     = np.quantile(Z_train_arr,    TAIL_Q, axis=0)
    train_thresh_shock = np.quantile(shocks_train,   TAIL_Q, axis=0)

    # Extend GARCH to test
    print("[EXP4] Extending GARCH to test period...")
    test_garch = extend_garch_to_test(train_ret, test_ret, garch_params_train)
    Z_test = test_garch["Z_test"]

    # Apply training transformation to test data
    Z_test_arr = Z_test.values
    shocks_test = recover_shocks(Z_test_arr, B0_train_sel)

    # Apply training-defined thresholds to test
    D_test  = (Z_test_arr <= train_thresh_Z[None, :]).astype(int)
    S_test  = (shocks_test <= train_thresh_shock[None, :]).astype(int)

    # Holdout tail metrics
    train_edges = [(i, j) for i in range(n_full) for j in range(n_full)
                   if B0_train_sel[i, j] != 0]

    # Only bank indices for holdout
    avail_bank_idx = [tickers.index(t) for t in avail_banks if t in tickers]
    train_bank_edges = [(i, j) for (i, j) in train_edges
                        if i in avail_bank_idx and j in avail_bank_idx]

    # Holdout CoLift (symmetric)
    colift_matrix_test = compute_colift_matrix(Z_test_arr, TAIL_Q)
    holdout_colift_linked = [colift_matrix_test[i, j] for (i, j) in train_bank_edges
                             if np.isfinite(colift_matrix_test[i, j])]
    mean_holdout_colift = float(np.mean(holdout_colift_linked)) if holdout_colift_linked else np.nan
    print(f"  Holdout symmetric lower-tail lift: {mean_holdout_colift:.4f}  (target: ~1.41)")

    # Holdout DirLift
    fwd_lifts, rev_lifts = [], []
    for (src, tgt) in train_bank_edges:
        fwd = compute_dirlift(D_test, S_test, src, tgt)
        rev = compute_dirlift(D_test, S_test, tgt, src)
        if np.isfinite(fwd): fwd_lifts.append(fwd)
        if np.isfinite(rev): rev_lifts.append(rev)

    mean_holdout_fwd = float(np.mean(fwd_lifts)) if fwd_lifts else np.nan
    mean_holdout_rev = float(np.mean(rev_lifts)) if rev_lifts else np.nan
    holdout_gap = mean_holdout_fwd - mean_holdout_rev
    print(f"  Holdout forward DirLift:  {mean_holdout_fwd:.4f}  (target: ~4.39)")
    print(f"  Holdout reverse DirLift:  {mean_holdout_rev:.4f}  (target: ~2.37)")
    print(f"  Holdout gap (fwd-rev):    {holdout_gap:.4f}  (target: ~2.02)")

    # Block-bootstrap interval for gap
    print("[EXP4] Computing block-bootstrap interval for holdout gap...")
    boot_lo, boot_hi = block_bootstrap_gap_interval(
        Z_test_arr, shocks_test, train_bank_edges,
        train_thresh_Z, train_thresh_shock,
        block_length=10, n_boot=1000, seed=GLOBAL_SEED,
    )
    print(f"  Bootstrap 90% interval: [{boot_lo:.4f}, {boot_hi:.4f}]  (target: [1.63, 2.56])")

    # Stress-day exclusions
    print("\n[EXP4] Stress-day sensitivity analysis...")
    for threshold_n in [10, 20]:
        n_coexceedances = D_test.sum(axis=1)
        stress_mask = n_coexceedances < threshold_n
        if stress_mask.sum() >= 50:
            D_stress = D_test[stress_mask]
            S_stress = S_test[stress_mask]
            fwd_stress, rev_stress = [], []
            for (src, tgt) in train_bank_edges:
                fwd = compute_dirlift(D_stress, S_stress, src, tgt)
                rev = compute_dirlift(D_stress, S_stress, tgt, src)
                if np.isfinite(fwd) and np.isfinite(rev):
                    fwd_stress.append(fwd)
                    rev_stress.append(rev)
            mean_gap_stress = (float(np.mean(fwd_stress)) - float(np.mean(rev_stress))
                                if fwd_stress else np.nan)
            print(f"  Excluding >= {threshold_n} co-exceedances: "
                  f"gap = {mean_gap_stress:.4f}  (n_sessions = {stress_mask.sum()})")

    # ── PART C: Alternative-order comparator ──────────────────────────────────
    print(f"\n=== EXP 4 Part C: Alternative-Order Comparator ({N_ALT_ORDERS} orders) ===")
    alt_gaps = _run_alternative_order_comparator(
        Z_train_arr, Z_test_arr, D_test, S_test,
        train_thresh_Z, train_thresh_shock,
        tickers, avail_bank_idx, n_full, N_ALT_ORDERS
    )

    if len(alt_gaps) > 0:
        n_as_large = int(np.sum(np.array(alt_gaps) >= holdout_gap))
        alt_frac = n_as_large / len(alt_gaps)
        print(f"  Alt orders with gap >= DirectLiNGAM gap: {n_as_large}/{len(alt_gaps)} "
              f"= {alt_frac:.1%}  (target: ~42.0%)")
    else:
        alt_frac = np.nan
        print("  Alternative order comparator: no valid results")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_dirlift_comparison(mean_fwd_full, mean_rev_full,
                              mean_holdout_fwd, mean_holdout_rev)

    # ── Save results ───────────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_4",
        "T_train": T_train, "T_test": T_test,
        "n_train_selected_edges": n_train_edges,
        "full_sample_mean_fwd_dirlift": mean_fwd_full,
        "full_sample_mean_rev_dirlift": mean_rev_full,
        "holdout_symmetric_colift": mean_holdout_colift,
        "holdout_fwd_dirlift": mean_holdout_fwd,
        "holdout_rev_dirlift": mean_holdout_rev,
        "holdout_gap": holdout_gap,
        "holdout_boot_lo": boot_lo,
        "holdout_boot_hi": boot_hi,
        "alt_order_frac_as_large": float(alt_frac) if np.isfinite(alt_frac) else None,
        "n_alt_orders": N_ALT_ORDERS,
        "n_alt_valid_gaps": len(alt_gaps),
        "perm_pvalue": perm_result["p_value"],
    }
    with open(RESULTS_DIR / "exp4_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp4_metadata.json")

    return meta


def _run_alternative_order_comparator(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    D_test: np.ndarray,
    S_test: np.ndarray,
    train_thresh_Z: np.ndarray,
    train_thresh_shock: np.ndarray,
    tickers: list[str],
    bank_idx: list[int],
    n: int,
    n_alt: int,
) -> list[float]:
    """Generate alternative orders and compute holdout DirLift gaps."""
    rng = np.random.RandomState(GLOBAL_SEED + 42)

    # Market node last (bank nodes in any order)
    market_idx = tickers.index(MARKET_TICKER) if MARKET_TICKER in tickers else None
    bank_indices = [i for i in range(n) if i != market_idx]
    n_banks_local = len(bank_indices)

    alt_gaps = []

    for alt in range(n_alt):
        # Permute bank nodes; keep market fixed at position 0
        perm_banks = list(rng.permutation(n_banks_local))
        if market_idx is not None:
            alt_order = [market_idx] + [bank_indices[p] for p in perm_banks]
        else:
            alt_order = [bank_indices[p] for p in perm_banks]

        # Fit OLS predecessor regressions under this order
        B0_alt = fit_predecessor_regressions(Z_train, alt_order)

        # Selection (simplified: just materiality threshold for speed)
        B0_alt_sel = np.where(np.abs(B0_alt) >= 0.05, B0_alt, 0.0)

        # Training thresholds for this alt order
        shocks_train_alt = recover_shocks(Z_train, B0_alt_sel)
        thresh_shock_alt = np.quantile(shocks_train_alt, TAIL_Q, axis=0)

        # Test shocks under this order
        shocks_test_alt = recover_shocks(Z_test, B0_alt_sel)

        # Tail indicators using training thresholds
        D_alt = (Z_test <= train_thresh_Z[None, :]).astype(int)
        S_alt = (shocks_test_alt <= thresh_shock_alt[None, :]).astype(int)

        # Edges for this order
        alt_edges = [(i, j) for i in range(n) for j in range(n)
                     if B0_alt_sel[i, j] != 0 and i in bank_idx and j in bank_idx]

        if not alt_edges:
            continue

        fwd_alt, rev_alt = [], []
        for (src, tgt) in alt_edges[:100]:  # limit for speed
            fwd = compute_dirlift(D_alt, S_alt, src, tgt)
            rev = compute_dirlift(D_alt, S_alt, tgt, src)
            if np.isfinite(fwd) and np.isfinite(rev):
                fwd_alt.append(fwd)
                rev_alt.append(rev)

        if fwd_alt:
            gap = float(np.mean(fwd_alt) - np.mean(rev_alt))
            alt_gaps.append(gap)

    return alt_gaps


def _plot_dirlift_comparison(
    fwd_full: float, rev_full: float,
    fwd_hold: float, rev_hold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [0.5, 1.5, 3.0, 4.0]
    heights = [fwd_full, rev_full, fwd_hold, rev_hold]
    colors  = ["steelblue", "lightblue", "coral", "lightsalmon"]
    labels  = ["Forward\n(full sample)", "Reverse\n(full sample)",
               "Forward\n(holdout)", "Reverse\n(holdout)"]
    bars = ax.bar(x, heights, width=0.7, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("DirLift (shock-conditioned lower-tail lift)", fontsize=11)
    ax.set_title("Directional Tail Lift: Full Sample and Holdout\n"
                 "(higher forward vs reverse = directional asymmetry)", fontsize=12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar, h in zip(bars, heights):
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_dirlift_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] DirLift comparison saved to results/exp4_dirlift_comparison.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_4(dataset)
    print("\n=== EXP 4 COMPLETE ===")
