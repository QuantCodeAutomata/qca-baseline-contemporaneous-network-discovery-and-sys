"""
Experiment 7: Recursiveness diagnostics via ICA-Cholesky distance
and backward-mass measures.

Assesses whether the baseline standardized data are consistent with
a recursive acyclic contemporaneous structure.
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

from src.data_loader import BANK_TICKERS, build_baseline_dataset
from src.garch_filter import fit_garch_filter
from src.lingam_utils import estimate_direct_lingam, fit_predecessor_regressions
from src.recursiveness import (
    cholesky_factor,
    run_fastica,
    align_ica_to_cholesky,
    random_rotation_reference,
    ilmonen_distance,
    backward_mass,
    acyclic_reference_backward_mass,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
N_ICA_STARTS     = 10
N_ROT_REFERENCE  = 10_000
N_RANDOM_ORDERS  = 5_000
N_EXACT_ACYCLIC  = 20


def run_exp_7(Z_baseline: pd.DataFrame | None = None, dataset: dict | None = None) -> dict:
    """Run the recursiveness diagnostics experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 7: Recursiveness Diagnostics ===")

    # Load data
    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    if Z_baseline is None:
        z_path = DATA_DIR / "Z_baseline.parquet"
        if z_path.exists():
            Z_baseline = pd.read_parquet(z_path)
        else:
            garch_res = fit_garch_filter(dataset["baseline_returns"], seed=GLOBAL_SEED)
            Z_baseline = garch_res["Z"]

    tickers = list(Z_baseline.columns)
    Z_arr = Z_baseline.values
    n = Z_arr.shape[1]

    # ── Step 1: Recover baseline order ────────────────────────────────────────
    print("\n[EXP7] Recovering baseline DirectLiNGAM order...")
    causal_path = DATA_DIR / "exp1_causal_order.csv"
    if causal_path.exists():
        causal_order = list(pd.read_csv(causal_path, header=None)[1].astype(int))
        print(f"  Loaded causal order (first 5): {causal_order[:5]}")
    else:
        lingam_res = estimate_direct_lingam(Z_arr, seed=GLOBAL_SEED)
        causal_order = lingam_res["order"]

    # ── Step 3 & 4: Reorder data and compute Cholesky factor ──────────────────
    print("\n[EXP7] Computing sample covariance and Cholesky factor...")
    # Reorder Z according to DirectLiNGAM causal order
    Z_reordered = Z_arr[:, causal_order]
    D_chol = cholesky_factor(Z_arr, causal_order)
    print(f"  Cholesky factor shape: {D_chol.shape}")

    # ── Step 5: FastICA multi-start ────────────────────────────────────────────
    print(f"\n[EXP7] Running FastICA with {N_ICA_STARTS} starts...")
    mixing_matrices = run_fastica(Z_reordered, n_init=N_ICA_STARTS, seed=GLOBAL_SEED)
    ica_results = [
        {
            "converged": True,
            "mixing_matrix": C,
            "components": np.linalg.pinv(C),
            "start_idx": idx,
        }
        for idx, C in enumerate(mixing_matrices)
    ]
    print(f"  Converged starts: {sum(r['converged'] for r in ica_results)}/{N_ICA_STARTS}")

    # ── Step 6: Signed-permutation alignment ──────────────────────────────────
    print("\n[EXP7] Aligning ICA mixing matrices to Cholesky factor...")
    alignment_dists = []
    W_matrices = []
    for res in ica_results:
        if not res["converged"]:
            continue
        C_ica = res["mixing_matrix"]
        C_aligned, dist = align_ica_to_cholesky(C_ica, D_chol)
        alignment_dists.append(dist)
        W_ica = res["components"]  # unmixing
        W_matrices.append((W_ica, dist))
        print(f"  Start {res['start_idx']}: Frobenius distance = {dist:.4f}")

    if not alignment_dists:
        print("  WARNING: No ICA starts converged; skipping alignment diagnostics")
        return {"exp_id": "exp_7", "status": "no_ica_convergence"}

    best_dist = min(alignment_dists)
    median_dist = float(np.median(alignment_dists))
    print(f"\n  Best aligned distance: {best_dist:.4f}")
    print(f"  Median aligned distance: {median_dist:.4f}")

    # ── Step 7: Random rotation reference ────────────────────────────────────
    print(f"\n[EXP7] Building random-rotation reference ({N_ROT_REFERENCE} draws)...")
    # Use best ICA result for reference comparison
    best_idx = alignment_dists.index(best_dist)
    best_ica = [r for r in ica_results if r["converged"]][best_idx]
    C_ica_best = best_ica["mixing_matrix"]

    ref_dist_rot_ica, ref_dist_rot_chol = random_rotation_reference(
        C_ica_best, D_chol, n_rotations=N_ROT_REFERENCE, seed=GLOBAL_SEED
    )

    pct_ica  = float(np.mean(ref_dist_rot_ica  >= best_dist))
    pct_chol = float(np.mean(ref_dist_rot_chol >= best_dist))
    print(f"  Fraction of rotated ICA distances >= observed:  {pct_ica:.4f}")
    print(f"  Fraction of rotated Chol distances >= observed: {pct_chol:.4f}")
    print("  (Small fraction → observed distance unusually small → supports recursiveness)")

    # ── Step 8: Ilmonen distance index ────────────────────────────────────────
    print("\n[EXP7] Computing Ilmonen minimum-distance index...")
    il_index = ilmonen_distance(C_ica_best, D_chol)
    print(f"  Ilmonen index: {il_index:.4f}")

    # ── Step 9-10: Backward mass ──────────────────────────────────────────────
    print("\n[EXP7] Computing backward mass for baseline order...")
    W_best = best_ica["components"]
    # Normalize diagonal to unit
    W_norm = W_best / (np.diag(W_best)[:, None] + 1e-15)
    bm_baseline = backward_mass(W_norm, causal_order)
    print(f"  Backward mass (baseline order): {bm_baseline:.4f}")
    print("  (Lower backward mass → more consistent with recursive order)")

    # ── Step 11: Random-order reference distribution ──────────────────────────
    print(f"\n[EXP7] Generating {N_RANDOM_ORDERS} random-order backward mass values...")
    rng = np.random.RandomState(GLOBAL_SEED)
    random_bm_values = []
    for _ in range(N_RANDOM_ORDERS):
        rand_order = list(rng.permutation(n))
        bm_rand = backward_mass(W_norm, rand_order)
        random_bm_values.append(bm_rand)

    pct_lower_than_baseline = float(np.mean(np.array(random_bm_values) <= bm_baseline))
    rank_baseline_bm = float(np.mean(np.array(random_bm_values) < bm_baseline))
    print(f"  Fraction of random orders with BM <= baseline: {pct_lower_than_baseline:.4f}")
    print(f"  Rank of baseline (fraction below):             {rank_baseline_bm:.4f}")
    print(f"  (Small fraction → baseline order unusually consistent with recursiveness)")

    # ── Step 12: Exact acyclic simulated reference ────────────────────────────
    print(f"\n[EXP7] Simulating {N_EXACT_ACYCLIC} exact acyclic reference systems...")
    acyclic_bm_values = list(acyclic_reference_backward_mass(
        n_vars=n,
        n_systems=N_EXACT_ACYCLIC,
        T=Z_arr.shape[0],
        seed=GLOBAL_SEED,
    ))

    mean_acyclic_bm = float(np.mean(acyclic_bm_values)) if acyclic_bm_values else np.nan
    print(f"  Mean backward mass of {len(acyclic_bm_values)} exact acyclic systems: {mean_acyclic_bm:.4f}")

    # ── Step 13: Summarize across ICA starts ─────────────────────────────────
    print(f"\n[EXP7] Across-start stability of backward mass:")
    bm_across_starts = []
    for W_s, _ in W_matrices:
        W_n = W_s / (np.diag(W_s)[:, None] + 1e-15)
        bm = backward_mass(W_n, causal_order)
        bm_across_starts.append(bm)
    print(f"  BM values: {[f'{b:.4f}' for b in bm_across_starts]}")
    print(f"  Range: [{min(bm_across_starts):.4f}, {max(bm_across_starts):.4f}]")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_backward_mass_distribution(
        bm_baseline, random_bm_values, mean_acyclic_bm, bm_across_starts
    )
    _plot_rotation_reference(best_dist, ref_dist_rot_ica, ref_dist_rot_chol)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_7",
        "n_ica_starts": N_ICA_STARTS,
        "n_converged_starts": sum(r["converged"] for r in ica_results),
        "best_ica_cholesky_dist": best_dist,
        "median_ica_cholesky_dist": median_dist,
        "ilmonen_index": il_index,
        "backward_mass_baseline": bm_baseline,
        "pct_random_orders_le_baseline_bm": pct_lower_than_baseline,
        "mean_acyclic_bm": mean_acyclic_bm,
        "n_acyclic_sims": len(acyclic_bm_values),
        "pct_rot_ica_dist_ge_obs": pct_ica,
        "pct_rot_chol_dist_ge_obs": pct_chol,
        "bm_across_starts": bm_across_starts,
    }
    with open(RESULTS_DIR / "exp7_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp7_metadata.json")

    return meta


def _plot_backward_mass_distribution(
    bm_baseline: float,
    random_bm: list[float],
    mean_acyclic: float,
    bm_starts: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(random_bm, bins=50, color="lightsteelblue", edgecolor="white",
            label=f"Random orders (N={len(random_bm)})")
    ax.axvline(bm_baseline, color="red", linewidth=2, linestyle="-",
               label=f"DirectLiNGAM order BM={bm_baseline:.4f}")
    if np.isfinite(mean_acyclic):
        ax.axvline(mean_acyclic, color="green", linewidth=2, linestyle="--",
                   label=f"Mean exact-acyclic BM={mean_acyclic:.4f}")
    for bm_s in bm_starts:
        ax.axvline(bm_s, color="orange", linewidth=1, alpha=0.4)
    ax.set_xlabel("Backward mass R(W, order)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Backward Mass: DirectLiNGAM vs Random Orders\n"
                 "(lower = more consistent with recursive structure)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_backward_mass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Backward mass plot saved to results/exp7_backward_mass.png")


def _plot_rotation_reference(
    best_dist: float,
    ref_ica: list[float],
    ref_chol: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ref, label in zip(axes, [ref_ica, ref_chol], ["Rotated ICA", "Rotated Cholesky"]):
        ax.hist(ref, bins=50, color="lightcoral", edgecolor="white")
        ax.axvline(best_dist, color="navy", linewidth=2.5,
                   label=f"Observed dist={best_dist:.4f}")
        ax.set_xlabel("Frobenius distance to Cholesky", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Rotation reference: {label}\n"
                     f"(fraction >= observed: {np.mean(np.array(ref) >= best_dist):.4f})",
                     fontsize=10)
        ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_rotation_reference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Rotation reference plot saved to results/exp7_rotation_reference.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_7(dataset=dataset)
    print("\n=== EXP 7 COMPLETE ===")
