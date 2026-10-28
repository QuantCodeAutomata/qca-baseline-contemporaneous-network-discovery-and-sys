"""
Experiment 8: Appendix simulation study for edge recovery and volatility-filtering
effects under controlled designs.

Implements Monte Carlo study:
  Design 1: Homoskedastic recursive systems with independent non-Gaussian shocks
  Design 2: Heteroskedastic recursive systems (GARCH-type scaling)
  50 replications per design, burn-in 500, global seed 1121.
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

from src.simulation import (
    evaluate_edge_recovery,
    run_simulation_study,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
N_REPS      = int(os.environ.get("N_SIM_REPS", "50"))   # paper: 50 per design
BURN_IN     = 500


def run_exp_8() -> dict:
    """Run the full simulation study."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 8: Simulation Study ===")
    print(f"  N_REPS={N_REPS}, BURN_IN={BURN_IN}, GLOBAL_SEED={GLOBAL_SEED}")

    # ── Run simulation study ───────────────────────────────────────────────────
    print("\n[EXP8] Running Monte Carlo simulation study...")
    results_df = run_simulation_study(
        n_vars=10,
        T=1990,
        n_replications=N_REPS,
        burn_in=BURN_IN,
        seed=GLOBAL_SEED,
    )

    # ── Aggregate metrics per design ───────────────────────────────────────────
    summary_df = results_df.groupby("design").agg(
        n_reps=("replication", "count"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_orientation_acc=("orientation_acc", "mean"),
        mean_tp=("tp", "mean"),
        mean_fp=("fp", "mean"),
    ).reset_index()

    summary_df["estimator"] = summary_df["design"].apply(
        lambda d: "filtered" if d.endswith("-filtered") else "unfiltered"
    )

    # ── Print per-design summaries ─────────────────────────────────────────────
    for design_name in results_df["design"].unique():
        sub = results_df[results_df["design"] == design_name]
        metrics = {
            "mean_precision": sub["precision"].mean(),
            "mean_recall": sub["recall"].mean(),
            "mean_f1": sub["f1"].mean(),
            "mean_orientation_acc": sub["orientation_acc"].mean(),
            "mean_tp": sub["tp"].mean(),
            "mean_fp": sub["fp"].mean(),
        }
        print(f"\n  {design_name}:")
        _print_summary(metrics)

    # ── Compile summary table ──────────────────────────────────────────────────
    print("\n=== EXP 8: Simulation Summary Table ===")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    summary_df.to_csv(RESULTS_DIR / "exp8_simulation_summary.csv", index=False)
    print("[OUTPUT] Summary table saved to results/exp8_simulation_summary.csv")

    # ── Verify GARCH filter improves recovery ─────────────────────────────────
    print("\n=== EXP 8: Filtering Effect Analysis ===")
    all_designs = set(results_df["design"].unique())
    for design_name in sorted(all_designs):
        if design_name.endswith("-filtered"):
            unfilt_name = design_name.replace("-filtered", "-unfiltered")
            if unfilt_name in all_designs:
                f1_unfl = results_df[results_df["design"] == unfilt_name]["f1"].mean()
                f1_filt = results_df[results_df["design"] == design_name]["f1"].mean()
                f1_improves = (
                    f1_filt > f1_unfl
                    if (np.isfinite(f1_filt) and np.isfinite(f1_unfl))
                    else None
                )
                print(f"  {design_name} vs {unfilt_name}:")
                print(f"    F1 unfiltered: {f1_unfl:.4f}   F1 filtered: {f1_filt:.4f}")
                print(f"    Filtering improves F1: {f1_improves}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_simulation_results(summary_df)
    _plot_boxplots(results_df)

    # ── Save full results ──────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_8",
        "n_reps": N_REPS,
        "burn_in": BURN_IN,
        "designs": list(results_df["design"].unique()),
        "summary": summary_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "exp8_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print("[OUTPUT] Metadata saved to results/exp8_metadata.json")

    return meta


def _print_summary(metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")


def _plot_simulation_results(summary_df: pd.DataFrame) -> None:
    """Bar chart comparing F1 scores across designs and estimators."""
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(summary_df))
    bars = ax.bar(x, summary_df["mean_f1"].fillna(0), color=[
        "steelblue" if e == "unfiltered" else "coral"
        for e in summary_df["estimator"]
    ], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['design']}\n({r['estimator']})" for _, r in summary_df.iterrows()],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean F1 Score (edge recovery)", fontsize=11)
    ax.set_title("Simulation Study: Edge Recovery Performance\n"
                 "(GARCH filtering effect on heteroskedastic designs)", fontsize=12)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, summary_df["mean_f1"].fillna(0)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Unfiltered"),
        Patch(facecolor="coral", label="GARCH-Filtered"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp8_simulation_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Simulation F1 bar chart saved to results/exp8_simulation_f1.png")


def _plot_boxplots(results_df: pd.DataFrame) -> None:
    """Box plots of per-replication F1 across designs."""
    fig, ax = plt.subplots(figsize=(12, 5))
    data_to_plot = []
    labels_to_plot = []

    for design_name in sorted(results_df["design"].unique()):
        f1_per_rep = results_df[results_df["design"] == design_name]["f1"].tolist()
        if f1_per_rep:
            data_to_plot.append(f1_per_rep)
            estimator = "filt" if design_name.endswith("-filtered") else "unfilt"
            labels_to_plot.append(f"{design_name[:10]}\n({estimator})")

    if not data_to_plot:
        plt.close(fig)
        return

    ax.boxplot(data_to_plot, tick_labels=labels_to_plot, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"))
    ax.set_ylabel("F1 Score per Replication", fontsize=11)
    ax.set_title("Simulation Study: Distribution of Edge Recovery F1\n"
                 "(50 replications per design)", fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp8_simulation_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Simulation boxplots saved to results/exp8_simulation_boxplots.png")


if __name__ == "__main__":
    results = run_exp_8()
    print("\n=== EXP 8 COMPLETE ===")
