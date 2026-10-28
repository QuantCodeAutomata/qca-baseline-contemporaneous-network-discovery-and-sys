"""
Experiment 5: Benchmark comparison with Diebold-Yilmaz connectedness,
Marginal Expected Shortfall (MES), and size/common-factor controls.
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
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BANK_TICKERS, MARKET_TICKER, MARKET_CAPS, build_baseline_dataset
from src.garch_filter import fit_garch_filter
from src.lingam_utils import (
    estimate_direct_lingam, fit_predecessor_regressions,
    compute_bootstrap_se, wald_bh_selection,
)
from src.systemic_scores import compute_total_effect_matrix, compute_systemic_scores
from src.benchmarks import (
    compute_dy_connectedness,
    compute_mes,
    compute_pc1_loadings,
    spearman_with_pvalue,
    partial_rank_correlation_one_control,
    partial_rank_correlation_two_controls,
    benchmark_comparison_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"

GLOBAL_SEED = 1121
BOOT_SEED   = 731
N_BOOT      = int(os.environ.get("N_BOOT_EXP5", "1000"))
MATERIALITY = 0.05


def run_exp_5(dataset: dict | None = None) -> dict:
    """Run the benchmark comparison experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP 5: Benchmark Comparison (DY, MES, Size/PC1 Controls) ===")

    if dataset is None:
        dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)

    baseline_ret = dataset["baseline_returns"]
    tickers = list(baseline_ret.columns)
    avail_banks = [t for t in BANK_TICKERS if t in tickers]
    n_banks = len(avail_banks)

    # ── Step 1: Load or compute SIS/SVS from baseline ─────────────────────────
    print("\n[EXP5] Loading systemic scores...")
    scores_path = RESULTS_DIR / "exp1_systemic_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path, index_col=0)
        print(f"  Loaded scores for {len(scores)} banks from exp1 output.")
    else:
        print("  Recomputing systemic scores (exp1 output not found)...")
        # Need to recompute
        z_path = DATA_DIR / "Z_baseline.parquet"
        if z_path.exists():
            Z = pd.read_parquet(z_path)
        else:
            garch_res = fit_garch_filter(baseline_ret, seed=GLOBAL_SEED)
            Z = garch_res["Z"]

        b0_path = DATA_DIR / "exp1_B0_selected.npy"
        if b0_path.exists():
            B0_sel = np.load(b0_path)
            zt = list(Z.columns)
        else:
            lingam_res = estimate_direct_lingam(Z.values, seed=GLOBAL_SEED)
            causal_order = lingam_res["order"]
            B0 = fit_predecessor_regressions(Z.values, causal_order)
            SE = compute_bootstrap_se(Z.values, causal_order,
                                       n_boot=N_BOOT, block_length=10, seed=BOOT_SEED)
            sel = wald_bh_selection(B0, SE, causal_order, q_bh=0.05, materiality_threshold=MATERIALITY)
            B0_sel = sel["B0_selected_bh"]
            zt = list(Z.columns)

        zt = list(Z.columns)
        bank_idx = [zt.index(t) for t in avail_banks if t in zt]
        B0_bank  = B0_sel[np.ix_(bank_idx, bank_idx)]
        C_B = compute_total_effect_matrix(B0_bank, avail_banks)
        scores = compute_systemic_scores(C_B, MARKET_CAPS, avail_banks)

    # Align to available banks
    score_banks = [t for t in avail_banks if t in scores.index]
    scores = scores.loc[score_banks]

    SIS  = scores["SIS"].values
    SVS  = scores["SVS"].values
    nSVS = scores["nSVS"].values
    market_caps = np.array([MARKET_CAPS.get(t, 1.0) for t in score_banks])

    print(f"  Systemic scores for {len(score_banks)} banks")

    # ── Step 3: DY Connectedness ──────────────────────────────────────────────
    print("\n[EXP5] Computing Diebold-Yilmaz connectedness...")
    # Standardize bank returns (mean 0, std 1) on screened sample
    bank_ret = baseline_ret[[t for t in avail_banks if t in baseline_ret.columns]].copy()
    for col in bank_ret.columns:
        mu, sd = bank_ret[col].mean(), bank_ret[col].std(ddof=1)
        bank_ret[col] = (bank_ret[col] - mu) / (sd if sd > 0 else 1.0)

    dy_result = compute_dy_connectedness(bank_ret, lag=1, horizon=10)
    dy_to      = dy_result["to_others"]     # DY 'to others' (transmitter)
    dy_from    = dy_result["from_others"]   # DY 'from others' (receiver)

    # Align to score_banks
    dy_to_vals   = np.array([dy_to.get(t, np.nan)   for t in score_banks])
    dy_from_vals = np.array([dy_from.get(t, np.nan) for t in score_banks])

    # ── Step 6: MES ───────────────────────────────────────────────────────────
    print("\n[EXP5] Computing Marginal Expected Shortfall (MES)...")
    market_raw = baseline_ret[MARKET_TICKER] if MARKET_TICKER in baseline_ret.columns else None
    bank_raw_aligned = baseline_ret[[t for t in score_banks if t in baseline_ret.columns]].copy()
    if market_raw is None:
        print("  WARNING: market proxy not available; MES cannot be computed")
        mes_vals = np.full(len(score_banks), np.nan)
    else:
        mes = compute_mes(bank_raw_aligned, market_raw, quantile=0.05)
        mes_vals = np.array([mes.get(t, np.nan) for t in score_banks])

    # ── Step 7: PC1 loadings ──────────────────────────────────────────────────
    print("\n[EXP5] Computing PC1 loadings...")
    pc1_loadings = compute_pc1_loadings(bank_ret)
    pc1_vals = np.array([pc1_loadings.get(t, np.nan) for t in score_banks])

    # ── Step 8-10: Rank correlations ──────────────────────────────────────────
    print("\n[EXP5] Computing Spearman rank correlations...")
    log_caps = np.log(market_caps + 1e-6)

    # Masks for valid values
    valid_dy  = ~np.isnan(dy_to_vals)   & ~np.isnan(dy_from_vals)
    valid_mes = ~np.isnan(mes_vals)
    valid_pc1 = ~np.isnan(pc1_vals)
    valid_all = valid_dy & valid_mes & valid_pc1

    def safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        x_m, y_m = x[mask], y[mask]
        if len(x_m) < 5:
            return np.nan, np.nan
        r, p = stats.spearmanr(x_m, y_m)
        return float(r), float(p)

    def safe_partial_1(x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> float:
        x_m, y_m, z_m = x[mask], y[mask], z[mask]
        if len(x_m) < 5:
            return np.nan
        return partial_rank_correlation_one_control(
            pd.Series(x_m), pd.Series(y_m), pd.Series(z_m)
        )

    def safe_partial_2(x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, mask: np.ndarray) -> float:
        x_m, y_m, z1_m, z2_m = x[mask], y[mask], z1[mask], z2[mask]
        if len(x_m) < 5:
            return np.nan
        return partial_rank_correlation_two_controls(
            pd.Series(x_m), pd.Series(y_m), pd.Series(z1_m), pd.Series(z2_m)
        )

    # SIS vs DY 'to'
    r_sis_dy,    p_sis_dy    = safe_corr(SIS, dy_to_vals, valid_dy)
    r_svs_dyfr,  p_svs_dyfr  = safe_corr(SVS, dy_from_vals, valid_dy)
    r_sis_mes,   p_sis_mes   = safe_corr(SIS, mes_vals, valid_mes)
    r_svs_mes,   p_svs_mes   = safe_corr(SVS, mes_vals, valid_mes)

    print(f"\n  Raw Spearman correlations:")
    print(f"    SIS vs DY 'to':    {r_sis_dy:.4f}  (target: ~0.416)")
    print(f"    SVS vs DY 'from':  {r_svs_dyfr:.4f}  (target: ~0.721)")
    print(f"    SIS vs MES:        {r_sis_mes:.4f}  (target: ~0.149)")
    print(f"    SVS vs MES:        {r_svs_mes:.4f}  (target: ~0.449)")

    # Size-controlled
    r_sis_dy_sz   = safe_partial_1(SIS, dy_to_vals, log_caps, valid_dy)
    r_svs_dyfr_sz = safe_partial_1(SVS, dy_from_vals, log_caps, valid_dy)
    r_sis_mes_sz  = safe_partial_1(SIS, mes_vals, log_caps, valid_mes)
    r_svs_mes_sz  = safe_partial_1(SVS, mes_vals, log_caps, valid_mes)

    print(f"\n  Size-controlled partial rank correlations:")
    print(f"    SIS vs DY 'to':    {r_sis_dy_sz:.4f}  (target: ~0.302)")
    print(f"    SVS vs DY 'from':  {r_svs_dyfr_sz:.4f}  (target: ~0.639)")
    print(f"    SIS vs MES:        {r_sis_mes_sz:.4f}  (target: ~0.040)")
    print(f"    SVS vs MES:        {r_svs_mes_sz:.4f}  (target: ~0.335)")

    # Size + PC1 controlled
    r_sis_dy_sz_pc1   = safe_partial_2(SIS, dy_to_vals, log_caps, np.abs(pc1_vals), valid_dy & valid_pc1)
    r_svs_dyfr_sz_pc1 = safe_partial_2(SVS, dy_from_vals, log_caps, np.abs(pc1_vals), valid_dy & valid_pc1)
    r_sis_mes_sz_pc1  = safe_partial_2(SIS, mes_vals, log_caps, np.abs(pc1_vals), valid_mes & valid_pc1)
    r_svs_mes_sz_pc1  = safe_partial_2(SVS, mes_vals, log_caps, np.abs(pc1_vals), valid_mes & valid_pc1)

    print(f"\n  Size+PC1-controlled partial rank correlations:")
    print(f"    SIS vs DY 'to':    {r_sis_dy_sz_pc1:.4f}  (target: ~-0.006)")
    print(f"    SVS vs DY 'from':  {r_svs_dyfr_sz_pc1:.4f}  (target: ~-0.043)")
    print(f"    SIS vs MES:        {r_sis_mes_sz_pc1:.4f}  (target: ~-0.033)")
    print(f"    SVS vs MES:        {r_svs_mes_sz_pc1:.4f}  (target: ~0.252)")

    # ── Build summary table ────────────────────────────────────────────────────
    corr_table = pd.DataFrame({
        "Pair": ["SIS vs DY-to", "SVS vs DY-from", "SIS vs MES", "SVS vs MES"],
        "Raw_rho": [r_sis_dy, r_svs_dyfr, r_sis_mes, r_svs_mes],
        "Raw_p": [p_sis_dy, p_svs_dyfr, p_sis_mes, p_svs_mes],
        "Size_ctrl_rho": [r_sis_dy_sz, r_svs_dyfr_sz, r_sis_mes_sz, r_svs_mes_sz],
        "Size_PC1_ctrl_rho": [r_sis_dy_sz_pc1, r_svs_dyfr_sz_pc1, r_sis_mes_sz_pc1, r_svs_mes_sz_pc1],
    })
    corr_table.to_csv(RESULTS_DIR / "exp5_benchmark_correlations.csv", index=False)
    print("\n[OUTPUT] Benchmark correlation table saved to results/exp5_benchmark_correlations.csv")
    print(corr_table.to_string(index=False, float_format="{:.4f}".format))

    # ── Bank-level benchmark table ────────────────────────────────────────────
    bm_df = pd.DataFrame({
        "ticker": score_banks,
        "SIS": SIS,
        "SVS": SVS,
        "nSVS": nSVS,
        "DY_to": dy_to_vals,
        "DY_from": dy_from_vals,
        "MES": mes_vals,
        "PC1_loading": pc1_vals,
        "market_cap_eur_bn": market_caps,
    })
    bm_df.to_csv(RESULTS_DIR / "exp5_benchmark_data.csv", index=False)
    print("[OUTPUT] Benchmark data saved to results/exp5_benchmark_data.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_scatter_benchmark(bm_df)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "exp_id": "exp_5",
        "n_banks": n_banks,
        "raw_sis_dy": r_sis_dy, "raw_svs_dyfrom": r_svs_dyfr,
        "raw_sis_mes": r_sis_mes, "raw_svs_mes": r_svs_mes,
        "size_sis_dy": r_sis_dy_sz, "size_svs_dyfrom": r_svs_dyfr_sz,
        "size_sis_mes": r_sis_mes_sz, "size_svs_mes": r_svs_mes_sz,
        "size_pc1_sis_dy": r_sis_dy_sz_pc1, "size_pc1_svs_dyfrom": r_svs_dyfr_sz_pc1,
        "size_pc1_sis_mes": r_sis_mes_sz_pc1, "size_pc1_svs_mes": r_svs_mes_sz_pc1,
        "svs_stronger_than_sis_raw_dy": bool(abs(r_svs_dyfr) > abs(r_sis_dy)) if np.isfinite(r_svs_dyfr) else None,
        "svs_stronger_than_sis_raw_mes": bool(abs(r_svs_mes) > abs(r_sis_mes)) if np.isfinite(r_svs_mes) else None,
    }
    with open(RESULTS_DIR / "exp5_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OUTPUT] Metadata saved to results/exp5_metadata.json")

    return meta


def _plot_scatter_benchmark(bm_df: pd.DataFrame) -> None:
    """Scatter plots: SIS/SVS vs DY and MES."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    pairs = [
        ("SIS", "DY_to",   "SIS vs DY 'to others' [transmitter]"),
        ("SVS", "DY_from", "SVS vs DY 'from others' [receiver]"),
        ("SIS", "MES",     "SIS vs MES [transmitter vs tail vulnerability]"),
        ("SVS", "MES",     "SVS vs MES [receiver vs tail vulnerability]"),
    ]
    for ax, (x_col, y_col, title) in zip(axes.flat, pairs):
        sub = bm_df[[x_col, y_col, "ticker"]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        r, p = stats.spearmanr(sub[x_col], sub[y_col])
        ax.scatter(sub[x_col], sub[y_col], alpha=0.7, s=50, color="steelblue", edgecolors="white")
        for _, row in sub.iterrows():
            ax.annotate(row["ticker"], (row[x_col], row[y_col]),
                        fontsize=6, alpha=0.7, ha="center")
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f"{title}\n(Spearman ρ={r:.3f}, p={p:.3f})", fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_benchmark_scatters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OUTPUT] Benchmark scatter plots saved to results/exp5_benchmark_scatters.png")


if __name__ == "__main__":
    dataset = build_baseline_dataset(cache_dir=str(DATA_DIR), seed=GLOBAL_SEED)
    results = run_exp_5(dataset)
    print("\n=== EXP 5 COMPLETE ===")
