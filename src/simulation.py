"""
Simulation study module for edge recovery and volatility-filtering effects.

Generates synthetic data from recursive acyclic linear systems and evaluates
DirectLiNGAM-based network recovery under controlled designs.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import lingam

logger = logging.getLogger(__name__)

GLOBAL_SEED = 1121


def generate_recursive_system(
    n_vars: int,
    T: int,
    burn_in: int = 500,
    sparsity: float = 0.3,
    coef_range: tuple[float, float] = (0.1, 0.5),
    shock_dist: str = "laplace",
    heteroskedastic: bool = False,
    seed: int = GLOBAL_SEED,
) -> dict:
    """Generate data from a recursive acyclic linear system.

    System: X_t = B_true @ X_t + eps_t  ⟹  X_t = (I - B_true)^{-1} eps_t
    B_true is strictly lower triangular (recursive ordering = 0, 1, ..., n-1).

    Parameters
    ----------
    n_vars        : number of variables
    T             : target sample size (after burn-in)
    burn_in       : number of initial observations to discard
    sparsity      : fraction of lower-triangular entries that are non-zero
    coef_range    : (min_abs, max_abs) for non-zero coefficients
    shock_dist    : distribution of shocks ('laplace', 't', 'normal')
    heteroskedastic: if True, multiplies shocks by GARCH(1,1)-like scale
    seed          : random seed

    Returns
    -------
    dict with: 'data' (T x n_vars), 'B0_true' (n x n), 'causal_order' (list),
               'shocks' (T x n_vars)
    """
    rng = np.random.RandomState(seed)

    # Build sparse lower-triangular coefficient matrix
    B_true = np.zeros((n_vars, n_vars))
    for i in range(1, n_vars):
        for j in range(i):
            if rng.rand() < sparsity:
                sign = rng.choice([-1, 1])
                mag  = rng.uniform(*coef_range)
                B_true[i, j] = sign * mag

    total_T = T + burn_in

    # Generate shocks
    if shock_dist == "laplace":
        eps_raw = rng.laplace(0, 1.0 / np.sqrt(2), size=(total_T, n_vars))
    elif shock_dist == "t":
        eps_raw = rng.standard_t(df=5, size=(total_T, n_vars))
        eps_raw /= np.sqrt(5.0 / 3.0)  # normalize variance to 1
    else:  # normal
        eps_raw = rng.randn(total_T, n_vars)

    # Optional GARCH-type heteroskedasticity
    if heteroskedastic:
        omega, alpha_g, beta_g = 0.01, 0.1, 0.85
        h = np.ones((total_T, n_vars))
        for t in range(1, total_T):
            h[t] = omega + alpha_g * eps_raw[t - 1] ** 2 + beta_g * h[t - 1]
        sigma = np.sqrt(np.maximum(h, 1e-8))
        eps = eps_raw * sigma
    else:
        eps = eps_raw
        sigma = np.ones_like(eps)

    # Generate observed data: X_t = (I - B)^{-1} eps_t  (standard LiNGAM convention)
    # In matrix form: X = ((I-B)^{-1} eps^T)^T = eps @ (I-B)^{-T}
    I_minus_B = np.eye(n_vars) - B_true
    try:
        # solve (I-B) @ result = eps.T  →  result = (I-B)^{-1} @ eps.T  →  X = result.T
        X = np.linalg.solve(I_minus_B, eps.T).T
    except np.linalg.LinAlgError:
        X = eps.copy()

    # Discard burn-in
    X_ret = X[burn_in:].copy()
    eps_ret = eps[burn_in:].copy()

    return {
        "data": X_ret,
        "B0_true": B_true,
        "causal_order": list(range(n_vars)),
        "shocks": eps_ret,
    }


def evaluate_edge_recovery(
    B0_estimated: np.ndarray,
    B0_true: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Compare estimated coefficient matrix to ground truth.

    An edge is detected if |B0_estimated[i,j]| >= threshold.
    An edge is true if B0_true[i,j] != 0.

    Parameters
    ----------
    B0_estimated : (n, n) estimated coefficient matrix
    B0_true      : (n, n) true coefficient matrix
    threshold    : detection threshold

    Returns
    -------
    dict with: tp, fp, fn, tn, precision, recall, f1, orientation_acc
    """
    detected = np.abs(B0_estimated) >= threshold
    true_edges = B0_true != 0

    tp = int(np.sum(detected & true_edges))
    fp = int(np.sum(detected & ~true_edges))
    fn = int(np.sum(~detected & true_edges))
    tn = int(np.sum(~detected & ~true_edges))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Orientation accuracy: among detected true edges, correct sign?
    orientation_correct = 0
    n_true_detected = 0
    n = B0_true.shape[0]
    for i in range(n):
        for j in range(n):
            if true_edges[i, j] and detected[i, j]:
                n_true_detected += 1
                if np.sign(B0_estimated[i, j]) == np.sign(B0_true[i, j]):
                    orientation_correct += 1

    orientation_acc = orientation_correct / n_true_detected if n_true_detected > 0 else np.nan

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "orientation_acc": orientation_acc,
    }


def _fit_garch_filter_simple(
    data: np.ndarray,
    seed: int = GLOBAL_SEED,
) -> np.ndarray:
    """Simple GARCH(1,1) filtering for simulation study.

    Returns filtered (standardized) data.
    """
    try:
        from arch import arch_model
    except ImportError:
        return data

    T, n = data.shape
    filtered = np.zeros_like(data)

    for j in range(n):
        series = pd.Series(data[:, j])
        series -= series.mean()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(series, mean="Zero", vol="GARCH", p=1, q=1, dist="StudentsT")
                res = am.fit(disp="off", options={"maxiter": 200})
            cond_vol = res.conditional_volatility.values
            cond_vol = np.where(cond_vol > 0, cond_vol, 1.0)
            filt_j = series.values / cond_vol
        except Exception:
            filt_j = series.values
        filt_j = (filt_j - filt_j.mean()) / (filt_j.std() + 1e-8)
        filtered[:, j] = filt_j

    return filtered


def run_simulation_study(
    n_vars: int = 10,
    T: int = 1990,
    n_replications: int = 50,
    burn_in: int = 500,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """Run simulation study with 50 replications across 4 designs.

    Designs:
    1. homoskedastic-laplace : Laplace shocks, no heteroskedasticity, unfiltered
    2. homoskedastic-t       : Student-t shocks, no heteroskedasticity, unfiltered
    3. heterosk-unfiltered   : GARCH heteroskedasticity, unfiltered estimation
    4. heterosk-filtered     : GARCH heteroskedasticity, with GARCH filtering

    Parameters
    ----------
    n_vars         : number of variables
    T              : sample size
    n_replications : number of Monte Carlo replications (default 50 per paper)
    burn_in        : burn-in period to discard
    seed           : global seed

    Returns
    -------
    DataFrame with columns: design, replication, tp, fp, fn, precision, recall, f1, orientation_acc
    """
    designs = [
        {"name": "homoskedastic-laplace", "shock_dist": "laplace", "heteroskedastic": False, "filter": False},
        {"name": "homoskedastic-t",       "shock_dist": "t",       "heteroskedastic": False, "filter": False},
        {"name": "heterosk-unfiltered",   "shock_dist": "laplace", "heteroskedastic": True,  "filter": False},
        {"name": "heterosk-filtered",     "shock_dist": "laplace", "heteroskedastic": True,  "filter": True},
    ]

    records = []
    for design in designs:
        for rep in range(n_replications):
            rep_seed = seed + rep * 1000 + hash(design["name"]) % 10000

            sys_data = generate_recursive_system(
                n_vars=n_vars, T=T, burn_in=burn_in,
                sparsity=0.3, coef_range=(0.15, 0.45),
                shock_dist=design["shock_dist"],
                heteroskedastic=design["heteroskedastic"],
                seed=rep_seed,
            )
            data   = sys_data["data"]
            B_true = sys_data["B0_true"]

            # Apply GARCH filter if required
            if design["filter"]:
                data_input = _fit_garch_filter_simple(data, seed=rep_seed + 1)
            else:
                # Mean-center and unit-variance standardize
                data_input = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

            # Estimate DirectLiNGAM
            try:
                np.random.seed(rep_seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = lingam.DirectLiNGAM(measure="pwling")
                    model.fit(data_input)
                B_est = model.adjacency_matrix_
            except Exception as exc:
                logger.warning("DirectLiNGAM failed in rep %d: %s", rep, exc)
                B_est = np.zeros_like(B_true)

            metrics = evaluate_edge_recovery(B_est, B_true, threshold=0.05)
            metrics["design"]      = design["name"]
            metrics["replication"] = rep
            records.append(metrics)

    return pd.DataFrame(records)


def summarize_simulation_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation results by design.

    Parameters
    ----------
    results : output DataFrame from run_simulation_study

    Returns
    -------
    Summary DataFrame with mean, median, std per metric per design
    """
    metric_cols = ["tp", "fp", "fn", "precision", "recall", "f1", "orientation_acc"]
    available = [c for c in metric_cols if c in results.columns]

    groups = results.groupby("design")[available]

    rows = []
    for col in available:
        agg = groups[col].agg(["mean", "median", "std"]).reset_index()
        agg.columns = ["design", f"{col}_mean", f"{col}_median", f"{col}_std"]
        rows.append(agg.set_index("design"))

    summary = pd.concat(rows, axis=1).reset_index()
    return summary
