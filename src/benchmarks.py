"""
Benchmark comparison module: Diebold-Yilmaz connectedness and MES.

Implements DY generalized FEVD connectedness and Marginal Expected Shortfall
for comparison with SIS/SVS systemic risk measures.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

logger = logging.getLogger(__name__)


# ── Diebold-Yilmaz connectedness ───────────────────────────────────────────────

def _compute_ma_coefficients(
    coef_matrices: list[np.ndarray],
    H: int,
    n: int,
) -> list[np.ndarray]:
    """Compute MA(H-1) coefficient matrices Phi_0, ..., Phi_{H-1} from VAR coefficients.

    Uses the recursive formula: Phi_h = sum_{k=1}^{min(h,p)} A_k Phi_{h-k}
    with Phi_0 = I, Phi_h = 0 for h < 0.

    Parameters
    ----------
    coef_matrices : list of VAR(p) coefficient matrices [A_1, ..., A_p]
    H             : forecast horizon
    n             : number of variables

    Returns
    -------
    list of (n x n) MA coefficient matrices [Phi_0, ..., Phi_{H-1}]
    """
    p = len(coef_matrices)
    Phi = [np.eye(n)] + [np.zeros((n, n)) for _ in range(H - 1)]

    for h in range(1, H):
        for k in range(1, min(h, p) + 1):
            Phi[h] += coef_matrices[k - 1] @ Phi[h - k]

    return Phi


def compute_dy_connectedness(
    returns: pd.DataFrame,
    lag: int = 1,
    horizon: int = 10,
) -> dict:
    """Compute Diebold-Yilmaz generalized FEVD connectedness.

    Uses generalized (non-orthogonalized) FEVD:
        theta_ij(H) = sigma_jj^{-1} * sum_{h=0}^{H-1} (e_i' Phi_h Sigma e_j)^2
                      / sum_{h=0}^{H-1} e_i' Phi_h Sigma Phi_h' e_i

    Row-normalizes: tilde_theta_ij = theta_ij / sum_j theta_ij

    Parameters
    ----------
    returns : DataFrame of bank returns (T x n_banks), market proxy excluded
    lag     : VAR lag order (default 1)
    horizon : forecast horizon H (default 10)

    Returns
    -------
    dict with: 'fevd_matrix', 'to_others', 'from_others', 'net',
               'total_connectedness', 'tickers'
    """
    # Standardize returns
    scaler = StandardScaler()
    Z = pd.DataFrame(
        scaler.fit_transform(returns),
        index=returns.index,
        columns=returns.columns,
    )
    n = Z.shape[1]
    tickers = list(Z.columns)

    # Fit VAR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_model = VAR(Z)
        try:
            var_result = var_model.fit(lag, trend="c")
        except Exception as exc:
            logger.warning("VAR fit failed: %s. Using lag=1.", exc)
            var_result = var_model.fit(1, trend="c")

    # Extract VAR coefficient matrices A_1, ..., A_p
    p_fitted = var_result.k_ar
    coef_matrices = []
    for k in range(1, p_fitted + 1):
        coef_matrices.append(var_result.coefs[k - 1])  # shape (n, n)

    # MA coefficients
    Phi = _compute_ma_coefficients(coef_matrices, horizon, n)

    # Residual covariance — convert to plain numpy array for indexing
    Sigma = np.asarray(var_result.sigma_u)  # (n, n)

    # Generalized FEVD
    theta = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        denom_sum = 0.0
        for h in range(horizon):
            Phi_h = Phi[h]
            denom_sum += float(e_i @ Phi_h @ Sigma @ Phi_h.T @ e_i)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            sigma_jj = float(Sigma[j, j])
            if sigma_jj <= 0:
                sigma_jj = 1.0
            numer_sum = 0.0
            for h in range(horizon):
                Phi_h = Phi[h]
                numer_sum += float(e_i @ Phi_h @ Sigma @ e_j) ** 2
            theta[i, j] = (1.0 / sigma_jj) * numer_sum / denom_sum if denom_sum > 0 else 0.0

    # Row-normalize
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    tilde_theta = theta / row_sums  # (n, n)

    # DY connectedness measures
    # 'to others' = column sum of off-diagonal (how much j contributes to others)
    to_others = np.array([
        sum(tilde_theta[i, j] for i in range(n) if i != j) for j in range(n)
    ])
    # 'from others' = row sum of off-diagonal (how much i receives from others)
    from_others = np.array([
        sum(tilde_theta[i, j] for j in range(n) if j != i) for i in range(n)
    ])
    net = to_others - from_others
    total_conn = float(np.sum(to_others) / n)

    return {
        "fevd_matrix": pd.DataFrame(tilde_theta, index=tickers, columns=tickers),
        "to_others": pd.Series(to_others, index=tickers),
        "from_others": pd.Series(from_others, index=tickers),
        "net": pd.Series(net, index=tickers),
        "total_connectedness": total_conn,
        "tickers": tickers,
    }


# ── Marginal Expected Shortfall ─────────────────────────────────────────────────

def compute_mes(
    bank_returns: pd.DataFrame,
    market_returns: pd.Series,
    quantile: float = 0.05,
) -> pd.Series:
    """Compute Marginal Expected Shortfall (MES) for each bank.

    Identifies worst `quantile` fraction of raw market return days,
    then computes the average bank loss on those stress dates.

    MES_i = -mean(bank_return_i | market in bottom quantile%)

    Parameters
    ----------
    bank_returns    : (T, n_banks) raw log return DataFrame
    market_returns  : (T,) raw market (EXSA.DE) log return Series
    quantile        : stress threshold (default 0.05 = worst 5%)

    Returns
    -------
    Series of MES values indexed by bank ticker (positive = expected loss)
    """
    threshold = float(np.quantile(market_returns.values, quantile))
    stress_mask = market_returns <= threshold
    stress_dates = market_returns[stress_mask].index

    # Align
    common_dates = bank_returns.index.intersection(stress_dates)

    mes = {}
    for ticker in bank_returns.columns:
        returns_on_stress = bank_returns.loc[common_dates, ticker]
        mes[ticker] = float(-returns_on_stress.mean())

    return pd.Series(mes)


# ── PCA loadings ────────────────────────────────────────────────────────────────

def compute_pc1_loadings(
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute first principal component loadings from standardized bank returns.

    Parameters
    ----------
    returns : (T, n_banks) return DataFrame

    Returns
    -------
    Series of PC1 loadings indexed by bank ticker
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(returns)

    pca = PCA(n_components=1)
    pca.fit(Z)
    loadings = pca.components_[0]  # (n_banks,)

    # Ensure positive orientation (largest absolute loading is positive)
    if loadings[np.argmax(np.abs(loadings))] < 0:
        loadings = -loadings

    return pd.Series(loadings, index=returns.columns)


# ── Rank correlation utilities ──────────────────────────────────────────────────

def spearman_with_pvalue(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    """Compute Spearman rank correlation and two-sided p-value.

    Parameters
    ----------
    x, y : Series to correlate (must share index)

    Returns
    -------
    (rho, p_value) tuple
    """
    common = x.index.intersection(y.index)
    rho, pval = stats.spearmanr(x.loc[common], y.loc[common])
    return float(rho), float(pval)


def partial_rank_correlation_one_control(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
) -> float:
    """Partial Spearman rank correlation between x and y controlling for z.

    Formula: rho_XY.Z = (rho_XY - rho_XZ * rho_YZ) / sqrt((1-rho_XZ^2)(1-rho_YZ^2))

    Parameters
    ----------
    x, y, z : Series (same index assumed)

    Returns
    -------
    Partial Spearman rho (float)
    """
    common = x.index.intersection(y.index).intersection(z.index)
    xc, yc, zc = x.loc[common], y.loc[common], z.loc[common]

    rho_xy = float(stats.spearmanr(xc, yc)[0])
    rho_xz = float(stats.spearmanr(xc, zc)[0])
    rho_yz = float(stats.spearmanr(yc, zc)[0])

    denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    if denom < 1e-10:
        return np.nan
    return float((rho_xy - rho_xz * rho_yz) / denom)


def partial_rank_correlation_two_controls(
    x: pd.Series,
    y: pd.Series,
    z1: pd.Series,
    z2: pd.Series,
) -> float:
    """Partial rank correlation between x and y, controlling for z1 and z2.

    Method: rank-transform all variables, regress x_rank and y_rank on
    [intercept, z1_rank, z2_rank] via OLS, correlate residuals.

    Parameters
    ----------
    x, y, z1, z2 : Series (same index assumed)

    Returns
    -------
    Pearson correlation of OLS residuals (float)
    """
    common = (x.index.intersection(y.index)
               .intersection(z1.index)
               .intersection(z2.index))
    xc  = stats.rankdata(x.loc[common])
    yc  = stats.rankdata(y.loc[common])
    z1c = stats.rankdata(z1.loc[common])
    z2c = stats.rankdata(z2.loc[common])

    n = len(xc)
    controls = np.column_stack([np.ones(n), z1c, z2c])

    # Residuals from OLS
    beta_x, _, _, _ = np.linalg.lstsq(controls, xc, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(controls, yc, rcond=None)
    rx = xc - controls @ beta_x
    ry = yc - controls @ beta_y

    r = float(np.corrcoef(rx, ry)[0, 1])
    return r


# ── Summary table ───────────────────────────────────────────────────────────────

def benchmark_comparison_table(
    scores: pd.DataFrame,
    dy_scores: dict,
    mes: pd.Series,
    market_caps: dict,
    pc1_loadings: pd.Series,
) -> pd.DataFrame:
    """Assemble benchmark comparison table with raw and controlled rank correlations.

    Parameters
    ----------
    scores       : systemic score DataFrame (SIS, SVS, nSVS columns) indexed by ticker
    dy_scores    : dict from compute_dy_connectedness
    mes          : MES Series indexed by ticker
    market_caps  : {ticker: market_cap_EUR_bn}
    pc1_loadings : PC1 loading Series indexed by ticker

    Returns
    -------
    DataFrame with raw and controlled rank correlations
    """
    common_tickers = scores.index.intersection(dy_scores["to_others"].index).intersection(mes.index)

    SIS = scores.loc[common_tickers, "SIS"]
    SVS = scores.loc[common_tickers, "SVS"]
    dy_to   = dy_scores["to_others"].loc[common_tickers]
    dy_from = dy_scores["from_others"].loc[common_tickers]
    mes_c   = mes.loc[common_tickers]
    mc_s    = pd.Series({t: market_caps.get(t, np.nan) for t in common_tickers})
    pc1_s   = pc1_loadings.loc[common_tickers] if all(t in pc1_loadings.index for t in common_tickers) else pd.Series(np.nan, index=common_tickers)

    rows = []
    pairs = [
        ("SIS", "DY_to",   SIS, dy_to),
        ("SVS", "DY_from", SVS, dy_from),
        ("SIS", "MES",     SIS, mes_c),
        ("SVS", "MES",     SVS, mes_c),
    ]
    for (score_name, bench_name, score_s, bench_s) in pairs:
        rho_raw, p_raw = spearman_with_pvalue(score_s, bench_s)
        rho_sz   = partial_rank_correlation_one_control(score_s, bench_s, mc_s)
        rho_szpc = partial_rank_correlation_two_controls(score_s, bench_s, mc_s, pc1_s)
        rows.append({
            "score": score_name,
            "benchmark": bench_name,
            "rho_raw": rho_raw,
            "p_raw": p_raw,
            "rho_size_controlled": rho_sz,
            "rho_size_pc1_controlled": rho_szpc,
        })

    return pd.DataFrame(rows)
