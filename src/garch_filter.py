"""
GARCH(1,1)-Student-t volatility filtering module.

Implements univariate GARCH filtering for each return series and
extends the conditional variance recursion to out-of-sample test periods.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


def fit_garch_filter(
    returns: pd.DataFrame,
    seed: int = 1121,
) -> dict:
    """Fit univariate GARCH(1,1) with Student-t innovations to each series.

    Pipeline per series:
    1. Mean-center the return series.
    2. Fit GARCH(1,1)-StudentT using arch (mean='Zero' on pre-centered series).
    3. Extract conditional volatility sigma_t.
    4. Divide centered returns by sigma_t to obtain standardized returns.
    5. Re-center and rescale each filtered series to mean=0, std=1.

    Parameters
    ----------
    returns : DataFrame of raw return series (T x n)
    seed    : random seed for GARCH optimizer

    Returns
    -------
    dict with keys:
        'centered'     : mean-centered returns (DataFrame)
        'cond_vol'     : fitted conditional volatilities (DataFrame)
        'standardized' : r_t / sigma_t (before unit-variance rescaling)
        'garch_params' : dict {ticker: {'omega', 'alpha', 'beta', 'nu',
                                        'last_h', 'last_eps', 'mean', 'std'}}
        'Z'            : re-centered, unit-variance filtered returns (DataFrame)
        'garch_results': dict {ticker: arch ARCHModelResult}
    """
    np.random.seed(seed)
    n, k = returns.shape
    tickers = list(returns.columns)

    centered_data = {}
    cond_vol_data = {}
    standardized_data = {}
    garch_params = {}
    garch_results = {}

    for ticker in tickers:
        series = returns[ticker].copy()
        mu = series.mean()
        centered = series - mu

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                centered,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                dist="StudentsT",
            )
            try:
                res = am.fit(disp="off", options={"maxiter": 500})
            except Exception as exc:
                logger.warning("GARCH fit failed for %s: %s", ticker, exc)
                # Fallback: use rolling std as conditional volatility
                cond_vol = centered.rolling(21, min_periods=5).std().fillna(method="bfill").fillna(centered.std())
                standardized = centered / cond_vol
                garch_params[ticker] = {"omega": np.nan, "alpha": np.nan, "beta": np.nan, "nu": np.nan,
                                        "last_h": cond_vol.iloc[-1] ** 2, "last_eps": centered.iloc[-1],
                                        "mean": mu, "std": standardized.std()}
                centered_data[ticker] = centered
                cond_vol_data[ticker] = cond_vol
                standardized_data[ticker] = standardized
                garch_results[ticker] = None
                continue

        cond_vol = res.conditional_volatility
        standardized = centered / cond_vol

        # Store parameters for out-of-sample extension
        params = res.params
        omega = float(params.get("omega", params.iloc[0]))
        alpha = float(params.get("alpha[1]", params.iloc[1]))
        beta  = float(params.get("beta[1]",  params.iloc[2]))
        nu    = float(params.get("nu", 8.0))

        last_h   = float(cond_vol.iloc[-1] ** 2)
        last_eps = float(centered.iloc[-1])

        garch_params[ticker] = {
            "omega": omega, "alpha": alpha, "beta": beta, "nu": nu,
            "last_h": last_h, "last_eps": last_eps,
            "mean": mu, "std": float(standardized.std()),
        }
        garch_results[ticker] = res
        centered_data[ticker] = centered
        cond_vol_data[ticker] = cond_vol
        standardized_data[ticker] = standardized

    centered_df    = pd.DataFrame(centered_data)
    cond_vol_df    = pd.DataFrame(cond_vol_data)
    standardized_df = pd.DataFrame(standardized_data)

    # Re-center and rescale to mean=0, std=1 to form Z_t
    Z = standardized_df.copy()
    for ticker in tickers:
        col = Z[ticker]
        mu_s  = col.mean()
        std_s = col.std(ddof=1)
        if std_s > 0:
            Z[ticker] = (col - mu_s) / std_s
        # Store rescaling stats back
        garch_params[ticker]["z_mean"] = float(mu_s)
        garch_params[ticker]["z_std"]  = float(std_s)

    return {
        "centered": centered_df,
        "cond_vol": cond_vol_df,
        "standardized": standardized_df,
        "garch_params": garch_params,
        "Z": Z,
        "garch_results": garch_results,
    }


def extend_garch_to_test(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    garch_params: dict,
) -> dict:
    """Continue GARCH variance recursion into test set with fixed training parameters.

    Uses manually implemented GARCH(1,1) recursion:
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    Parameters
    ----------
    train_returns : training raw returns (used only for training mean)
    test_returns  : test raw returns (same columns as train)
    garch_params  : dict from fit_garch_filter (per-ticker parameters)

    Returns
    -------
    dict with keys:
        'cond_vol_test'     : conditional volatilities for test period (DataFrame)
        'standardized_test' : test standardized returns before unit-variance scaling
        'Z_test'            : re-centered, unit-variance filtered test returns (DataFrame)
    """
    tickers = list(test_returns.columns)
    T_test = len(test_returns)

    cond_vol_test_data = {}
    standardized_test_data = {}

    for ticker in tickers:
        params = garch_params[ticker]
        omega  = params["omega"]
        alpha  = params["alpha"]
        beta   = params["beta"]
        mu     = params["mean"]   # training mean
        last_h = params["last_h"]
        last_eps = params["last_eps"]

        # Fallback for failed fits
        if np.isnan(omega):
            raw = test_returns[ticker]
            centered_test = raw - mu
            sigma_t = pd.Series(np.full(T_test, np.sqrt(last_h)), index=test_returns.index)
            cond_vol_test_data[ticker] = sigma_t
            standardized_test_data[ticker] = centered_test / sigma_t
            continue

        raw = test_returns[ticker]
        centered_test = (raw - mu).values

        h = np.empty(T_test)
        eps_sq = last_eps ** 2
        h_prev = last_h

        for t in range(T_test):
            h[t] = omega + alpha * eps_sq + beta * h_prev
            eps_sq = centered_test[t] ** 2
            h_prev = h[t]

        sigma_t = pd.Series(np.sqrt(np.maximum(h, 1e-12)), index=test_returns.index)
        cond_vol_test_data[ticker] = sigma_t
        standardized_test_data[ticker] = pd.Series(
            (raw - mu).values / sigma_t.values, index=test_returns.index
        )

    cond_vol_test_df    = pd.DataFrame(cond_vol_test_data)
    standardized_test_df = pd.DataFrame(standardized_test_data)

    # Apply training z-score rescaling (use training mean and std from params)
    Z_test = standardized_test_df.copy()
    for ticker in tickers:
        z_mean = garch_params[ticker].get("z_mean", 0.0)
        z_std  = garch_params[ticker].get("z_std",  1.0)
        if z_std > 0:
            Z_test[ticker] = (Z_test[ticker] - z_mean) / z_std

    return {
        "cond_vol_test": cond_vol_test_df,
        "standardized_test": standardized_test_df,
        "Z_test": Z_test,
    }
