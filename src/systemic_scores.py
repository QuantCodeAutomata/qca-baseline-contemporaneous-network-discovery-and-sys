"""
Systemic Impact Score (SIS) and Systemic Vulnerability Score (SVS) computation.

Implements the paper's formulas for deriving bank-level systemic risk scores
from the selected total-effect matrix C^B = (I - B0_BB^s)^{-1}.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_total_effect_matrix(
    B0_bank_selected: np.ndarray,
    bank_tickers: list[str],
) -> pd.DataFrame:
    """Compute C^B = (I - B0_BB^s)^{-1}.

    Verifies invertibility via condition number before inversion.

    Parameters
    ----------
    B0_bank_selected : (n_banks x n_banks) selected coefficient block (bank-only)
    bank_tickers     : list of bank ticker strings (length n_banks)

    Returns
    -------
    DataFrame C^B indexed and columned by bank_tickers.
    Raises ValueError if matrix is near-singular.
    """
    n = B0_bank_selected.shape[0]
    assert n == len(bank_tickers), f"Shape mismatch: {n} vs {len(bank_tickers)}"

    I_minus_B = np.eye(n) - B0_bank_selected
    cond_num = np.linalg.cond(I_minus_B)

    if cond_num > 1e12:
        warnings.warn(
            f"(I - B0_BB^s) has condition number {cond_num:.2e}; matrix may be near-singular.",
            RuntimeWarning,
        )
        logger.warning("Near-singular I - B0_BB^s, condition number: %.2e", cond_num)

    try:
        C_B = np.linalg.inv(I_minus_B)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"I - B0_BB^s is singular: {exc}") from exc

    return pd.DataFrame(C_B, index=bank_tickers, columns=bank_tickers)


def compute_systemic_scores(
    C_B: pd.DataFrame,
    market_caps: dict[str, float],
    bank_tickers: list[str],
) -> pd.DataFrame:
    """Compute SIS, SIS+, SIS-, dampening, SVS, nSVS, and ranks.

    Paper formulas (using fixed 1 Jul 2025 market capitalizations):

    For each SOURCE bank j:
        SIS_j       = sum_{i != j} |C^B[i, j]| * m_i
        SIS_j+      = sum_{i != j, C^B[i,j] > 0}  C^B[i,j] * m_i
        SIS_j-      = sum_{i != j, C^B[i,j] < 0} |C^B[i,j]| * m_i
        dampening_j = SIS_j- / SIS_j

    For each RECEIVER bank i:
        SVS_i       = sum_{j != i} |C^B[i, j]| * m_j
        nSVS_i      = SVS_i / m_i

    Parameters
    ----------
    C_B         : (n_banks x n_banks) total-effect DataFrame (index=receiver, columns=source)
    market_caps : {ticker: market_cap_EUR_bn}
    bank_tickers: ordered list of bank tickers

    Returns
    -------
    DataFrame indexed by bank tickers with columns:
        SIS, SIS_plus, SIS_minus, dampening, SVS, nSVS,
        SIS_rank, SVS_rank, nSVS_rank
    """
    n = len(bank_tickers)
    m = np.array([market_caps[t] for t in bank_tickers])

    # C_B[i, j] = total effect of shock at j on outcome at i
    C_arr = C_B.loc[bank_tickers, bank_tickers].values  # (n x n)

    SIS       = np.zeros(n)
    SIS_plus  = np.zeros(n)
    SIS_minus = np.zeros(n)
    SVS       = np.zeros(n)

    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            val = C_arr[i, j]
            # SIS: source = j, weighted by receiver market cap m_i
            SIS[j]  += abs(val) * m[i]
            if val > 0:
                SIS_plus[j]  += val * m[i]
            elif val < 0:
                SIS_minus[j] += abs(val) * m[i]
            # SVS: receiver = i, weighted by source market cap m_j
            SVS[i] += abs(val) * m[j]

    dampening = np.where(SIS > 0, SIS_minus / SIS, 0.0)
    nSVS = np.where(m > 0, SVS / m, 0.0)

    # Ranks: higher value = rank 1 (use average for ties)
    def rank_desc(arr: np.ndarray) -> np.ndarray:
        """Rank descending with average ties."""
        return stats.rankdata(-arr, method="average")

    scores = pd.DataFrame(
        {
            "SIS": SIS,
            "SIS_plus": SIS_plus,
            "SIS_minus": SIS_minus,
            "dampening": dampening,
            "SVS": SVS,
            "nSVS": nSVS,
            "SIS_rank": rank_desc(SIS),
            "SVS_rank": rank_desc(SVS),
            "nSVS_rank": rank_desc(nSVS),
        },
        index=bank_tickers,
    )

    return scores


def compute_rank_correlation(scores: pd.DataFrame) -> float:
    """Compute Spearman rank correlation between SIS and SVS.

    Parameters
    ----------
    scores : DataFrame with 'SIS' and 'SVS' columns

    Returns
    -------
    Spearman rho (float)
    """
    rho, _ = stats.spearmanr(scores["SIS"], scores["SVS"])
    return float(rho)


def print_top_banks(
    scores: pd.DataFrame,
    column: str = "SIS",
    n_top: int = 5,
) -> None:
    """Print top n_top banks by the specified score column."""
    top = scores.nlargest(n_top, column)[[column, f"{column}_rank" if f"{column}_rank" in scores else column]]
    print(f"\n[SCORES] Top {n_top} banks by {column}:")
    for ticker, row in top.iterrows():
        print(f"  {ticker}: {row[column]:.3f}")
