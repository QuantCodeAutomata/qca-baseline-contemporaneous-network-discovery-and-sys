"""
Data loading module for European bank systemic risk analysis.

Downloads and processes adjusted close prices for 51 European banks + EXSA.DE market proxy.
Implements calendar screening as described in the paper's methodology.
"""

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
from dateutil.easter import easter

logger = logging.getLogger(__name__)

# ── Universe constants ──────────────────────────────────────────────────────────

BANK_TICKERS: list[str] = [
    "A5G.IR", "ABN.AS", "ACA.PA", "ALPHA.AT", "BAMI.MI", "BBVA.MC",
    "BCP.LS", "BGN.MI", "BIRG.IR", "BKT.MC", "BMED.MI", "BMPS.MI",
    "BNP.PA", "BPE.MI", "BPSO.MI", "BST.MI", "BWB.DE", "CABK.MC",
    "CBK.DE", "CE.MI", "DANSKE.CO", "DBK.DE", "EBS.VI", "ETE.AT",
    "EUROB.AT", "FBK.MI", "GLE.PA", "IF.MI", "ING.WA", "INGA.AS",
    "ISP.MI", "JYSK.CO", "KBC.BR", "KOMB.PR", "MB.MI", "MBK.WA",
    "NDA-FI.HE", "OTP.BD", "PBB.DE", "PEO.WA", "PKO.WA", "PRO.MI",
    "RBI.VI", "SAB.MC", "SAN.MC", "SEB-A.ST", "SHB-A.ST", "SPL.WA",
    "SYDB.CO", "TPEIR.AT", "UCG.MI",
]

MARKET_TICKER: str = "EXSA.DE"
ALL_TICKERS: list[str] = BANK_TICKERS + [MARKET_TICKER]

# Fixed market capitalizations as of 1 Jul 2025 (EUR bn)
MARKET_CAPS: dict[str, float] = {
    "A5G.IR": 14.509, "ABN.AS": 19.377, "ACA.PA": 48.248, "ALPHA.AT": 6.908,
    "BAMI.MI": 14.774, "BBVA.MC": 73.973, "BCP.LS": 9.814, "BGN.MI": 5.294,
    "BIRG.IR": 11.478, "BKT.MC": 9.930, "BMED.MI": 10.554, "BMPS.MI": 8.908,
    "BNP.PA": 84.983, "BPE.MI": 10.586, "BPSO.MI": 5.143, "BST.MI": 0.148,
    "BWB.DE": 0.212, "CABK.MC": 51.917, "CBK.DE": 30.104, "CE.MI": 4.197,
    "DANSKE.CO": 28.245, "DBK.DE": 47.083, "EBS.VI": 27.809, "ETE.AT": 10.025,
    "EUROB.AT": 10.839, "FBK.MI": 11.357, "GLE.PA": 37.231, "IF.MI": 1.183,
    "ING.WA": 9.486, "INGA.AS": 56.062, "ISP.MI": 85.512, "JYSK.CO": 5.168,
    "KBC.BR": 34.398, "KOMB.PR": 7.788, "MB.MI": 15.543, "MBK.WA": 7.932,
    "NDA-FI.HE": 41.191, "OTP.BD": 17.414, "PBB.DE": 0.707, "PEO.WA": 11.269,
    "PKO.WA": 20.000, "PRO.MI": 0.110, "RBI.VI": 8.315, "SAB.MC": 13.973,
    "SAN.MC": 103.543, "SEB-A.ST": 29.896, "SHB-A.ST": 22.450, "SPL.WA": 11.707,
    "SYDB.CO": 3.129, "TPEIR.AT": 4.447, "UCG.MI": 86.580,
}

# ── Price download ──────────────────────────────────────────────────────────────

def download_prices(
    tickers: list[str],
    start: str = "2015-11-20",
    end: str = "2025-06-27",
    cache_dir: str = "/workspace/project/data",
) -> pd.DataFrame:
    """Download adjusted close prices using yfinance, cached to parquet.

    Parameters
    ----------
    tickers : list of ticker strings
    start   : ISO start date (inclusive)
    end     : ISO end date (inclusive)
    cache_dir: directory to cache the parquet file

    Returns
    -------
    DataFrame indexed by date, columns are tickers (inner join).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / "aligned_prices.parquet"

    if cache_path.exists():
        logger.info("Loading prices from cache: %s", cache_path)
        prices = pd.read_parquet(cache_path)
        return prices

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Extract Close prices (yfinance returns MultiIndex when multiple tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # Drop tickers with all-NaN (unavailable)
    prices = prices.dropna(axis=1, how="all")
    # Inner join: keep only dates where ALL remaining tickers have data
    prices = prices.dropna(axis=0, how="any")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    prices.to_parquet(cache_path)
    logger.info("Saved %d x %d price panel to %s", *prices.shape, cache_path)
    return prices


# ── Return computation ──────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute adjacent log returns log(P_t / P_{t-1}).

    Parameters
    ----------
    prices : aligned price DataFrame indexed by date

    Returns
    -------
    DataFrame of log returns, same columns, dates are return dates.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ── Calendar screen helpers ─────────────────────────────────────────────────────

def _get_easter_dates(year: int) -> tuple[date, date]:
    """Return (Good Friday, Easter Monday) for given year."""
    easter_sunday = easter(year)
    good_friday = easter_sunday - timedelta(days=2)
    easter_monday = easter_sunday + timedelta(days=1)
    return good_friday, easter_monday


def is_eligible_weekday(dt: pd.Timestamp) -> bool:
    """Return True if dt is a Mon-Fri weekday that is NOT a common holiday.

    Common holidays (excluding weekends):
    - 1 January
    - Good Friday
    - Easter Monday
    - 1 May
    - 25 December
    - 26 December

    Parameters
    ----------
    dt : timestamp to check
    """
    d = dt.date()
    # Check weekday (Mon=0 ... Fri=4)
    if d.weekday() >= 5:
        return False
    year = d.year
    good_friday, easter_monday = _get_easter_dates(year)
    holidays = {
        date(year, 1, 1),
        good_friday,
        easter_monday,
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    return d not in holidays


def apply_calendar_screen(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Remove return intervals that span an intervening eligible weekday.

    A return interval is identified by its return date t, which corresponds
    to the price change from prices.index[t-1] to prices.index[t].
    The interval fails the screen if ANY eligible weekday exists strictly
    between the two consecutive price dates.

    Do NOT recompute returns after deletions.

    Parameters
    ----------
    returns : DataFrame of log returns indexed by return dates
    prices  : aligned price DataFrame (used to determine interval endpoints)

    Returns
    -------
    Filtered returns DataFrame with screened-out rows removed.
    """
    price_dates = prices.index.to_list()
    # Build a set for fast lookup
    price_date_set = set(price_dates)

    keep_mask = []
    return_dates = returns.index.to_list()

    for ret_date in return_dates:
        # Find previous price date
        idx_in_prices = price_dates.index(ret_date)
        if idx_in_prices == 0:
            keep_mask.append(True)
            continue
        prev_price_date = price_dates[idx_in_prices - 1]

        # Check all calendar dates strictly between prev_price_date and ret_date
        cur = prev_price_date + timedelta(days=1)
        has_eligible_weekday = False
        while cur < ret_date:
            ts = pd.Timestamp(cur)
            if is_eligible_weekday(ts):
                has_eligible_weekday = True
                break
            cur += timedelta(days=1)

        keep_mask.append(not has_eligible_weekday)

    filtered = returns[keep_mask]
    return filtered


# ── Full pipeline ───────────────────────────────────────────────────────────────

def build_baseline_dataset(
    cache_dir: str = "/workspace/project/data",
    seed: int = 1121,
) -> dict:
    """Full data pipeline: download → log-returns → calendar screen.

    Methodology steps:
    1. Download aligned prices for all 52 tickers.
    2. Compute adjacent log returns (2,204 expected).
    3. Remove pre-2016 intervals.
    4. Apply calendar screen to post-2015 intervals.
    5. Retain baseline 1,990 observations (5 Jan 2016 – 27 Jun 2025).

    Parameters
    ----------
    cache_dir : directory for caching downloaded prices
    seed      : random seed for reproducibility

    Returns
    -------
    dict with keys:
        'prices'           : aligned price panel (~2205 rows)
        'raw_returns'      : all adjacent log returns (~2204 rows)
        'baseline_returns' : screened returns (~1990 rows)
        'pre2016_removed'  : count of pre-2016 intervals removed
        'screen_removed'   : count of post-2015 intervals removed
    """
    np.random.seed(seed)

    # Step 1: Download prices
    prices = download_prices(ALL_TICKERS, cache_dir=cache_dir)
    print(f"[DATA] Aligned price rows: {len(prices)} (target: 2205)")
    assert len(prices) >= 2000, f"Expected ~2205 price rows, got {len(prices)}"

    # Step 2: Compute log returns
    raw_returns = compute_log_returns(prices)
    print(f"[DATA] Raw return intervals: {len(raw_returns)} (target: 2204)")

    # Step 3: Remove pre-2016 intervals
    pre2016_mask = raw_returns.index < pd.Timestamp("2016-01-01")
    pre2016_removed = int(pre2016_mask.sum())
    post2015_returns = raw_returns[~pre2016_mask]
    print(f"[DATA] Pre-2016 intervals removed: {pre2016_removed} (target: 24)")

    # Step 4: Apply calendar screen to post-2015 intervals
    # The prices panel is used to determine interval endpoints
    screened_returns = apply_calendar_screen(post2015_returns, prices)
    screen_removed = len(post2015_returns) - len(screened_returns)
    print(f"[DATA] Post-2015 intervals removed by calendar screen: {screen_removed} (target: 190)")
    print(f"[DATA] Baseline observations retained: {len(screened_returns)} (target: 1990)")
    if len(screened_returns) > 0:
        print(f"[DATA] Baseline date range: {screened_returns.index[0].date()} to {screened_returns.index[-1].date()}")

    return {
        "prices": prices,
        "raw_returns": raw_returns,
        "baseline_returns": screened_returns,
        "pre2016_removed": pre2016_removed,
        "screen_removed": screen_removed,
    }
