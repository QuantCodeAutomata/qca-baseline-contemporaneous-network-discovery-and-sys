# Repository Knowledge — QCA European Bank Systemic Risk

## Project Overview
Replication of a causal-discovery systemic risk study for 51 European bank equities plus market proxy EXSA.DE.  
Core method: VAR(0)–DirectLiNGAM on GARCH(1,1)-Student-t volatility-standardized daily returns to recover same-day transmission networks and compute Systemic Impact Score (SIS) and Systemic Vulnerability Score (SVS).

## Directory Layout
```
src/              # All reusable modules
  data_loader.py  # Ticker universe, market caps, price download (yfinance), calendar screen
  garch_filter.py # Univariate GARCH(1,1)-t filtering, holdout extension
  lingam_utils.py # DirectLiNGAM wrapper, moving-block bootstrap Wald tests, BH/BY selection
  systemic_scores.py # Total-effect matrix, SIS/SVS/nSVS
  hsic_tests.py   # Exact Gaussian-kernel HSIC with circular-shift reference distribution
  tail_diagnostics.py # CoLift, DirLift, permutation tests
  benchmarks.py   # Diebold-Yilmaz FEVD, MES, partial rank correlations
  recursiveness.py # FastICA-Cholesky distance, Ilmonen index, backward-mass diagnostic
  simulation.py   # Synthetic recursive systems, volatility-filtering comparison

exp/              # Standalone scripts (exp_1.py – exp_8.py), each imports from src/
data/             # Cached price data (aligned_prices.parquet excluded from git) + derived artefacts
results/          # Deliverable outputs: CSVs, PNGs, RESULTS.md
tests/            # pytest suite (test_src.py, 93 tests)
```

## Key Constants (src/data_loader.py)
- `ALL_TICKERS`: 51 bank tickers + "EXSA.DE" (52 total)
- `BANK_TICKERS`: 51 bank-only tickers
- `MARKET_TICKER`: "EXSA.DE"
- `MARKET_CAPS`: dict of EUR bn market caps as of 1 Jul 2025 (hard-coded)
- Global seed: 1121; coefficient-bootstrap seed: 731

## Data Pipeline
1. Download adjusted closes via yfinance → cached at `data/aligned_prices.parquet` (excluded from git)
2. Inner-join → 2,205 aligned rows (20 Nov 2015 – 27 Jun 2025)
3. Adjacent log-returns → 2,204 intervals
4. Drop pre-2016 (24 intervals) + calendar screen (190 intervals) → 1,990 baseline rows
5. GARCH(1,1)-t filter per column → standardize → 1,990 × 52 Z matrix

## Calendar Screen Rule
Remove any return interval where the business-day gap between the two price dates is > 1 business day (after accounting for common-holiday adjustments: Jan 1, Good Friday, Easter Monday, May 1, Dec 25, Dec 26). Do NOT bridge gaps; keep original adjacent-price returns for retained intervals.

## DirectLiNGAM Settings
- `lingam.DirectLiNGAM(measure='pwling')` for pairwise likelihood-ratio ordering
- Bootstrap: moving-block resampler, block length 10, center but do NOT re-standardize
- BH correction at q=0.05 over 1,326 order-allowed coefficients
- Post-BH materiality: |B0_ij| >= 0.05

## Key Replication Targets (exp_1)
- Aligned rows: 2,205; retained baseline: 1,990
- Descriptive |B|>=0.05 links: ~312
- BH+materiality retained edges: ~214
- Top SIS: BKT.MC ~269.5, FBK.MI ~175.8, CABK.MC ~160.3
- Top SVS: ACA.PA ~96.4, GLE.PA ~91.1, BNP.PA ~82.7
- SIS-SVS rank correlation: ~-0.02

## Test Suite
- Location: `tests/test_src.py`
- Count: 93 tests (all passing)
- Run: `pytest tests/ -v`
- Known benign warning: `ConstantInputWarning` from Spearman on constant input in benchmark tests

## Experiment Dependencies
- exp_1 must run first → generates Z_baseline.parquet, C_B_baseline.parquet, exp1_B0.npy, etc.
- exp_2, exp_3, exp_5, exp_6, exp_7 depend on baseline data
- exp_4 depends on the baseline network (calls exp_1 functions internally or reads its outputs)
- exp_8 is fully independent (simulation study, no market data needed)

## Computational Notes
- exp_1: ~5-15 min (DirectLiNGAM on 52 vars + 1,000-resample bootstrap)
- exp_2: ~30-60 min (1,326 × 1,951 circular-shift HSIC tests)
- exp_3: ~hours (500 outer × 200 inner bootstrap re-discovery)
- exp_4: heavy (200 alternative orders × 1,000 inner bootstraps)
- exp_7: moderate (10 FastICA starts + 5,000 random order permutations)

## Git Notes
- `data/aligned_prices.parquet` is gitignored (raw Yahoo Finance download)
- All derived data files in `data/` (Z_baseline.parquet, exp1_B0.npy, etc.) are committed
- All outputs in `results/` are committed
