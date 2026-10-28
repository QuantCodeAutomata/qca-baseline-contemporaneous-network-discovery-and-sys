# QCA: Baseline Contemporaneous Network Discovery and Systemic Score Replication

Replication of a causal discovery–based systemic risk analysis for 51 European listed banks plus a market proxy (EXSA.DE).  
The methodology uses VAR(0)–DirectLiNGAM on volatility-standardized daily returns to recover same-day transmission networks and derive Systemic Impact Scores (SIS) and Systemic Vulnerability Scores (SVS).

---

## Repository Structure

```
.
├── data/           # Cached price data and computed intermediate artefacts
├── src/            # Core implementation modules
├── exp/            # Experiment scripts (exp_1.py – exp_8.py)
├── tests/          # pytest test suite
├── results/        # Output artefacts (CSVs, PNGs, RESULTS.md)
└── README.md
```

---

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Baseline contemporaneous network discovery and systemic score replication |
| exp_2 | Recovered-shock independence support via exact pairwise HSIC |
| exp_3 | Order and coefficient re-discovery bootstrap, backbone extraction |
| exp_4 | Tail-dependence diagnostics, temporal holdout, alternative-order comparator |
| exp_5 | Benchmark comparison with Diebold-Yilmaz connectedness and MES |
| exp_6 | Sensitivity and robustness modules across alternative samples |
| exp_7 | Recursiveness diagnostics via ICA-Cholesky distance and backward-mass |
| exp_8 | Appendix simulation study for edge recovery and volatility-filtering effects |

---

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run a single experiment
python exp/exp_1.py

# Run all experiments sequentially
for i in 1 2 3 4 5 6 7 8; do python exp/exp_$i.py; done
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Data

Adjusted close prices are downloaded from Yahoo Finance via `yfinance` and cached locally at `data/aligned_prices.parquet`.  
The 52-ticker universe covers 51 European bank equities plus `EXSA.DE` as market proxy.  
Date range: 20 Nov 2015 – 27 Jun 2025 (2,205 aligned rows; 1,990 retained baseline observations after calendar screening).

Market capitalizations are hard-coded as of 1 Jul 2025 (EUR bn) in `src/data_loader.py`.

---

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for a full summary of findings.

---

## Environment

- Python 3.10+
- Key packages: `arch`, `lingam`, `statsmodels`, `scipy`, `scikit-learn`, `yfinance`, `networkx`

---

## Seeds

| Purpose | Seed |
|---------|------|
| Global baseline | 1121 |
| Coefficient bootstrap | 731 |
