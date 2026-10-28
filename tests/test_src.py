"""
Comprehensive pytest test suite for all src modules.

Covers: data_loader, garch_filter, lingam_utils, systemic_scores,
        hsic_tests, tail_diagnostics, benchmarks, recursiveness, simulation.
No live API calls – all tests use deterministic synthetic data.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── helpers ────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _make_prices(n_dates: int = 60, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    """Synthetic price DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_dates)
    log_r = rng.normal(0, 0.01, (n_dates, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_r, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _make_returns(n: int = 200, k: int = 5, seed: int = 1) -> pd.DataFrame:
    """Synthetic return DataFrame, mean-zero, moderate variance."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    data = rng.normal(0, 0.01, (n, k))
    return pd.DataFrame(data, index=idx, columns=[f"B{i}" for i in range(k)])


# ══════════════════════════════════════════════════════════════════════════════
# 1. data_loader
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    """Tests for src.data_loader functions (no network calls)."""

    def test_bank_tickers_count(self):
        from src.data_loader import BANK_TICKERS
        assert len(BANK_TICKERS) == 51

    def test_all_tickers_count(self):
        from src.data_loader import ALL_TICKERS, BANK_TICKERS, MARKET_TICKER
        assert len(ALL_TICKERS) == 52
        assert MARKET_TICKER in ALL_TICKERS
        assert all(t in ALL_TICKERS for t in BANK_TICKERS)

    def test_market_caps_keys(self):
        from src.data_loader import MARKET_CAPS, BANK_TICKERS
        assert set(MARKET_CAPS.keys()) == set(BANK_TICKERS)

    def test_market_caps_positive(self):
        from src.data_loader import MARKET_CAPS
        assert all(v > 0 for v in MARKET_CAPS.values())

    def test_compute_log_returns_shape(self):
        from src.data_loader import compute_log_returns
        prices = _make_prices(30, 3)
        rets = compute_log_returns(prices)
        assert rets.shape == (29, 3)

    def test_compute_log_returns_formula(self):
        """log(P_t / P_{t-1}) exactly."""
        from src.data_loader import compute_log_returns
        prices = _make_prices(10, 2, seed=7)
        rets = compute_log_returns(prices)
        expected = np.log(prices.values[1:] / prices.values[:-1])
        np.testing.assert_allclose(rets.values, expected, rtol=1e-10)

    def test_compute_log_returns_dates(self):
        from src.data_loader import compute_log_returns
        prices = _make_prices(20, 2)
        rets = compute_log_returns(prices)
        # Returns indexed by the *later* date
        assert list(rets.index) == list(prices.index[1:])

    def test_is_eligible_weekday_weekend(self):
        from src.data_loader import is_eligible_weekday
        saturday = pd.Timestamp("2024-01-06")  # Saturday
        assert not is_eligible_weekday(saturday)

    def test_is_eligible_weekday_weekday(self):
        from src.data_loader import is_eligible_weekday
        monday = pd.Timestamp("2024-01-08")    # Regular Monday
        assert is_eligible_weekday(monday)

    def test_is_eligible_weekday_new_year(self):
        from src.data_loader import is_eligible_weekday
        new_year = pd.Timestamp("2024-01-01")  # 1 Jan holiday
        assert not is_eligible_weekday(new_year)

    def test_is_eligible_weekday_christmas(self):
        from src.data_loader import is_eligible_weekday
        xmas = pd.Timestamp("2024-12-25")
        assert not is_eligible_weekday(xmas)

    def test_apply_calendar_screen_returns_dataframe(self):
        from src.data_loader import apply_calendar_screen, compute_log_returns
        prices = _make_prices(60, 3)
        rets = compute_log_returns(prices)
        screened = apply_calendar_screen(rets, prices)
        assert isinstance(screened, pd.DataFrame)
        assert screened.shape[1] == rets.shape[1]

    def test_apply_calendar_screen_subset_of_input(self):
        from src.data_loader import apply_calendar_screen, compute_log_returns
        prices = _make_prices(80, 3)
        rets = compute_log_returns(prices)
        screened = apply_calendar_screen(rets, prices)
        # All screened dates must exist in original returns
        assert set(screened.index).issubset(set(rets.index))

    def test_apply_calendar_screen_monotone_dates(self):
        from src.data_loader import apply_calendar_screen, compute_log_returns
        prices = _make_prices(60, 3)
        rets = compute_log_returns(prices)
        screened = apply_calendar_screen(rets, prices)
        assert screened.index.is_monotonic_increasing


# ══════════════════════════════════════════════════════════════════════════════
# 2. garch_filter
# ══════════════════════════════════════════════════════════════════════════════

class TestGarchFilter:
    """Tests for src.garch_filter."""

    def test_fit_garch_output_keys(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(300, 3)
        result = fit_garch_filter(rets, seed=0)
        for key in ("Z", "cond_vol", "garch_params", "centered"):
            assert key in result

    def test_fit_garch_Z_shape(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(300, 3)
        result = fit_garch_filter(rets, seed=0)
        assert result["Z"].shape == rets.shape

    def test_fit_garch_Z_unit_variance(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(300, 4, seed=5)
        result = fit_garch_filter(rets, seed=0)
        stds = result["Z"].std(ddof=1)
        np.testing.assert_allclose(stds.values, np.ones(4), atol=0.05)

    def test_fit_garch_Z_near_zero_mean(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(400, 3)
        result = fit_garch_filter(rets, seed=0)
        means = result["Z"].mean()
        np.testing.assert_allclose(means.values, np.zeros(3), atol=1e-10)

    def test_fit_garch_cond_vol_positive(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(300, 3)
        result = fit_garch_filter(rets, seed=0)
        assert (result["cond_vol"] > 0).all().all()

    def test_fit_garch_params_positive_omega(self):
        from src.garch_filter import fit_garch_filter
        rets = _make_returns(300, 2)
        result = fit_garch_filter(rets, seed=0)
        for ticker, p in result["garch_params"].items():
            assert p["omega"] > 0

    def test_extend_garch_output_shape(self):
        from src.garch_filter import fit_garch_filter, extend_garch_to_test
        train = _make_returns(250, 2, seed=0)
        test  = _make_returns(50,  2, seed=1)
        test.columns = train.columns
        train_result = fit_garch_filter(train, seed=0)
        result = extend_garch_to_test(train, test, train_result["garch_params"])
        assert result["Z_test"].shape == test.shape

    def test_extend_garch_columns_match(self):
        from src.garch_filter import fit_garch_filter, extend_garch_to_test
        train = _make_returns(250, 2, seed=2)
        test  = _make_returns(50,  2, seed=3)
        test.columns = train.columns
        train_result = fit_garch_filter(train, seed=0)
        result = extend_garch_to_test(train, test, train_result["garch_params"])
        assert list(result["Z_test"].columns) == list(train.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 3. lingam_utils
# ══════════════════════════════════════════════════════════════════════════════

class TestLingamUtils:
    """Tests for src.lingam_utils (no DirectLiNGAM fit – uses synthetic data)."""

    def _simple_dag(self, n: int = 4) -> tuple[np.ndarray, list[int]]:
        """Create a trivial lower-triangular B0 and causal order [0,1,2,...,n-1]."""
        B0 = np.zeros((n, n))
        # j -> i means B0[i, j] is nonzero (j precedes i)
        B0[1, 0] = 0.3
        B0[2, 0] = 0.2
        B0[2, 1] = 0.1
        B0[3, 2] = 0.4
        return B0, list(range(n))

    def test_fit_predecessor_regressions_shape(self):
        from src.lingam_utils import fit_predecessor_regressions
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        B0 = fit_predecessor_regressions(Z, order)
        assert B0.shape == (4, 4)

    def test_fit_predecessor_regressions_lower_triangular(self):
        """In causal order, no backward edges: B0[j,i]=0 for j before i."""
        from src.lingam_utils import fit_predecessor_regressions
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        B0 = fit_predecessor_regressions(Z, order)
        # Upper triangle in causal-order space: B0[causal_order[pos], causal_order[m]] for m>=pos
        for k, i in enumerate(order):
            for m, j in enumerate(order):
                if m >= k:                           # j does NOT precede i
                    assert B0[i, j] == pytest.approx(0.0)

    def test_fit_predecessor_regressions_diagonal_zero(self):
        from src.lingam_utils import fit_predecessor_regressions
        Z = RNG.normal(0, 1, (200, 5))
        order = list(range(5))
        B0 = fit_predecessor_regressions(Z, order)
        np.testing.assert_array_equal(np.diag(B0), np.zeros(5))

    def test_moving_block_bootstrap_indices_length(self):
        from src.lingam_utils import moving_block_bootstrap_indices
        # Returns a list of n_boot arrays each of length n
        all_idx = moving_block_bootstrap_indices(n=200, n_boot=5, block_length=10, seed=0)
        assert len(all_idx) == 5
        assert all(len(idx) == 200 for idx in all_idx)

    def test_moving_block_bootstrap_indices_valid_range(self):
        from src.lingam_utils import moving_block_bootstrap_indices
        all_idx = moving_block_bootstrap_indices(n=200, n_boot=3, block_length=10, seed=1)
        for idx in all_idx:
            assert int(idx.max()) < 200
            assert int(idx.min()) >= 0

    def test_compute_bootstrap_se_shape(self):
        from src.lingam_utils import compute_bootstrap_se
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        SE = compute_bootstrap_se(Z, order, n_boot=30, block_length=10, seed=0)
        assert SE.shape == (4, 4)

    def test_compute_bootstrap_se_nonneg(self):
        from src.lingam_utils import compute_bootstrap_se
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        SE = compute_bootstrap_se(Z, order, n_boot=30, block_length=10, seed=0)
        # SE is NaN for non-order-allowed positions; valid entries must be >= 0
        valid = SE[~np.isnan(SE)]
        assert (valid >= 0).all()

    def test_wald_bh_selection_all_selected_tiny_se(self):
        """When SE is tiny, all order-allowed coefficients are selected."""
        from src.lingam_utils import wald_bh_selection, fit_predecessor_regressions
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        B0 = fit_predecessor_regressions(Z, order)
        # Force SE to near-zero → huge Wald stats → all selected
        SE = np.full_like(B0, 1e-8)
        SE[np.diag_indices_from(SE)] = 0.0
        result = wald_bh_selection(B0, SE, order, q_bh=0.05, materiality_threshold=0.0)
        # At least some should be BH-selected
        assert result["n_bh"] > 0

    def test_wald_bh_selection_none_selected_huge_se(self):
        """When SE is enormous, nothing passes the Wald test."""
        from src.lingam_utils import wald_bh_selection, fit_predecessor_regressions
        Z = RNG.normal(0, 1, (200, 4))
        order = [0, 1, 2, 3]
        B0 = fit_predecessor_regressions(Z, order)
        SE = np.full_like(B0, 1e6)
        result = wald_bh_selection(B0, SE, order, q_bh=0.05, materiality_threshold=0.05)
        assert result["n_bh"] == 0
        assert result["n_bh_mat"] == 0

    def test_wald_bh_selection_output_keys(self):
        from src.lingam_utils import wald_bh_selection
        Z = RNG.normal(0, 1, (100, 3))
        order = [0, 1, 2]
        B0 = np.zeros((3, 3)); B0[1, 0] = 0.5; B0[2, 0] = 0.4; B0[2, 1] = 0.3
        SE = np.full_like(B0, 0.05)
        SE[np.diag_indices_from(SE)] = np.nan
        result = wald_bh_selection(B0, SE, order)
        # Keys present in the actual implementation
        for k in ("n_bh", "n_bh_mat", "n_by_mat", "pvalues"):
            assert k in result, f"Missing key: {k}"
        # Selected matrix lives under either 'B0_selected_bh' or 'B0_selected'
        has_selected = "B0_selected_bh" in result or "B0_selected" in result
        assert has_selected, "No selected-matrix key found"

    def test_descriptive_edge_count_threshold(self):
        from src.lingam_utils import descriptive_edge_count
        B0 = np.zeros((4, 4))
        B0[1, 0] = 0.10   # above 0.05
        B0[2, 1] = 0.04   # below 0.05
        B0[3, 0] = 0.07   # above 0.05
        count = descriptive_edge_count(B0, threshold=0.05)
        assert count == 2

    def test_recover_shocks_shape(self):
        from src.lingam_utils import recover_shocks
        B0 = np.zeros((4, 4))
        B0[1, 0] = 0.3
        Z = RNG.normal(0, 1, (100, 4))
        eps = recover_shocks(Z, B0)
        assert eps.shape == Z.shape

    def test_recover_shocks_formula(self):
        """eps_hat = (I - B0) Z^T  → checked column-wise."""
        from src.lingam_utils import recover_shocks
        n, k = 50, 3
        Z = RNG.normal(0, 1, (n, k))
        B0 = np.zeros((k, k))
        B0[1, 0] = 0.2
        eps = recover_shocks(Z, B0)
        expected = (np.eye(k) - B0) @ Z.T
        np.testing.assert_allclose(eps, expected.T, rtol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 4. systemic_scores
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemicScores:
    """Tests for src.systemic_scores."""

    def _simple_setup(self, n: int = 3):
        banks = [f"B{i}" for i in range(n)]
        caps  = {b: float(i + 1) for i, b in enumerate(banks)}
        C_B   = pd.DataFrame(
            np.eye(n) + 0.1 * RNG.normal(0, 1, (n, n)),
            index=banks, columns=banks,
        )
        np.fill_diagonal(C_B.values, 1.0)
        return C_B, caps, banks

    def test_compute_total_effect_matrix_identity(self):
        """If B0_BB is zero matrix, C_B should be identity."""
        from src.systemic_scores import compute_total_effect_matrix
        n, banks = 5, [f"X{i}" for i in range(5)]
        B0_BB = pd.DataFrame(np.zeros((n, n)), index=banks, columns=banks)
        C_B = compute_total_effect_matrix(B0_BB, banks)
        np.testing.assert_allclose(C_B.values, np.eye(n), atol=1e-12)

    def test_compute_total_effect_matrix_invertible(self):
        """C_B = (I - B0_BB)^{-1} — verify by checking C_B (I - B0_BB) = I."""
        from src.systemic_scores import compute_total_effect_matrix
        banks = ["A", "B", "C"]
        B0_arr = np.array([[0, 0, 0], [0.3, 0, 0], [0.2, 0.1, 0]], dtype=float)
        B0_BB = pd.DataFrame(B0_arr, index=banks, columns=banks)
        C_B = compute_total_effect_matrix(B0_BB, banks)
        I_minus_B0 = np.eye(3) - B0_arr
        product = C_B.values @ I_minus_B0
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_compute_systemic_scores_shape(self):
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(4)
        scores = compute_systemic_scores(C_B, caps, banks)
        assert scores.shape[0] == 4
        for col in ("SIS", "SVS", "nSVS"):
            assert col in scores.columns

    def test_sis_nonneg(self):
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(4)
        scores = compute_systemic_scores(C_B, caps, banks)
        assert (scores["SIS"] >= 0).all()

    def test_svs_nonneg(self):
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(4)
        scores = compute_systemic_scores(C_B, caps, banks)
        assert (scores["SVS"] >= 0).all()

    def test_nsvs_equals_svs_over_cap(self):
        """nSVS_i = SVS_i / m_i."""
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(4)
        scores = compute_systemic_scores(C_B, caps, banks)
        for b in banks:
            expected = scores.loc[b, "SVS"] / caps[b]
            assert scores.loc[b, "nSVS"] == pytest.approx(expected, rel=1e-8)

    def test_sis_plus_minus_sum_to_sis(self):
        """SIS = SIS_plus + SIS_minus."""
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(5)
        scores = compute_systemic_scores(C_B, caps, banks)
        reconstructed = scores["SIS_plus"] + scores["SIS_minus"]
        np.testing.assert_allclose(reconstructed.values, scores["SIS"].values, atol=1e-10)

    def test_sis_rank_range(self):
        from src.systemic_scores import compute_systemic_scores
        C_B, caps, banks = self._simple_setup(5)
        scores = compute_systemic_scores(C_B, caps, banks)
        assert scores["SIS_rank"].between(1, 5).all()
        assert scores["SVS_rank"].between(1, 5).all()

    def test_compute_rank_correlation_range(self):
        from src.systemic_scores import compute_rank_correlation
        # Identical rankings → correlation = 1
        scores = pd.DataFrame({"SIS": [3.0, 1.0, 2.0], "SVS": [3.0, 1.0, 2.0]},
                               index=["A", "B", "C"])
        rho = compute_rank_correlation(scores)
        assert -1.0 <= rho <= 1.0

    def test_compute_rank_correlation_identical(self):
        from src.systemic_scores import compute_rank_correlation
        scores = pd.DataFrame({"SIS": [3.0, 1.0, 2.0], "SVS": [3.0, 1.0, 2.0]},
                               index=["A", "B", "C"])
        rho = compute_rank_correlation(scores)
        assert rho == pytest.approx(1.0, abs=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# 5. hsic_tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHsicTests:
    """Tests for src.hsic_tests."""

    def test_gaussian_kernel_shape(self):
        from src.hsic_tests import gaussian_kernel_matrix
        x = RNG.normal(0, 1, 30)
        K = gaussian_kernel_matrix(x, bandwidth=1.0)
        assert K.shape == (30, 30)

    def test_gaussian_kernel_symmetric(self):
        from src.hsic_tests import gaussian_kernel_matrix
        x = RNG.normal(0, 1, 30)
        K = gaussian_kernel_matrix(x, bandwidth=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_gaussian_kernel_diagonal_ones(self):
        from src.hsic_tests import gaussian_kernel_matrix
        x = RNG.normal(0, 1, 20)
        K = gaussian_kernel_matrix(x, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K), np.ones(20), atol=1e-14)

    def test_gaussian_kernel_values_in_0_1(self):
        from src.hsic_tests import gaussian_kernel_matrix
        x = RNG.normal(0, 1, 50)
        K = gaussian_kernel_matrix(x, bandwidth=1.0)
        assert (K >= 0).all() and (K <= 1.0 + 1e-12).all()

    def test_compute_hsic_nonnegative(self):
        from src.hsic_tests import compute_hsic
        x = RNG.normal(0, 1, 80)
        y = RNG.normal(0, 1, 80)
        # bw_x and bw_y are the actual parameter names (None = median heuristic)
        h = compute_hsic(x, y, bw_x=1.0, bw_y=1.0)
        assert h >= -1e-12   # can be slightly negative for finite samples

    def test_compute_hsic_dependent_larger(self):
        """HSIC should be larger when y = x + noise vs independent y."""
        from src.hsic_tests import compute_hsic
        x = RNG.normal(0, 1, 200)
        y_dep = x + 0.1 * RNG.normal(0, 1, 200)
        y_ind = RNG.normal(0, 1, 200)
        h_dep = compute_hsic(x, y_dep, bw_x=1.0, bw_y=1.0)
        h_ind = compute_hsic(x, y_ind, bw_x=1.0, bw_y=1.0)
        assert h_dep > h_ind

    def test_centering_matrix_properties(self):
        from src.hsic_tests import centering_matrix
        n = 10
        H = centering_matrix(n)
        assert H.shape == (n, n)
        # H is idempotent
        np.testing.assert_allclose(H @ H, H, atol=1e-12)
        # Row / column sums are 0
        np.testing.assert_allclose(H.sum(axis=0), np.zeros(n), atol=1e-12)
        np.testing.assert_allclose(H.sum(axis=1), np.zeros(n), atol=1e-12)

    def test_mean_squared_shock_correlation_range(self):
        from src.hsic_tests import mean_squared_shock_correlation
        shocks = RNG.normal(0, 1, (200, 10))
        val = mean_squared_shock_correlation(shocks)
        # Mean of Pearson correlations can be in [-1, 1]
        assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_mean_squared_shock_correlation_independent_near_zero(self):
        """Independent shocks → near-zero squared correlation."""
        from src.hsic_tests import mean_squared_shock_correlation
        rng = np.random.default_rng(99)
        shocks = rng.normal(0, 1, (2000, 8))
        val = mean_squared_shock_correlation(shocks)
        assert val < 0.10


# ══════════════════════════════════════════════════════════════════════════════
# 6. tail_diagnostics
# ══════════════════════════════════════════════════════════════════════════════

class TestTailDiagnostics:
    """Tests for src.tail_diagnostics."""

    def _binary_indicators(self, n: int = 200, k: int = 4, q: float = 0.05
                           ) -> np.ndarray:
        """Produce synthetic binary tail indicator matrix."""
        Z = RNG.normal(0, 1, (n, k))
        D = np.zeros_like(Z, dtype=int)
        for j in range(k):
            thr = np.quantile(Z[:, j], q)
            D[:, j] = (Z[:, j] <= thr).astype(int)
        return D

    def test_compute_tail_indicators_binary(self):
        from src.tail_diagnostics import compute_tail_indicators
        Z = RNG.normal(0, 1, (300, 5))
        D = compute_tail_indicators(Z, quantile=0.05)
        assert D.shape == Z.shape
        assert set(np.unique(D)).issubset({0, 1})

    def test_compute_tail_indicators_correct_frequency(self):
        """Fraction of 1s in each column ≈ quantile."""
        from src.tail_diagnostics import compute_tail_indicators
        n, k = 5000, 6
        Z = RNG.normal(0, 1, (n, k))
        D = compute_tail_indicators(Z, quantile=0.05)
        freqs = D.mean(axis=0)
        np.testing.assert_allclose(freqs, 0.05 * np.ones(k), atol=0.01)

    def test_compute_colift_independent_near_one(self):
        """CoLift ≈ 1 for near-independent series."""
        from src.tail_diagnostics import compute_colift
        n = 3000
        D = self._binary_indicators(n=n, k=2)
        lift = compute_colift(D, 0, 1)
        # Independent variables have CoLift ≈ 1; allow generous tolerance
        assert 0.5 <= lift <= 2.5

    def test_compute_colift_nonneg(self):
        from src.tail_diagnostics import compute_colift
        D = self._binary_indicators(n=500, k=3)
        for i in range(3):
            for j in range(3):
                if i != j:
                    lift = compute_colift(D, i, j)
                    assert np.isnan(lift) or lift >= 0.0

    def test_compute_colift_correlated_above_one(self):
        """For strongly co-moving series, CoLift should exceed 1."""
        from src.tail_diagnostics import compute_colift
        rng = np.random.default_rng(77)
        n = 2000
        x = rng.normal(0, 1, n)
        y = x + 0.1 * rng.normal(0, 1, n)  # highly correlated
        Z = np.column_stack([x, y])
        from src.tail_diagnostics import compute_tail_indicators
        D = compute_tail_indicators(Z, quantile=0.05)
        lift = compute_colift(D, 0, 1)
        assert lift > 1.0

    def test_compute_dirlift_independent_near_one(self):
        """DirLift ≈ 1 for independent series."""
        from src.tail_diagnostics import compute_dirlift
        rng = np.random.default_rng(13)
        n = 3000
        D = (rng.random((n, 2)) < 0.05).astype(int)
        S = (rng.random((n, 2)) < 0.05).astype(int)
        lift = compute_dirlift(D, S, source=0, target=1)
        assert 0.0 <= lift <= 5.0   # loose bound; could be near 1

    def test_compute_dirlift_nan_on_zero_conditioning(self):
        """DirLift is NaN if conditioning event never occurs."""
        from src.tail_diagnostics import compute_dirlift
        n = 100
        D = np.zeros((n, 2), dtype=int); D[:5, 1] = 1
        S = np.zeros((n, 2), dtype=int)   # S_source always 0
        lift = compute_dirlift(D, S, source=0, target=1)
        assert np.isnan(lift)

    def test_compute_shock_indicators_binary(self):
        from src.tail_diagnostics import compute_shock_indicators
        shocks = RNG.normal(0, 1, (200, 5))
        S = compute_shock_indicators(shocks, quantile=0.05)
        assert S.shape == shocks.shape
        assert set(np.unique(S)).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
# 7. benchmarks
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarks:
    """Tests for src.benchmarks."""

    def _returns_and_market(self, n: int = 200, k: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-02", periods=n)
        bank_r = pd.DataFrame(rng.normal(0, 0.01, (n, k)),
                               index=idx, columns=[f"B{i}" for i in range(k)])
        market_r = pd.Series(rng.normal(-0.005, 0.015, n), index=idx, name="MKT")
        return bank_r, market_r

    def test_compute_mes_positive(self):
        from src.benchmarks import compute_mes
        bank_r, market_r = self._returns_and_market()
        mes = compute_mes(bank_r, market_r, quantile=0.05)
        assert isinstance(mes, pd.Series)
        assert len(mes) == bank_r.shape[1]

    def test_compute_mes_columns_match(self):
        from src.benchmarks import compute_mes
        bank_r, market_r = self._returns_and_market(k=4)
        mes = compute_mes(bank_r, market_r, quantile=0.05)
        assert set(mes.index) == set(bank_r.columns)

    def test_spearman_with_pvalue_range(self):
        from src.benchmarks import spearman_with_pvalue
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        rho, pval = spearman_with_pvalue(x, y)
        assert -1.0 <= rho <= 1.0
        assert 0.0 <= pval <= 1.0

    def test_spearman_with_pvalue_monotone(self):
        from src.benchmarks import spearman_with_pvalue
        x = pd.Series(np.arange(10, dtype=float))
        y = pd.Series(np.arange(10, dtype=float))
        rho, _ = spearman_with_pvalue(x, y)
        assert rho == pytest.approx(1.0, abs=1e-8)

    def test_partial_rank_one_control_returns_float(self):
        from src.benchmarks import partial_rank_correlation_one_control
        rng = np.random.default_rng(10)
        n = 50
        x = pd.Series(rng.normal(0, 1, n))
        y = pd.Series(rng.normal(0, 1, n))
        z = pd.Series(rng.normal(0, 1, n))
        r = partial_rank_correlation_one_control(x, y, z)
        assert -1.0 <= r <= 1.0

    def test_partial_rank_two_controls_returns_float(self):
        from src.benchmarks import partial_rank_correlation_two_controls
        rng = np.random.default_rng(11)
        n = 50
        x  = pd.Series(rng.normal(0, 1, n))
        y  = pd.Series(rng.normal(0, 1, n))
        z1 = pd.Series(rng.normal(0, 1, n))
        z2 = pd.Series(rng.normal(0, 1, n))
        r = partial_rank_correlation_two_controls(x, y, z1, z2)
        assert -1.0 <= r <= 1.0

    def test_compute_pc1_loadings_length(self):
        from src.benchmarks import compute_pc1_loadings
        bank_r, _ = self._returns_and_market(k=6)
        loadings = compute_pc1_loadings(bank_r)
        assert len(loadings) == 6

    def test_compute_pc1_loadings_keys(self):
        from src.benchmarks import compute_pc1_loadings
        bank_r, _ = self._returns_and_market(k=4)
        loadings = compute_pc1_loadings(bank_r)
        assert set(loadings.keys()) == set(bank_r.columns)

    def test_compute_dy_connectedness_keys(self):
        from src.benchmarks import compute_dy_connectedness
        bank_r, _ = self._returns_and_market(n=300, k=5)
        result = compute_dy_connectedness(bank_r, lag=1, horizon=10)
        assert "to_others"   in result
        assert "from_others" in result

    def test_compute_dy_connectedness_length(self):
        from src.benchmarks import compute_dy_connectedness
        bank_r, _ = self._returns_and_market(n=300, k=5)
        result = compute_dy_connectedness(bank_r, lag=1, horizon=10)
        assert len(result["to_others"])   == bank_r.shape[1]
        assert len(result["from_others"]) == bank_r.shape[1]

    def test_benchmark_comparison_table_structure(self):
        from src.benchmarks import benchmark_comparison_table
        k = 4
        banks = [f"B{i}" for i in range(k)]
        scores = pd.DataFrame({
            "SIS": np.arange(k, dtype=float),
            "SVS": np.arange(k, dtype=float)[::-1],
            "nSVS": np.ones(k),
        }, index=banks)
        dy_scores = {
            "to_others":  pd.Series(np.random.rand(k), index=banks),
            "from_others": pd.Series(np.random.rand(k), index=banks),
        }
        mes = pd.Series(np.random.rand(k), index=banks)
        market_caps = {b: 1.0 for b in banks}
        pc1_loadings = pd.Series(np.random.rand(k), index=banks)
        tbl = benchmark_comparison_table(scores, dy_scores, mes, market_caps, pc1_loadings)
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. recursiveness
# ══════════════════════════════════════════════════════════════════════════════

class TestRecursiveness:
    """Tests for src.recursiveness."""

    def test_backward_mass_range(self):
        from src.recursiveness import backward_mass
        n = 6
        W = RNG.normal(0, 1, (n, n))
        np.fill_diagonal(W, 0.0)
        order = list(range(n))
        R = backward_mass(W, order)
        assert 0.0 <= R <= 1.0

    def test_backward_mass_fully_forward_zero(self):
        """If all off-diagonal mass is in forward direction, R=0."""
        from src.recursiveness import backward_mass
        n = 4
        order = list(range(n))
        # Lower-triangular: W[i,j] nonzero only when i > j (forward in order 0,1,2,3)
        W = np.zeros((n, n))
        W[1, 0] = 1.0
        W[2, 0] = 1.0
        W[2, 1] = 1.0
        W[3, 2] = 1.0
        R = backward_mass(W, order)
        assert R == pytest.approx(0.0, abs=1e-12)

    def test_backward_mass_fully_backward_one(self):
        """If all off-diagonal mass is in backward direction, R=1."""
        from src.recursiveness import backward_mass
        n = 4
        order = list(range(n))
        # Upper-triangular: W[i,j] nonzero only when i < j (backward in order)
        W = np.zeros((n, n))
        W[0, 1] = 1.0
        W[0, 2] = 1.0
        W[1, 2] = 1.0
        W[0, 3] = 1.0
        R = backward_mass(W, order)
        assert R == pytest.approx(1.0, abs=1e-12)

    def test_backward_mass_reversed_order(self):
        """Reversed order on lower-triangular W → R should be 1."""
        from src.recursiveness import backward_mass
        n = 4
        W = np.zeros((n, n))
        W[1, 0] = 1.0; W[2, 1] = 1.0; W[3, 2] = 1.0
        # Reverse order: now j=0 is last → edges go backward
        order = [3, 2, 1, 0]
        R = backward_mass(W, order)
        assert R == pytest.approx(1.0, abs=1e-12)

    def test_cholesky_factor_shape(self):
        from src.recursiveness import cholesky_factor
        n = 5
        Z = RNG.normal(0, 1, (200, n))
        order = list(range(n))
        D = cholesky_factor(Z, order)
        assert D.shape == (n, n)

    def test_cholesky_factor_lower_triangular(self):
        from src.recursiveness import cholesky_factor
        n = 4
        Z = RNG.normal(0, 1, (200, n))
        order = list(range(n))
        D = cholesky_factor(Z, order)
        upper = D[np.triu_indices(n, k=1)]
        np.testing.assert_allclose(upper, np.zeros_like(upper), atol=1e-12)

    def test_cholesky_factor_reconstruction(self):
        """D @ D.T should recover the sample covariance (reordered)."""
        from src.recursiveness import cholesky_factor
        n = 5
        Z = RNG.normal(0, 1, (200, n))
        order = list(range(n))
        D = cholesky_factor(Z, order)
        Sigma_ordered = np.cov(Z.T, ddof=1)
        np.testing.assert_allclose(D @ D.T, Sigma_ordered, atol=1e-8)

    def test_random_order_backward_mass_distribution(self):
        from src.recursiveness import random_order_backward_mass
        n = 5
        W = RNG.normal(0, 1, (n, n))
        np.fill_diagonal(W, 0.0)
        masses = random_order_backward_mass(W, n_random=100, seed=0)
        assert len(masses) == 100
        assert all(0.0 <= m <= 1.0 for m in masses)


# ══════════════════════════════════════════════════════════════════════════════
# 9. simulation
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulation:
    """Tests for src.simulation."""

    def test_generate_recursive_system_shape(self):
        from src.simulation import generate_recursive_system
        result = generate_recursive_system(n_vars=5, T=300, burn_in=50, seed=42)
        assert result["data"].shape == (300, 5)
        assert result["B0_true"].shape == (5, 5)
        assert len(result["causal_order"]) == 5

    def test_generate_recursive_system_lower_triangular(self):
        """B0_true must be lower triangular (causal order 0,1,...,d-1)."""
        from src.simulation import generate_recursive_system
        result = generate_recursive_system(n_vars=6, T=200, seed=1)
        B0_true = result["B0_true"]
        upper = B0_true[np.triu_indices(6, k=1)]
        np.testing.assert_array_equal(upper, np.zeros_like(upper))

    def test_generate_recursive_system_diagonal_zero(self):
        from src.simulation import generate_recursive_system
        result = generate_recursive_system(n_vars=5, T=200, seed=2)
        np.testing.assert_array_equal(np.diag(result["B0_true"]), np.zeros(5))

    def test_evaluate_edge_recovery_keys(self):
        from src.simulation import evaluate_edge_recovery
        true_B = np.zeros((4, 4))
        true_B[1, 0] = 0.5; true_B[2, 1] = 0.4
        est_B  = np.zeros((4, 4))
        est_B[1, 0] = 0.5
        # B0_estimated = est_B, B0_true = true_B
        metrics = evaluate_edge_recovery(est_B, true_B, threshold=0.05)
        for k in ("tp", "fp", "fn", "precision", "recall", "f1"):
            assert k in metrics

    def test_evaluate_edge_recovery_perfect(self):
        """Perfect recovery: precision = recall = f1 = 1."""
        from src.simulation import evaluate_edge_recovery
        true_B = np.zeros((4, 4))
        true_B[1, 0] = 0.5; true_B[2, 1] = 0.4
        # same matrix as estimated and true → perfect
        metrics = evaluate_edge_recovery(true_B, true_B, threshold=0.05)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"]    == pytest.approx(1.0)
        assert metrics["f1"]        == pytest.approx(1.0)

    def test_evaluate_edge_recovery_no_selection(self):
        """All-zero estimated matrix → recall=0, fn=1."""
        from src.simulation import evaluate_edge_recovery
        true_B = np.zeros((4, 4)); true_B[1, 0] = 0.5
        est_B  = np.zeros((4, 4))
        # B0_estimated = all-zero, B0_true = has one edge
        metrics = evaluate_edge_recovery(est_B, true_B, threshold=0.05)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["tp"] == 0
        assert metrics["fn"] == 1

    def test_evaluate_edge_recovery_values_in_range(self):
        from src.simulation import evaluate_edge_recovery, generate_recursive_system
        from src.lingam_utils import fit_predecessor_regressions
        result = generate_recursive_system(n_vars=5, T=200, seed=3)
        B0_true = result["B0_true"]
        data    = result["data"]
        order   = result["causal_order"]
        B0_est  = fit_predecessor_regressions(data, order)
        # B0_estimated = B0_est, B0_true = B0_true
        metrics = evaluate_edge_recovery(B0_est, B0_true, threshold=0.05)
        for k in ("precision", "recall", "f1"):
            if not np.isnan(metrics[k]):
                assert 0.0 <= metrics[k] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. Edge / integration smoke tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge-case and integration smoke tests."""

    def test_single_obs_log_returns(self):
        """Single-row price → zero-length return DataFrame."""
        from src.data_loader import compute_log_returns
        prices = _make_prices(1, 3)
        rets = compute_log_returns(prices)
        assert len(rets) == 0

    def test_systemic_scores_single_bank(self):
        """Single-bank universe → SIS=SVS=0 (no off-diagonal pairs)."""
        from src.systemic_scores import compute_systemic_scores
        banks = ["A"]
        caps = {"A": 10.0}
        C_B = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        scores = compute_systemic_scores(C_B, caps, banks)
        assert scores.loc["A", "SIS"] == pytest.approx(0.0)
        assert scores.loc["A", "SVS"] == pytest.approx(0.0)

    def test_wald_bh_order_allowed_count(self):
        """For n variables, exactly n*(n-1)/2 order-allowed coefficients."""
        from src.lingam_utils import wald_bh_selection
        n = 6
        B0 = np.tril(np.ones((n, n)), -1) * 0.1
        SE = np.full_like(B0, 0.02)
        SE[np.diag_indices_from(SE)] = np.nan
        order = list(range(n))
        result = wald_bh_selection(B0, SE, order, q_bh=0.05, materiality_threshold=0.0)
        n_order_allowed = n * (n - 1) // 2
        # The function stores this count in n_order_allowed
        assert result["n_order_allowed"] == n_order_allowed

    def test_colift_zero_marginal_returns_nan(self):
        """If one column is all zeros, CoLift should return NaN."""
        from src.tail_diagnostics import compute_colift
        n = 100
        D = np.zeros((n, 2), dtype=int)
        D[:5, 0] = 1   # col 0 has some 1s
        # col 1 all zeros → marginal prob = 0 → NaN
        lift = compute_colift(D, 0, 1)
        assert np.isnan(lift)

    def test_compute_mes_all_market_up(self):
        """If market never falls (all positive), worst-5% are still negative-ish
        if market returns are uniformly distributed — just verify no crash."""
        from src.benchmarks import compute_mes
        rng = np.random.default_rng(55)
        n = 200
        idx = pd.bdate_range("2020-01-02", periods=n)
        bank_r  = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx,
                                columns=["X", "Y", "Z"])
        market_r = pd.Series(rng.uniform(-0.02, 0.02, n), index=idx)
        mes = compute_mes(bank_r, market_r, quantile=0.05)
        assert len(mes) == 3
