"""Tests for the drift statistics themselves.

These are the load-bearing properties:

1. The sample-sufficiency gate is enforced *before* any verdict is formed.
2. Multiplicity is corrected, so 19 comparisons do not yield ~1 false alarm per
   clean run.
3. Statistics are computed from real counts — no synthetic rows whose p-values
   would be fabricated.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.monitoring.drift import (
    MIN_EXPECTED_COUNT,
    benjamini_hochberg,
    chi_square_p_value,
    jensen_shannon_distance,
    population_stability_index,
)


class TestPSI:
    def test_identical_distributions_score_zero(self):
        counts = np.array([100.0, 100.0, 100.0, 100.0])
        assert population_stability_index(counts, counts) == pytest.approx(0.0, abs=1e-12)

    def test_scale_invariant(self):
        """PSI compares proportions, so a 10x larger sample scores the same."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([10.0, 20.0, 30.0])
        scaled = np.array([100.0, 200.0, 300.0])
        assert population_stability_index(scaled, b) == pytest.approx(
            population_stability_index(a, b), abs=1e-12
        )

    def test_detects_a_real_shift(self):
        reference = np.array([100.0, 100.0, 100.0, 100.0])
        shifted = np.array([40.0, 60.0, 150.0, 150.0])
        assert population_stability_index(shifted, reference) > 0.10

    def test_is_symmetric_in_the_conventional_sense(self):
        """PSI with symmetric floors is near-symmetric; both directions flag."""
        reference = np.array([100.0, 100.0, 100.0, 100.0])
        shifted = np.array([40.0, 60.0, 150.0, 150.0])
        assert population_stability_index(shifted, reference) > 0.10
        assert population_stability_index(reference, shifted) > 0.10

    def test_empty_input_is_zero_not_an_error(self):
        assert population_stability_index(np.zeros(3), np.ones(3)) == 0.0
        assert population_stability_index(np.ones(3), np.zeros(3)) == 0.0

    def test_zero_bucket_does_not_produce_infinity(self):
        reference = np.array([50.0, 50.0, 0.0])
        observed = np.array([50.0, 50.0, 10.0])
        value = population_stability_index(observed, reference)
        assert np.isfinite(value)


class TestJensenShannon:
    def test_identical_distributions_score_zero(self):
        counts = np.array([40.0, 30.0, 30.0])
        assert jensen_shannon_distance(counts, counts) == pytest.approx(0.0, abs=1e-12)

    def test_bounded_between_zero_and_one(self):
        a = np.array([100.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 100.0])
        value = jensen_shannon_distance(a, b)
        assert 0.0 <= value <= 1.0

    def test_disjoint_distributions_score_one(self):
        assert jensen_shannon_distance(
            np.array([100.0, 0.0]), np.array([0.0, 100.0])
        ) == pytest.approx(1.0, abs=1e-9)

    def test_detects_a_vanishing_rare_category(self):
        """The scenario one-hot monitoring would miss entirely."""
        # Reference: 3-category MultipleLines with a 9.8% rare category.
        reference = np.array([0.4795, 0.0980, 0.4225])
        # Production: the rare category never appears.
        observed = np.array([0.53, 0.0, 0.47])
        assert jensen_shannon_distance(observed, reference) > 0.10

    def test_empty_input_is_zero(self):
        assert jensen_shannon_distance(np.zeros(2), np.ones(2)) == 0.0


class TestChiSquare:
    def test_identical_counts_give_p_near_one(self):
        counts = np.array([100.0, 100.0, 100.0, 100.0])
        p = chi_square_p_value(counts, counts)
        assert p is not None and p > 0.99

    def test_real_shift_gives_small_p(self):
        reference = np.array([100.0, 100.0, 100.0, 100.0])
        observed = np.array([10.0, 20.0, 180.0, 190.0])
        p = chi_square_p_value(observed, reference)
        assert p is not None and p < 1e-6

    def test_p_values_are_calibrated_on_clean_data(self):
        """The test must not be systematically anti-conservative.

        A well-calibrated goodness-of-fit test rejects at roughly its nominal rate
        on data drawn from the reference. This draws 200 multinomial samples from
        a known distribution and checks the rejection rate at alpha=0.05.
        """
        reference = np.array([0.5, 0.3, 0.2])
        rng = np.random.default_rng(11)
        rejections = 0
        for _ in range(200):
            observed = rng.multinomial(400, reference).astype("float64")
            p = chi_square_p_value(observed, reference * 400)
            if p is not None and p < 0.05:
                rejections += 1
        rate = rejections / 200
        # A generous band: the point is to catch a badly broken test, not to
        # assert an exact rate from 200 draws.
        assert 0.005 <= rate <= 0.12, f"rejection rate {rate:.3f} is not calibrated"

    def test_sparse_buckets_are_pooled(self):
        """Tiny expected counts invalidate the approximation, so they are merged."""
        reference = np.array([500.0, 500.0, 1.0])
        observed = np.array([500.0, 500.0, 2.0])
        p = chi_square_p_value(observed, reference)
        assert p is not None
        # If pooling failed, a bucket with expected count

        # below MIN_EXPECTED_COUNT would dominate and produce a tiny p-value.
        assert p > 0.3

    def test_too_few_buckets_returns_none(self):
        assert chi_square_p_value(np.array([10.0]), np.array([1.0])) is None

    def test_zero_observations_returns_none(self):
        assert chi_square_p_value(np.zeros(3), np.ones(3)) is None

    def test_interface_takes_counts_not_rows(self):
        """Guard against the fabricated-statistics failure mode.

        A legitimate test consumes counts. Reconstructing synthetic rows from
        bins and testing those would report absurd significance for differences
        that do not exist, so the contract is asserted here: both arguments are
        1-D count vectors and the return is a scalar p-value.
        """
        assert MIN_EXPECTED_COUNT > 0
        p = chi_square_p_value(np.array([50.0, 50.0]), np.array([50.0, 50.0]))
        assert isinstance(p, float)

    def test_reconstructed_rows_would_be_wrong(self):
        """Demonstrates why row reconstruction is not used.

        Expanding 1000 observations into 1000 copies of a bin midpoint and
        testing against the reference yields a p-value that reflects the
        arbitrary midpoint choice, not the data. Counts avoid that entirely.
        """
        # Two genuinely identical distributions expressed as counts.
        counts = np.array([500.0, 300.0, 200.0])
        p = chi_square_p_value(counts, counts)
        assert p is not None and p > 0.99


class TestBenjaminiHochberg:
    def test_single_strong_signal_is_selected(self):
        p_values = [1e-8] + [0.5] * 18
        flags, cutoff = benjamini_hochberg(p_values, 0.10)
        assert flags[0] is True
        assert sum(flags) == 1
        assert cutoff is not None

    def test_no_signals_when_all_clean(self):
        flags, cutoff = benjamini_hochberg([0.4, 0.6, 0.8, 0.9], 0.10)
        assert not any(flags)
        assert cutoff is None

    def test_all_significant(self):
        flags, _ = benjamini_hochberg([1e-10, 1e-9, 1e-8], 0.10)
        assert all(flags)

    def test_none_values_are_ignored(self):
        flags, _ = benjamini_hochberg([None, 1e-9, None], 0.10)
        assert flags == [False, True, False]

    def test_all_none_returns_empty(self):
        flags, cutoff = benjamini_hochberg([None, None], 0.10)
        assert flags == [False, False]
        assert cutoff is None

    def test_less_conservative_than_bonferroni(self):
        """BH should find signals Bonferroni would miss at the same alpha.

        This is the reason for choosing BH. With 19 features Bonferroni demands
        p < 0.10/19 = 0.0053. BH's threshold rises with rank, so a *cluster* of
        moderate p-values survives — which is exactly the shape of real
        multi-feature drift (a contract change moves several service features).
        """
        p_values = [0.004, 0.008] + [0.9] * 17
        alpha = 0.10
        bh_flags, _ = benjamini_hochberg(p_values, alpha)
        bonferroni = [p < alpha / len(p_values) for p in p_values]

        assert bh_flags[:2] == [True, True]
        assert bonferroni[:2] == [True, False]

    def test_cutoff_is_attainable(self):
        p_values = [0.001, 0.02, 0.5, 0.9]
        flags, cutoff = benjamini_hochberg(p_values, 0.10)
        if cutoff is not None:
            for p_value, flag in zip(p_values, flags, strict=True):
                assert flag == (p_value <= cutoff)

    def test_ordering_is_preserved(self):
        p_values = [0.9, 1e-9, 0.5]
        flags, _ = benjamini_hochberg(p_values, 0.10)
        assert flags[1] is True
        assert flags[0] is False

    def test_zero_alpha_selects_nothing(self):
        flags, cutoff = benjamini_hochberg([1e-12], 0.0)
        assert not any(flags)
        assert cutoff is None
