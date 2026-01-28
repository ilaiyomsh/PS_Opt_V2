# test/unit/test_bo.py
# Unit tests for BO.py - Bayesian Optimization functions

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# calculate_loss_function tests
# ============================================================================

class TestCalculateLossFunction:
    """Tests for calculate_loss_function() cost calculation."""

    @pytest.mark.unit
    def test_success_case_basic(self):
        """Test cost calculation for successful simulation (reached π)."""
        from BO import calculate_loss_function

        # Valid v_pi_l indicates success
        cost = calculate_loss_function(
            alpha=1.5,      # dB/cm
            v_pi_l=0.8,     # V*mm
            max_dphi=np.pi + 0.2
        )

        assert cost < 0, "Cost should be negative (for maximization)"
        assert not np.isnan(cost), "Cost should not be NaN"

    @pytest.mark.unit
    def test_success_returns_negative(self):
        """Test that successful cost returns negative value for BayesOpt maximization."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(alpha=2.0, v_pi_l=1.0, max_dphi=np.pi)

        assert cost < 0, "Cost must be negative for BayesOpt maximization"

    @pytest.mark.unit
    def test_penalty_case_nan_vpil(self):
        """Test penalty when v_pi_l is NaN (didn't reach π)."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5,
            v_pi_l=np.nan,  # Failed to reach π
            max_dphi=2.0    # Less than π
        )

        assert cost < 0, "Penalty cost should also be negative"
        assert not np.isnan(cost), "Cost should not be NaN even for failed sim"

    @pytest.mark.unit
    def test_penalty_case_none_vpil(self):
        """Test penalty when v_pi_l is None."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5,
            v_pi_l=None,
            max_dphi=2.5
        )

        assert cost < 0
        assert not np.isnan(cost)

    @pytest.mark.unit
    def test_penalty_larger_than_success(self):
        """Test that penalty cost is worse (more negative) than success cost."""
        from BO import calculate_loss_function

        # Successful case
        success_cost = calculate_loss_function(
            alpha=2.0,
            v_pi_l=1.0,
            max_dphi=np.pi + 0.5
        )

        # Failed case (didn't reach π)
        penalty_cost = calculate_loss_function(
            alpha=2.0,
            v_pi_l=np.nan,
            max_dphi=2.0  # Below π
        )

        # Penalty should be more negative (worse for maximization)
        assert penalty_cost < success_cost, "Penalty should be worse than success"

    @pytest.mark.unit
    def test_lower_loss_is_better(self):
        """Test that lower optical loss gives better (less negative) cost."""
        from BO import calculate_loss_function

        cost_high_loss = calculate_loss_function(alpha=3.0, v_pi_l=1.0)
        cost_low_loss = calculate_loss_function(alpha=1.0, v_pi_l=1.0)

        assert cost_low_loss > cost_high_loss, "Lower loss should give better cost"

    @pytest.mark.unit
    def test_lower_vpil_is_better(self):
        """Test that lower V_π*L gives better (less negative) cost."""
        from BO import calculate_loss_function

        cost_high_vpil = calculate_loss_function(alpha=2.0, v_pi_l=2.0)
        cost_low_vpil = calculate_loss_function(alpha=2.0, v_pi_l=0.5)

        assert cost_low_vpil > cost_high_vpil, "Lower V_π*L should give better cost"

    @pytest.mark.unit
    def test_custom_weights(self):
        """Test cost calculation with custom weights."""
        from BO import calculate_loss_function

        weights_loss_heavy = {'loss': 0.9, 'vpil': 0.1}
        weights_vpil_heavy = {'loss': 0.1, 'vpil': 0.9}

        # With loss-heavy weights, loss matters more
        cost_loss = calculate_loss_function(
            alpha=1.0, v_pi_l=2.0, weights=weights_loss_heavy
        )
        cost_vpil = calculate_loss_function(
            alpha=1.0, v_pi_l=2.0, weights=weights_vpil_heavy
        )

        # The costs should be different with different weights
        assert cost_loss != cost_vpil

    @pytest.mark.unit
    def test_custom_targets(self):
        """Test cost calculation with custom targets."""
        from BO import calculate_loss_function

        # Default targets: loss=2.0, vpil=1.0
        cost_default = calculate_loss_function(alpha=2.0, v_pi_l=1.0)

        # Custom targets where current values are "at target"
        custom_targets = {'loss': 4.0, 'vpil': 2.0}
        cost_custom = calculate_loss_function(
            alpha=4.0, v_pi_l=2.0, targets=custom_targets
        )

        # Both should have same normalized values
        assert np.isclose(cost_default, cost_custom, rtol=0.1)

    @pytest.mark.unit
    def test_zero_alpha(self):
        """Test edge case with zero optical loss."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(alpha=0, v_pi_l=1.0)

        assert not np.isnan(cost)
        assert cost < 0

    @pytest.mark.unit
    def test_penalty_increases_with_distance_from_pi(self):
        """Test that penalty increases as max_dphi gets further from π."""
        from BO import calculate_loss_function

        # Closer to π (dphi=3.0)
        cost_close = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=3.0
        )

        # Further from π (dphi=1.0)
        cost_far = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=1.0
        )

        # Being further from π should give worse (more negative) cost
        assert cost_far < cost_close, "Further from π should give worse penalty"

    @pytest.mark.unit
    def test_penalty_with_zero_phase(self):
        """Test penalty when max_dphi is zero."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=0.0
        )

        assert cost < 0
        assert not np.isnan(cost)

    @pytest.mark.unit
    def test_penalty_with_none_phase(self):
        """Test penalty when max_dphi is None."""
        from BO import calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=None
        )

        assert cost < 0
        assert not np.isnan(cost)


# ============================================================================
# _is_duplicate_params tests
# ============================================================================

class TestIsDuplicateParams:
    """Tests for _is_duplicate_params() duplicate detection."""

    @pytest.mark.unit
    def test_exact_duplicate(self, sample_results_df):
        """Test detection of exact duplicate parameters."""
        from BO import _is_duplicate_params

        # Create params that exactly match row 0
        new_params = {
            'w_r': 400e-9,
            'h_si': 100e-9,
            'doping': 5e17,
            'S': 0.4e-6,
            'lambda': 1310e-9,
            'length': 0.5e-3
        }

        is_dup = _is_duplicate_params(new_params, sample_results_df)

        assert is_dup, "Exact match should be detected as duplicate"

    @pytest.mark.unit
    def test_within_tolerance(self, sample_results_df):
        """Test detection within tolerance threshold."""
        from BO import _is_duplicate_params

        # Params within 1% of row 0
        new_params = {
            'w_r': 400e-9 * 1.005,     # 0.5% different
            'h_si': 100e-9 * 0.995,    # 0.5% different
            'doping': 5e17 * 1.008,    # 0.8% different
            'S': 0.4e-6,
            'lambda': 1310e-9,
            'length': 0.5e-3
        }

        is_dup = _is_duplicate_params(new_params, sample_results_df, tolerance=0.01)

        assert is_dup, "Params within 1% should be detected as duplicate"

    @pytest.mark.unit
    def test_outside_tolerance(self, sample_results_df):
        """Test non-detection when outside tolerance."""
        from BO import _is_duplicate_params

        # Params more than 1% different
        new_params = {
            'w_r': 400e-9 * 1.05,      # 5% different
            'h_si': 100e-9 * 0.90,     # 10% different
            'doping': 5e17,
            'S': 0.4e-6,
            'lambda': 1310e-9,
            'length': 0.5e-3
        }

        is_dup = _is_duplicate_params(new_params, sample_results_df, tolerance=0.01)

        assert not is_dup, "Params > 1% different should not be duplicate"

    @pytest.mark.unit
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        from BO import _is_duplicate_params

        new_params = {'w_r': 400e-9, 'h_si': 100e-9}
        empty_df = pd.DataFrame()

        is_dup = _is_duplicate_params(new_params, empty_df)

        assert not is_dup, "Empty DataFrame should return False"

    @pytest.mark.unit
    def test_none_dataframe(self):
        """Test with None DataFrame."""
        from BO import _is_duplicate_params

        new_params = {'w_r': 400e-9, 'h_si': 100e-9}

        is_dup = _is_duplicate_params(new_params, None)

        assert not is_dup, "None DataFrame should return False"

    @pytest.mark.unit
    def test_missing_param_in_new(self, sample_results_df):
        """Test when new_params is missing a parameter."""
        from BO import _is_duplicate_params

        # Missing 'length' parameter
        new_params = {
            'w_r': 400e-9,
            'h_si': 100e-9,
            'doping': 5e17,
            'S': 0.4e-6,
            'lambda': 1310e-9
            # 'length' missing
        }

        is_dup = _is_duplicate_params(new_params, sample_results_df)

        assert not is_dup, "Missing param should not match"

    @pytest.mark.unit
    def test_custom_tolerance(self, sample_results_df):
        """Test with custom tolerance values."""
        from BO import _is_duplicate_params

        # Params 3% different
        new_params = {
            'w_r': 400e-9 * 1.03,
            'h_si': 100e-9 * 1.03,
            'doping': 5e17 * 1.03,
            'S': 0.4e-6 * 1.03,
            'lambda': 1310e-9 * 1.03,
            'length': 0.5e-3 * 1.03
        }

        # Should be duplicate with 5% tolerance
        is_dup_loose = _is_duplicate_params(new_params, sample_results_df, tolerance=0.05)
        # Should NOT be duplicate with 1% tolerance
        is_dup_strict = _is_duplicate_params(new_params, sample_results_df, tolerance=0.01)

        assert is_dup_loose, "Should be dup with 5% tolerance"
        assert not is_dup_strict, "Should not be dup with 1% tolerance"

    @pytest.mark.unit
    def test_zero_value_handling(self):
        """Test handling when existing value is zero (S parameter can be 0)."""
        from BO import _is_duplicate_params

        df = pd.DataFrame([{
            'w_r': 400e-9, 'h_si': 100e-9, 'doping': 5e17,
            'S': 0,  # Zero value
            'lambda': 1310e-9, 'length': 0.5e-3
        }])

        # New params with small S value
        new_params = {
            'w_r': 400e-9, 'h_si': 100e-9, 'doping': 5e17,
            'S': 1e-9,  # Very small but not zero
            'lambda': 1310e-9, 'length': 0.5e-3
        }

        is_dup = _is_duplicate_params(new_params, df, tolerance=0.01)

        # With old_val=0, rel_diff = abs(new_val)
        # 1e-9 < 0.01 tolerance, so it IS considered a duplicate
        assert is_dup, "Very small value near zero should be duplicate"

        # Larger S value should NOT be duplicate
        new_params_far = {**new_params, 'S': 0.1}  # 0.1 > 0.01 tolerance
        is_dup_far = _is_duplicate_params(new_params_far, df, tolerance=0.01)
        assert not is_dup_far, "Larger value should not be duplicate"


# ============================================================================
# Integration tests
# ============================================================================

class TestBOIntegration:
    """Integration tests for BO module."""

    @pytest.mark.unit
    def test_cost_ordering(self):
        """Test that costs are ordered correctly for optimization."""
        from BO import calculate_loss_function

        # Generate a range of simulations from good to bad
        costs = []

        # Good: low loss, low vpil, reached π
        costs.append(('good', calculate_loss_function(alpha=1.0, v_pi_l=0.5)))

        # Medium: moderate values
        costs.append(('medium', calculate_loss_function(alpha=2.0, v_pi_l=1.0)))

        # Bad: high values
        costs.append(('bad_values', calculate_loss_function(alpha=5.0, v_pi_l=3.0)))

        # Worst: failed to reach π
        costs.append(('failed', calculate_loss_function(
            alpha=2.0, v_pi_l=np.nan, max_dphi=2.0
        )))

        # Verify ordering: good > medium > bad > failed
        assert costs[0][1] > costs[1][1], "Good should beat medium"
        assert costs[1][1] > costs[2][1], "Medium should beat bad"
        assert costs[2][1] > costs[3][1], "Bad should beat failed"

    @pytest.mark.unit
    def test_pareto_frontier_cost(self):
        """Test costs for typical Pareto frontier trade-offs."""
        from BO import calculate_loss_function

        # Low loss, high vpil
        cost_a = calculate_loss_function(alpha=1.0, v_pi_l=2.0)

        # High loss, low vpil
        cost_b = calculate_loss_function(alpha=3.0, v_pi_l=0.5)

        # Both should be valid costs (not extreme penalties)
        assert cost_a < 0
        assert cost_b < 0
        # The relative ordering depends on weights, but both should be reasonable
