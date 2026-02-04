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
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(alpha=2.0, v_pi_l=1.0, max_dphi=np.pi)

        assert cost < 0, "Cost must be negative for BayesOpt maximization"

    @pytest.mark.unit
    def test_penalty_case_nan_vpil(self):
        """Test penalty when v_pi_l is NaN (didn't reach π)."""
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

        cost_high_loss = calculate_loss_function(alpha=3.0, v_pi_l=1.0)
        cost_low_loss = calculate_loss_function(alpha=1.0, v_pi_l=1.0)

        assert cost_low_loss > cost_high_loss, "Lower loss should give better cost"

    @pytest.mark.unit
    def test_lower_vpil_is_better(self):
        """Test that lower V_π*L gives better (less negative) cost."""
        from cost import calculate_cost as calculate_loss_function

        cost_high_vpil = calculate_loss_function(alpha=2.0, v_pi_l=2.0)
        cost_low_vpil = calculate_loss_function(alpha=2.0, v_pi_l=0.5)

        assert cost_low_vpil > cost_high_vpil, "Lower V_π*L should give better cost"

    @pytest.mark.unit
    def test_custom_weights(self):
        """Test cost calculation with custom weights."""
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(alpha=0, v_pi_l=1.0)

        assert not np.isnan(cost)
        assert cost < 0

    @pytest.mark.unit
    def test_penalty_increases_with_distance_from_pi(self):
        """Test that penalty increases as max_dphi gets further from π."""
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=0.0
        )

        assert cost < 0
        assert not np.isnan(cost)

    @pytest.mark.unit
    def test_penalty_with_none_phase(self):
        """Test penalty when max_dphi is None."""
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5, v_pi_l=np.nan, max_dphi=None
        )

        assert cost < 0
        assert not np.isnan(cost)


# ============================================================================
# Integration tests
# ============================================================================

class TestBOIntegration:
    """Integration tests for BO module."""

    @pytest.mark.unit
    def test_cost_ordering(self):
        """Test that costs are ordered correctly for optimization."""
        from cost import calculate_cost as calculate_loss_function

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
        from cost import calculate_cost as calculate_loss_function

        # Low loss, high vpil
        cost_a = calculate_loss_function(alpha=1.0, v_pi_l=2.0)

        # High loss, low vpil
        cost_b = calculate_loss_function(alpha=3.0, v_pi_l=0.5)

        # Both should be valid costs (not extreme penalties)
        assert cost_a < 0
        assert cost_b < 0
        # The relative ordering depends on weights, but both should be reasonable
