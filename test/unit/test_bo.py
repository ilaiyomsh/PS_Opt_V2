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
        """Test cost calculation for successful simulation (reached pi)."""
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(
            alpha=1.5,      # dB/cm
            v_pi_l=0.8,     # V*mm
        )

        assert cost < 0, "Cost should be negative (for maximization)"
        assert not np.isnan(cost), "Cost should not be NaN"

    @pytest.mark.unit
    def test_success_returns_negative(self):
        """Test that successful cost returns negative value for BayesOpt maximization."""
        from cost import calculate_cost as calculate_loss_function

        cost = calculate_loss_function(alpha=2.0, v_pi_l=1.0)

        assert cost < 0, "Cost must be negative for BayesOpt maximization"

    @pytest.mark.unit
    def test_failed_worse_than_success(self):
        """Test that failed sim (worst-case values) gives worse cost than success."""
        from cost import calculate_cost as calculate_loss_function

        # Successful case: good values
        success_cost = calculate_loss_function(
            alpha=2.0,
            v_pi_l=1.0,
        )

        # Failed case: worst-case values (high loss, high VpiL)
        failed_cost = calculate_loss_function(
            alpha=380.0,    # worst-case loss from sweep
            v_pi_l=1.075,   # V_MAX * L
        )

        # Failed should be more negative (worse for maximization)
        assert failed_cost < success_cost, "Failed (worst-case) should be worse than success"

    @pytest.mark.unit
    def test_lower_loss_is_better(self):
        """Test that lower optical loss gives better (less negative) cost."""
        from cost import calculate_cost as calculate_loss_function

        cost_high_loss = calculate_loss_function(alpha=3.0, v_pi_l=1.0)
        cost_low_loss = calculate_loss_function(alpha=1.0, v_pi_l=1.0)

        assert cost_low_loss > cost_high_loss, "Lower loss should give better cost"

    @pytest.mark.unit
    def test_lower_vpil_is_better(self):
        """Test that lower V_pi*L gives better (less negative) cost."""
        from cost import calculate_cost as calculate_loss_function

        cost_high_vpil = calculate_loss_function(alpha=2.0, v_pi_l=2.0)
        cost_low_vpil = calculate_loss_function(alpha=2.0, v_pi_l=0.5)

        assert cost_low_vpil > cost_high_vpil, "Lower V_pi*L should give better cost"

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
    def test_linear_scaling(self):
        """Test that doubling alpha doubles the loss contribution."""
        from cost import calculate_cost as calculate_loss_function

        # Use loss-only weights to isolate the loss contribution
        weights = {'loss': 1.0, 'vpil': 0.0}

        cost_1x = calculate_loss_function(alpha=2.0, v_pi_l=1.0, weights=weights)
        cost_2x = calculate_loss_function(alpha=4.0, v_pi_l=1.0, weights=weights)

        # Doubling alpha should double the (positive) cost
        assert np.isclose(cost_2x / cost_1x, 2.0, rtol=1e-6), \
            "Doubling alpha should double the loss contribution"

    @pytest.mark.unit
    def test_failed_sim_cost_is_finite(self):
        """Test that worst-case values produce finite, reasonable cost."""
        from cost import calculate_cost as calculate_loss_function

        # Typical worst-case: high loss from sweep, V_MAX * L
        cost = calculate_loss_function(alpha=380.0, v_pi_l=1.075)

        assert np.isfinite(cost), "Failed sim cost must be finite"
        assert cost < 0, "Cost must be negative for maximization"
        # Should be in a reasonable range, not 1e9
        assert abs(cost) < 1000, f"Cost {cost} is unreasonably large"

    @pytest.mark.unit
    def test_cost_continuity(self):
        """Test that barely-valid and barely-failed costs are same order of magnitude."""
        from cost import calculate_cost as calculate_loss_function

        # Barely valid: loss at V_pi, V_pi * L
        # Imagine a device that just barely reaches pi at V=2.4V with L=0.5mm
        barely_valid_cost = calculate_loss_function(alpha=10.0, v_pi_l=1.2)

        # Barely failed: worst-case loss slightly worse, V_MAX * L slightly higher
        barely_failed_cost = calculate_loss_function(alpha=12.0, v_pi_l=1.25)

        # These should be within an order of magnitude of each other
        ratio = abs(barely_failed_cost / barely_valid_cost)
        assert 0.1 < ratio < 10, \
            f"Barely-valid ({barely_valid_cost:.2f}) and barely-failed ({barely_failed_cost:.2f}) " \
            f"should be same order of magnitude (ratio={ratio:.2f})"


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

        # Good: low loss, low vpil, reached pi
        costs.append(('good', calculate_loss_function(alpha=1.0, v_pi_l=0.5)))

        # Medium: moderate values
        costs.append(('medium', calculate_loss_function(alpha=2.0, v_pi_l=1.0)))

        # Bad: high values
        costs.append(('bad_values', calculate_loss_function(alpha=5.0, v_pi_l=3.0)))

        # Worst: failed sim with worst-case values
        costs.append(('failed', calculate_loss_function(
            alpha=380.0, v_pi_l=1.075
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
