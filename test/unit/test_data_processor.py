# test/unit/test_data_processor.py
# Unit tests for data_processor.py - pure calculation functions

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))

from data_processor import calc_alpha, calc_dneff, calc_dphi, calculate_v_pi


# ============================================================================
# calc_alpha tests - Optical loss calculation
# ============================================================================

class TestCalcAlpha:
    """Tests for calc_alpha() optical loss calculation."""

    @pytest.mark.unit
    def test_basic_calculation(self):
        """Test optical loss calculation with known values."""
        # Create neff with known imaginary part
        neff = np.array([2.5 + 1e-5j, 2.5 + 2e-5j, 2.5 + 3e-5j])
        wavelength = 1310e-9  # 1310 nm

        alpha = calc_alpha(neff, wavelength)

        assert alpha.shape == (3,)
        assert np.all(alpha >= 0), "Loss should be non-negative"
        # Loss should increase with imaginary part
        assert alpha[1] > alpha[0], "Loss should increase with Im(neff)"
        assert alpha[2] > alpha[1], "Loss should increase with Im(neff)"

    @pytest.mark.unit
    def test_zero_imaginary_part(self):
        """Test that zero imaginary part gives zero loss."""
        neff = np.array([2.5 + 0j, 2.51 + 0j])
        wavelength = 1310e-9

        alpha = calc_alpha(neff, wavelength)

        assert np.allclose(alpha, 0), "Zero imaginary part should give zero loss"

    @pytest.mark.unit
    def test_wavelength_dependence(self):
        """Test that loss depends on wavelength (alpha ~ 1/lambda)."""
        neff = np.array([2.5 + 1e-5j])

        alpha_1310 = calc_alpha(neff, 1310e-9)
        alpha_1550 = calc_alpha(neff, 1550e-9)

        # k0 = 2π/λ, so shorter wavelength = higher loss
        assert alpha_1310 > alpha_1550, "Shorter wavelength should have higher loss"

    @pytest.mark.unit
    def test_formula_correctness(self):
        """Verify the formula: alpha = 2 * k0 * Im(neff) * (10/ln(10)) * 1e-2."""
        neff = np.array([2.5 + 1e-5j])
        wavelength = 1310e-9

        k0 = 2 * np.pi / wavelength
        expected = 2 * k0 * 1e-5 * (10 / np.log(10)) * 1e-2

        alpha = calc_alpha(neff, wavelength)

        assert np.isclose(alpha[0], expected, rtol=1e-10)

    @pytest.mark.unit
    def test_array_shapes(self):
        """Test various array shapes work correctly."""
        wavelength = 1310e-9

        # 1D array
        neff_1d = np.array([2.5 + 1e-5j, 2.51 + 2e-5j])
        alpha_1d = calc_alpha(neff_1d, wavelength)
        assert alpha_1d.shape == (2,)

        # Single value
        neff_single = np.array([2.5 + 1e-5j])
        alpha_single = calc_alpha(neff_single, wavelength)
        assert alpha_single.shape == (1,)

    @pytest.mark.unit
    def test_units_dB_per_cm(self, sample_neff_array):
        """Verify output is in reasonable dB/cm range for typical values."""
        wavelength = 1310e-9

        alpha = calc_alpha(sample_neff_array, wavelength)

        # Typical PIN phase shifter loss is 0.5-10 dB/cm
        assert np.all(alpha < 100), "Loss should be < 100 dB/cm for typical values"
        assert np.all(alpha >= 0), "Loss should be non-negative"


# ============================================================================
# calc_dneff tests - Effective index change
# ============================================================================

class TestCalcDneff:
    """Tests for calc_dneff() effective index change calculation."""

    @pytest.mark.unit
    def test_first_element_zero(self):
        """Test that first element of dneff is always zero."""
        neff = np.array([2.5 + 1e-5j, 2.51 + 2e-5j, 2.52 + 3e-5j])

        d_neff = calc_dneff(neff)

        assert d_neff[0] == 0, "First element should be zero (reference)"

    @pytest.mark.unit
    def test_real_part_only(self):
        """Test that only real part is used."""
        neff = np.array([2.5 + 1e-5j, 2.51 + 2e-5j])

        d_neff = calc_dneff(neff)

        # Change should be 0.01 (real part difference)
        expected = np.array([0, 0.01])
        assert np.allclose(d_neff, expected, rtol=1e-10)

    @pytest.mark.unit
    def test_shape_preserved(self, sample_neff_array):
        """Test that output shape matches input."""
        d_neff = calc_dneff(sample_neff_array)

        assert d_neff.shape == sample_neff_array.shape

    @pytest.mark.unit
    def test_negative_change(self):
        """Test handling of negative index change (decreasing neff)."""
        neff = np.array([2.5 + 1e-5j, 2.49 + 2e-5j])

        d_neff = calc_dneff(neff)

        assert d_neff[1] < 0, "Decreasing neff should give negative dneff"

    @pytest.mark.unit
    def test_constant_neff(self):
        """Test that constant neff gives all zeros."""
        neff = np.array([2.5 + 1e-5j, 2.5 + 2e-5j, 2.5 + 3e-5j])

        d_neff = calc_dneff(neff)

        assert np.allclose(d_neff, 0), "Constant real neff should give zero change"


# ============================================================================
# calc_dphi tests - Phase shift calculation
# ============================================================================

class TestCalcDphi:
    """Tests for calc_dphi() phase shift calculation."""

    @pytest.mark.unit
    def test_zero_dneff_zero_phase(self):
        """Test that zero dneff gives zero phase shift."""
        d_neff = np.array([0, 0, 0])
        length = 0.5e-3
        wavelength = 1310e-9

        d_phi = calc_dphi(d_neff, length, wavelength)

        assert np.allclose(d_phi, 0)

    @pytest.mark.unit
    def test_formula_correctness(self):
        """Verify formula: delta_phi = (2π * d_neff * L) / λ."""
        d_neff = np.array([0, 1e-3])
        length = 0.5e-3  # 0.5 mm
        wavelength = 1310e-9

        expected = (2 * np.pi * d_neff * length) / wavelength
        d_phi = calc_dphi(d_neff, length, wavelength)

        assert np.allclose(d_phi, expected, rtol=1e-10)

    @pytest.mark.unit
    def test_length_scaling(self):
        """Test that phase shift scales with length."""
        d_neff = np.array([0, 1e-3])
        wavelength = 1310e-9

        d_phi_short = calc_dphi(d_neff, 0.5e-3, wavelength)
        d_phi_long = calc_dphi(d_neff, 1.0e-3, wavelength)

        # Phase should double with length
        assert np.allclose(d_phi_long, 2 * d_phi_short, rtol=1e-10)

    @pytest.mark.unit
    def test_wavelength_scaling(self):
        """Test that phase shift inversely scales with wavelength."""
        d_neff = np.array([0, 1e-3])
        length = 0.5e-3

        d_phi_1310 = calc_dphi(d_neff, length, 1310e-9)
        d_phi_2620 = calc_dphi(d_neff, length, 2620e-9)  # Double wavelength

        # Phase should halve with doubled wavelength
        assert np.allclose(d_phi_2620, d_phi_1310 / 2, rtol=1e-10)

    @pytest.mark.unit
    def test_shape_preserved(self):
        """Test output shape matches input."""
        d_neff = np.array([0, 1e-4, 2e-4, 3e-4, 4e-4])

        d_phi = calc_dphi(d_neff, 0.5e-3, 1310e-9)

        assert d_phi.shape == d_neff.shape

    @pytest.mark.unit
    def test_negative_dneff(self):
        """Test handling of negative dneff (gives negative phase)."""
        d_neff = np.array([0, -1e-3])

        d_phi = calc_dphi(d_neff, 0.5e-3, 1310e-9)

        assert d_phi[1] < 0, "Negative dneff should give negative phase"


# ============================================================================
# calculate_v_pi tests - V_pi interpolation
# ============================================================================

class TestCalculateVPi:
    """Tests for calculate_v_pi() V_pi interpolation."""

    @pytest.mark.unit
    def test_reaches_pi(self):
        """Test V_pi calculation when phase reaches π."""
        V = np.linspace(0, 2.5, 25)
        # Create dphi that crosses π at V=1.5
        dphi = np.linspace(0, 4, 25)  # Exceeds π

        v_pi = calculate_v_pi(V, dphi)

        assert not np.isnan(v_pi), "V_pi should be found when phase exceeds π"
        # At π radians, interpolate V
        expected_v_pi = np.interp(np.pi, dphi, V)
        assert np.isclose(v_pi, expected_v_pi, rtol=1e-6)

    @pytest.mark.unit
    def test_not_reached(self):
        """Test V_pi returns NaN when phase doesn't reach π."""
        V = np.linspace(0, 2.5, 25)
        dphi = np.linspace(0, 2, 25)  # Below π (~3.14)

        v_pi = calculate_v_pi(V, dphi)

        assert np.isnan(v_pi), "V_pi should be NaN when π not reached"

    @pytest.mark.unit
    def test_exactly_pi(self):
        """Test when max phase is exactly π."""
        V = np.linspace(0, 2.5, 25)
        dphi = np.linspace(0, np.pi, 25)

        v_pi = calculate_v_pi(V, dphi)

        assert not np.isnan(v_pi), "V_pi should be found at exactly π"
        assert np.isclose(v_pi, 2.5, rtol=1e-6), "V_pi should be max voltage"

    @pytest.mark.unit
    def test_already_at_pi(self):
        """Test when phase already starts above π."""
        V = np.linspace(0, 2.5, 25)
        dphi = np.linspace(np.pi, 2 * np.pi, 25)  # Starts at π

        v_pi = calculate_v_pi(V, dphi)

        assert not np.isnan(v_pi)
        assert np.isclose(v_pi, 0, atol=0.1), "V_pi should be near 0"

    @pytest.mark.unit
    def test_monotonic_increase(self):
        """Test with typical monotonically increasing phase."""
        V = np.linspace(0, 2.5, 100)
        # Simulate typical phase vs voltage curve
        dphi = 0.5 * V ** 1.5  # Non-linear but monotonic

        v_pi = calculate_v_pi(V, dphi)

        max_dphi = np.max(dphi)
        if max_dphi >= np.pi:
            assert not np.isnan(v_pi)
            assert 0 < v_pi <= 2.5
        else:
            assert np.isnan(v_pi)

    @pytest.mark.unit
    def test_empty_arrays(self):
        """Test handling of empty arrays raises ValueError."""
        V = np.array([])
        dphi = np.array([])

        # Empty arrays cause np.max to fail - this is expected behavior
        with pytest.raises(ValueError):
            calculate_v_pi(V, dphi)

    @pytest.mark.unit
    def test_single_point(self):
        """Test with single data point."""
        V = np.array([1.0])
        dphi = np.array([np.pi + 0.1])

        v_pi = calculate_v_pi(V, dphi)

        # Single point above π should return that voltage
        assert not np.isnan(v_pi)


# ============================================================================
# Integration tests with full calculation chain
# ============================================================================

class TestCalculationChain:
    """Integration tests combining multiple calculation functions."""

    @pytest.mark.unit
    def test_full_chain_reaches_pi(self, sample_neff_reaches_pi, sample_voltage_array):
        """Test full calculation chain that reaches π phase shift."""
        wavelength = 1310e-9
        length = 0.5e-3

        # Calculate intermediate results
        d_neff = calc_dneff(sample_neff_reaches_pi)
        alpha = calc_alpha(sample_neff_reaches_pi, wavelength)
        d_phi = calc_dphi(d_neff, length, wavelength)
        abs_dphi = np.abs(d_phi)
        v_pi = calculate_v_pi(sample_voltage_array, abs_dphi)

        # Verify chain
        assert d_neff[0] == 0
        assert np.all(alpha >= 0)
        assert d_phi[0] == 0
        # This fixture is designed to reach π
        assert np.max(abs_dphi) >= np.pi, "Should reach π with sample_neff_reaches_pi"
        assert not np.isnan(v_pi), "V_pi should be found"

    @pytest.mark.unit
    def test_full_chain_no_pi(self, sample_neff_no_pi, sample_voltage_array):
        """Test full calculation chain that doesn't reach π."""
        wavelength = 1310e-9
        length = 0.5e-3

        d_neff = calc_dneff(sample_neff_no_pi)
        d_phi = calc_dphi(d_neff, length, wavelength)
        abs_dphi = np.abs(d_phi)
        v_pi = calculate_v_pi(sample_voltage_array, abs_dphi)

        # This fixture is designed NOT to reach π
        assert np.max(abs_dphi) < np.pi, "Should not reach π with sample_neff_no_pi"
        assert np.isnan(v_pi), "V_pi should be NaN"

    @pytest.mark.unit
    def test_v_pi_l_calculation(self, sample_neff_reaches_pi, sample_voltage_array):
        """Test V_π*L calculation."""
        wavelength = 1310e-9
        length = 0.5e-3  # 0.5 mm

        d_neff = calc_dneff(sample_neff_reaches_pi)
        d_phi = calc_dphi(d_neff, length, wavelength)
        abs_dphi = np.abs(d_phi)
        v_pi = calculate_v_pi(sample_voltage_array, abs_dphi)

        if not np.isnan(v_pi):
            v_pi_l = v_pi * length * 1e3  # Convert to V*mm
            # Typical values are 0.5-2 V*mm
            assert v_pi_l > 0, "V_π*L should be positive"
            assert v_pi_l < 10, "V_π*L should be reasonable"
