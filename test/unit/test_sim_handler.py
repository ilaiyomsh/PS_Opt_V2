# test/unit/test_sim_handler.py
# Unit tests for sim_handler module
# Tests run_full_simulation() returns merged DataFrame with electrical and optical data

import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_6_params():
    """Sample 6 parameters for simulation."""
    return {
        'w_r': 400e-9,       # 400nm rib width
        'h_si': 100e-9,      # 100nm silicon height
        'doping': 5e17,      # 5e17 cm^-3 doping
        'S': 0.4e-6,         # 400nm spacing
        'lambda': 1310e-9,   # 1310nm wavelength
        'length': 0.5e-3     # 0.5mm device length
    }


@pytest.fixture
def sample_charge_data():
    """Sample CHARGE simulation output data."""
    n_points = 25
    V_drain = np.linspace(0, 2.5, n_points)
    n = np.linspace(1e10, 1e12, n_points)  # electron count
    p = np.linspace(1e10, 1e12, n_points)  # hole count
    return {
        'V_drain': V_drain.reshape(-1, 1),
        'n': n.reshape(-1, 1),
        'p': p.reshape(-1, 1),
    }


@pytest.fixture
def sample_optical_data():
    """Sample FDE/optical simulation output data."""
    n_points = 25
    # Complex effective index: real part ~ 2.5, imag part ~ 1e-5
    real_part = np.linspace(2.5, 2.503, n_points)
    imag_part = np.linspace(1e-5, 3e-5, n_points)
    neff = real_part + 1j * imag_part
    return {
        'neff': neff.reshape(-1, 1)
    }


@pytest.fixture
def mock_lumapi_full(sample_charge_data, sample_optical_data):
    """Full mock of lumapi with DEVICE and MODE sessions."""
    mock_lumapi = MagicMock()

    # Mock DEVICE (CHARGE) session
    mock_charge = MagicMock()
    mock_charge.getresult.return_value = sample_charge_data
    mock_lumapi.DEVICE.return_value = mock_charge

    # Mock MODE (FDE) session
    mock_fde = MagicMock()
    mock_fde.getsweepresult.return_value = sample_optical_data
    mock_lumapi.MODE.return_value = mock_fde

    return mock_lumapi


# ============================================================================
# Tests for run_full_simulation return type
# ============================================================================

class TestRunFullSimulationReturnType:
    """Tests for run_full_simulation() return type and structure."""

    @pytest.mark.unit
    def test_returns_tuple(self, sample_6_params, mock_lumapi_full):
        """Test that run_full_simulation returns a tuple."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            result = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert isinstance(result, tuple), "Should return a tuple"
            assert len(result) == 2, "Should return tuple of (df, timing)"

    @pytest.mark.unit
    def test_returns_dataframe(self, sample_6_params, mock_lumapi_full):
        """Test that first element is a pandas DataFrame."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert isinstance(df, pd.DataFrame), "First element should be DataFrame"

    @pytest.mark.unit
    def test_returns_timing_dict(self, sample_6_params, mock_lumapi_full):
        """Test that second element is a timing dictionary."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert isinstance(timing, dict), "Second element should be dict"
            assert 'charge_time' in timing, "Should have charge_time"
            assert 'fde_time' in timing, "Should have fde_time"


# ============================================================================
# Tests for DataFrame structure
# ============================================================================

class TestDataFrameStructure:
    """Tests for the returned DataFrame structure."""

    @pytest.mark.unit
    def test_dataframe_has_voltage_column(self, sample_6_params, mock_lumapi_full):
        """Test DataFrame has V_drain column."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert 'V_drain' in df.columns, "DataFrame should have V_drain column"

    @pytest.mark.unit
    def test_dataframe_has_electrical_columns(self, sample_6_params, mock_lumapi_full):
        """Test DataFrame has n and p columns (electrical data)."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert 'n' in df.columns, "DataFrame should have n column"
            assert 'p' in df.columns, "DataFrame should have p column"

    @pytest.mark.unit
    def test_dataframe_has_optical_column(self, sample_6_params, mock_lumapi_full):
        """Test DataFrame has neff column (optical data)."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert 'neff' in df.columns, "DataFrame should have neff column"

    @pytest.mark.unit
    def test_dataframe_columns_aligned(self, sample_6_params, mock_lumapi_full):
        """Test all columns have same length (aligned per voltage step)."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            # All columns should have same length
            assert len(df['V_drain']) == len(df['n'])
            assert len(df['V_drain']) == len(df['p'])
            assert len(df['V_drain']) == len(df['neff'])

    @pytest.mark.unit
    def test_dataframe_has_expected_rows(self, sample_6_params, mock_lumapi_full):
        """Test DataFrame has expected number of rows (voltage steps)."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            # Should have 25 rows (from sample data fixtures)
            assert len(df) == 25, f"Expected 25 rows, got {len(df)}"


# ============================================================================
# Tests for data values
# ============================================================================

class TestDataValues:
    """Tests for the actual data values in returned DataFrame."""

    @pytest.mark.unit
    def test_voltage_values_correct(self, sample_6_params, mock_lumapi_full, sample_charge_data):
        """Test V_drain values match input charge data."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            expected_V = sample_charge_data['V_drain'].flatten()
            np.testing.assert_array_almost_equal(
                df['V_drain'].values, expected_V,
                err_msg="V_drain values should match input"
            )

    @pytest.mark.unit
    def test_electron_values_correct(self, sample_6_params, mock_lumapi_full, sample_charge_data):
        """Test n values match input charge data."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            expected_n = sample_charge_data['n'].flatten()
            np.testing.assert_array_almost_equal(
                df['n'].values, expected_n,
                err_msg="n values should match input"
            )

    @pytest.mark.unit
    def test_hole_values_correct(self, sample_6_params, mock_lumapi_full, sample_charge_data):
        """Test p values match input charge data."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            expected_p = sample_charge_data['p'].flatten()
            np.testing.assert_array_almost_equal(
                df['p'].values, expected_p,
                err_msg="p values should match input"
            )

    @pytest.mark.unit
    def test_neff_values_correct(self, sample_6_params, mock_lumapi_full, sample_optical_data):
        """Test neff values match input optical data."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            expected_neff = sample_optical_data['neff'].flatten()
            np.testing.assert_array_almost_equal(
                df['neff'].values, expected_neff,
                err_msg="neff values should match input"
            )

    @pytest.mark.unit
    def test_neff_is_complex(self, sample_6_params, mock_lumapi_full):
        """Test that neff column contains complex values."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, _ = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert np.iscomplexobj(df['neff'].values), "neff should be complex"


# ============================================================================
# Tests for timing data
# ============================================================================

class TestTimingData:
    """Tests for the timing dictionary."""

    @pytest.mark.unit
    def test_timing_has_charge_time(self, sample_6_params, mock_lumapi_full):
        """Test timing dict has charge_time key."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            _, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert 'charge_time' in timing

    @pytest.mark.unit
    def test_timing_has_fde_time(self, sample_6_params, mock_lumapi_full):
        """Test timing dict has fde_time key."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            _, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert 'fde_time' in timing

    @pytest.mark.unit
    def test_timing_values_are_numeric(self, sample_6_params, mock_lumapi_full):
        """Test timing values are numeric (float)."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            _, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert isinstance(timing['charge_time'], (int, float))
            assert isinstance(timing['fde_time'], (int, float))

    @pytest.mark.unit
    def test_timing_values_non_negative(self, sample_6_params, mock_lumapi_full):
        """Test timing values are non-negative."""
        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            _, timing = sim_handler.run_full_simulation(sample_6_params, sim_id=1)

            assert timing['charge_time'] >= 0
            assert timing['fde_time'] >= 0


# ============================================================================
# Tests with all 6 parameters
# ============================================================================

class TestWith6Parameters:
    """Tests verifying all 6 input parameters are accepted."""

    @pytest.mark.unit
    def test_accepts_all_6_params(self, mock_lumapi_full):
        """Test function accepts all 6 required parameters."""
        params = {
            'w_r': 400e-9,
            'h_si': 100e-9,
            'doping': 5e17,
            'S': 0.4e-6,
            'lambda': 1310e-9,
            'length': 0.5e-3
        }

        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            # Should not raise
            df, timing = sim_handler.run_full_simulation(params, sim_id=1)

            assert df is not None
            assert timing is not None

    @pytest.mark.unit
    def test_different_param_values(self, mock_lumapi_full):
        """Test function works with different parameter values."""
        params = {
            'w_r': 350e-9,       # min value
            'h_si': 130e-9,      # max value
            'doping': 1e18,      # max value
            'S': 0,              # min value
            'lambda': 1260e-9,   # min value
            'length': 1.0e-3     # max value
        }

        with patch.dict('sys.modules', {'lumapi': mock_lumapi_full}):
            import sim_handler
            sim_handler.lumapi = mock_lumapi_full

            df, timing = sim_handler.run_full_simulation(params, sim_id=99)

            assert isinstance(df, pd.DataFrame)
            assert isinstance(timing, dict)
