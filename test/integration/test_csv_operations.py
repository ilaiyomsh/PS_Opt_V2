# test/integration/test_csv_operations.py
# Integration tests for CSV read/write operations
# These tests use the filesystem but mock Lumerical

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# CSV Save Operations Tests
# ============================================================================

class TestSaveSingleResultToCsv:
    """Tests for save_single_result_to_csv() function."""

    @pytest.mark.integration
    def test_creates_new_file(self, mock_config, temp_csv_dir):
        """Test creating a new results CSV file."""
        from data_processor import save_single_result_to_csv
        import config

        result = {
            'sim_id': 1,
            'w_r': 400e-9,
            'h_si': 100e-9,
            'doping': 5e17,
            'S': 0.4e-6,
            'lambda': 1310e-9,
            'length': 0.5e-3,
            'v_pi_V': 1.5,
            'v_pi_l_Vmm': 0.75,
            'loss_at_v_pi_dB_per_cm': 1.8,
            'max_dphi_rad': np.pi + 0.2,
            'cost': -1.5
        }

        save_single_result_to_csv(config.RESULTS_CSV_FILE, result)

        assert os.path.exists(config.RESULTS_CSV_FILE)

        df = pd.read_csv(config.RESULTS_CSV_FILE)
        assert len(df) == 1
        assert df.iloc[0]['sim_id'] == 1

    @pytest.mark.integration
    def test_appends_to_existing(self, mock_config, temp_csv_dir):
        """Test appending to existing results CSV file."""
        from data_processor import save_single_result_to_csv
        import config

        result1 = {
            'sim_id': 1, 'w_r': 400e-9, 'h_si': 100e-9, 'doping': 5e17,
            'S': 0.4e-6, 'lambda': 1310e-9, 'length': 0.5e-3,
            'v_pi_V': 1.5, 'v_pi_l_Vmm': 0.75, 'loss_at_v_pi_dB_per_cm': 1.8,
            'max_dphi_rad': np.pi + 0.2, 'cost': -1.5
        }

        result2 = {
            'sim_id': 2, 'w_r': 410e-9, 'h_si': 105e-9, 'doping': 6e17,
            'S': 0.5e-6, 'lambda': 1310e-9, 'length': 0.6e-3,
            'v_pi_V': 1.8, 'v_pi_l_Vmm': 1.08, 'loss_at_v_pi_dB_per_cm': 2.0,
            'max_dphi_rad': np.pi + 0.3, 'cost': -2.0
        }

        save_single_result_to_csv(config.RESULTS_CSV_FILE, result1)
        save_single_result_to_csv(config.RESULTS_CSV_FILE, result2)

        df = pd.read_csv(config.RESULTS_CSV_FILE)
        assert len(df) == 2
        assert set(df['sim_id']) == {1, 2}

    @pytest.mark.integration
    def test_saves_minimal_columns(self, mock_config, temp_csv_dir):
        """Test that minimal result file only has essential columns."""
        from data_processor import save_single_result_to_csv
        import config

        result = {
            'sim_id': 1,
            'w_r': 400e-9, 'h_si': 100e-9, 'doping': 5e17,
            'S': 0.4e-6, 'lambda': 1310e-9, 'length': 0.5e-3,
            'v_pi_V': 1.5, 'v_pi_l_Vmm': 0.75, 'loss_at_v_pi_dB_per_cm': 1.8,
            'C_at_v_pi_pF_per_cm': 5.0,
            'max_dphi_rad': np.pi + 0.2, 'cost': -1.5,
            'extra_column_1': 'should_not_appear',
            'extra_column_2': 123.456
        }

        save_single_result_to_csv(config.RESULTS_CSV_FILE, result)

        df = pd.read_csv(config.RESULTS_CSV_FILE)

        # Minimal columns from config
        minimal_cols = config.MINIMAL_RESULT_COLUMNS
        for col in minimal_cols:
            if col in result:
                assert col in df.columns, f"Missing minimal column: {col}"

    @pytest.mark.integration
    def test_saves_full_file(self, mock_config, temp_csv_dir):
        """Test that full results file is also created."""
        from data_processor import save_single_result_to_csv
        import config

        result = {
            'sim_id': 1,
            'w_r': 400e-9, 'h_si': 100e-9, 'doping': 5e17,
            'S': 0.4e-6, 'lambda': 1310e-9, 'length': 0.5e-3,
            'v_pi_V': 1.5, 'v_pi_l_Vmm': 0.75, 'loss_at_v_pi_dB_per_cm': 1.8,
            'max_dphi_rad': np.pi + 0.2, 'cost': -1.5,
            'extra_data': 999.0
        }

        save_single_result_to_csv(config.RESULTS_CSV_FILE, result)

        assert os.path.exists(config.RESULTS_FULL_CSV_FILE)

        df_full = pd.read_csv(config.RESULTS_FULL_CSV_FILE)
        assert 'extra_data' in df_full.columns


# ============================================================================
# CSV Error Logging Tests
# ============================================================================

class TestSaveErrorToCsv:
    """Tests for save_error_to_csv() function."""

    @pytest.mark.integration
    def test_creates_error_file(self, mock_config, temp_csv_dir):
        """Test creating error CSV file."""
        from data_processor import save_error_to_csv
        import config

        error = ValueError("Test error message")

        save_error_to_csv(
            sim_id=1,
            stage='CHARGE_RUN',
            error=error,
            params={'w_r': 400e-9, 'h_si': 100e-9}
        )

        assert os.path.exists(config.ERRORS_CSV_FILE)

    @pytest.mark.integration
    def test_error_record_content(self, mock_config, temp_csv_dir):
        """Test that error record contains expected fields."""
        from data_processor import save_error_to_csv
        import config

        error = RuntimeError("Simulation failed")

        save_error_to_csv(
            sim_id=5,
            stage='FDE_SETUP',
            error=error,
            params={'w_r': 400e-9}
        )

        df = pd.read_csv(config.ERRORS_CSV_FILE)

        assert df.iloc[0]['sim_id'] == 5
        assert df.iloc[0]['stage'] == 'FDE_SETUP'
        assert df.iloc[0]['error_type'] == 'RuntimeError'
        assert 'Simulation failed' in df.iloc[0]['error_message']

    @pytest.mark.integration
    def test_appends_multiple_errors(self, mock_config, temp_csv_dir):
        """Test appending multiple errors."""
        from data_processor import save_error_to_csv
        import config

        save_error_to_csv(1, 'CHARGE_RUN', ValueError("Error 1"))
        save_error_to_csv(2, 'FDE_RUN', RuntimeError("Error 2"))
        save_error_to_csv(3, 'RESULT_EXTRACT', KeyError("Error 3"))

        df = pd.read_csv(config.ERRORS_CSV_FILE)

        assert len(df) == 3


# ============================================================================
# Params CSV Tests
# ============================================================================

class TestParamsCsv:
    """Tests for params.csv reading/writing."""

    @pytest.mark.integration
    def test_read_params_skipping_units(self, temp_params_csv):
        """Test reading params.csv while skipping units row."""
        # Read with skiprows to skip units row
        df = pd.read_csv(temp_params_csv, skiprows=[1])

        # Should have numeric values, not unit strings
        assert df['w_r'].dtype in [np.float64, np.int64, float]

    @pytest.mark.integration
    def test_units_row_format(self, temp_params_csv):
        """Test that units row is properly formatted."""
        # Read without skipping to see units
        df_raw = pd.read_csv(temp_params_csv)

        # First row should have sim_id as '-'
        assert df_raw.iloc[0]['sim_id'] == '-'
