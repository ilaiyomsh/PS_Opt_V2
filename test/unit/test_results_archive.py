# test/unit/test_results_archive.py
# Unit tests for results_archive.py

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# get_next_sim_id tests
# ============================================================================

class TestGetNextSimId:
    """Tests for get_next_sim_id() function."""

    @pytest.mark.unit
    def test_empty_results_returns_one(self, mock_config):
        """Test that empty results returns sim_id=1."""
        from results_archive import get_next_sim_id

        next_id = get_next_sim_id()

        assert next_id == 1, "Empty results should return sim_id=1"

    @pytest.mark.unit
    def test_increments_from_existing(self, mock_config, temp_csv_dir, sample_results_df):
        """Test that next sim_id increments from existing max."""
        from results_archive import get_next_sim_id
        import config

        # Save results with sim_ids 1, 2, 3
        sample_results_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        next_id = get_next_sim_id()

        assert next_id == 4, "Should return max(sim_id) + 1"

    @pytest.mark.unit
    def test_handles_gaps(self, mock_config, temp_csv_dir):
        """Test that handles gaps in sim_ids correctly."""
        from results_archive import get_next_sim_id
        import config

        # Create results with gap (1, 2, 10)
        df = pd.DataFrame([
            {'sim_id': 1, 'w_r': 400e-9},
            {'sim_id': 2, 'w_r': 410e-9},
            {'sim_id': 10, 'w_r': 420e-9},
        ])
        df.to_csv(config.RESULTS_CSV_FILE, index=False)

        next_id = get_next_sim_id()

        assert next_id == 11, "Should return max(sim_id) + 1 regardless of gaps"


# ============================================================================
# load_all_results_for_bo tests
# ============================================================================

class TestLoadAllResultsForBo:
    """Tests for load_all_results_for_bo() function."""

    @pytest.mark.unit
    def test_loads_current_results(self, mock_config, temp_csv_dir, sample_results_df):
        """Test loading from current result.csv."""
        from results_archive import load_all_results_for_bo
        import config

        sample_results_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        df = load_all_results_for_bo()

        assert len(df) == len(sample_results_df)
        assert set(df['sim_id']) == set(sample_results_df['sim_id'])

    @pytest.mark.unit
    def test_loads_archived_results(self, mock_config, temp_archive_dir):
        """Test loading from archive directory."""
        from results_archive import load_all_results_for_bo
        import config

        # Create an archived file
        archived_df = pd.DataFrame([
            {'sim_id': 100, 'w_r': 400e-9, 'v_pi_l_Vmm': 0.8},
            {'sim_id': 101, 'w_r': 410e-9, 'v_pi_l_Vmm': 0.9},
        ])
        archive_path = os.path.join(config.RESULTS_ARCHIVE_DIR, "result_archived.csv")
        archived_df.to_csv(archive_path, index=False)

        df = load_all_results_for_bo()

        assert 100 in df['sim_id'].values
        assert 101 in df['sim_id'].values

    @pytest.mark.unit
    def test_merges_current_and_archived(self, mock_config, temp_csv_dir, temp_archive_dir):
        """Test merging current and archived results."""
        from results_archive import load_all_results_for_bo
        import config

        # Current results
        current_df = pd.DataFrame([
            {'sim_id': 1, 'w_r': 400e-9, 'v_pi_l_Vmm': 0.8},
        ])
        current_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        # Archived results
        archived_df = pd.DataFrame([
            {'sim_id': 50, 'w_r': 410e-9, 'v_pi_l_Vmm': 0.9},
        ])
        archive_path = os.path.join(config.RESULTS_ARCHIVE_DIR, "result_old.csv")
        archived_df.to_csv(archive_path, index=False)

        df = load_all_results_for_bo()

        assert len(df) == 2
        assert 1 in df['sim_id'].values
        assert 50 in df['sim_id'].values

    @pytest.mark.unit
    def test_deduplicates_by_sim_id(self, mock_config, temp_csv_dir, temp_archive_dir):
        """Test that duplicate sim_ids are removed (keeps last in concat order)."""
        from results_archive import load_all_results_for_bo
        import config

        # Current results with sim_id=1
        current_df = pd.DataFrame([
            {'sim_id': 1, 'w_r': 400e-9, 'version': 'current'},
        ])
        current_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        # Archived results also with sim_id=1
        archived_df = pd.DataFrame([
            {'sim_id': 1, 'w_r': 380e-9, 'version': 'archived'},
        ])
        archive_path = os.path.join(config.RESULTS_ARCHIVE_DIR, "result_old.csv")
        archived_df.to_csv(archive_path, index=False)

        df = load_all_results_for_bo()

        assert len(df) == 1, "Duplicates should be removed"
        # Keeps 'last' from concat order: current is loaded first, archived second
        # So archived version is kept (it's last in the concatenated order)
        assert df.iloc[0]['version'] == 'archived'

    @pytest.mark.unit
    def test_empty_when_no_results(self, mock_config):
        """Test returns empty DataFrame when no results exist."""
        from results_archive import load_all_results_for_bo

        df = load_all_results_for_bo()

        assert df.empty or len(df) == 0


# ============================================================================
# load_all_archived_results tests
# ============================================================================

class TestLoadAllArchivedResults:
    """Tests for load_all_archived_results() function."""

    @pytest.mark.unit
    def test_loads_multiple_files(self, mock_config, temp_archive_dir):
        """Test loading multiple archive files."""
        from results_archive import load_all_archived_results
        import config

        # Create multiple archive files
        for i in range(3):
            df = pd.DataFrame([
                {'sim_id': i * 10, 'w_r': 400e-9 + i * 10e-9}
            ])
            path = os.path.join(config.RESULTS_ARCHIVE_DIR, f"result_{i}.csv")
            df.to_csv(path, index=False)

        combined = load_all_archived_results()

        assert len(combined) == 3
        assert 0 in combined['sim_id'].values
        assert 10 in combined['sim_id'].values
        assert 20 in combined['sim_id'].values

    @pytest.mark.unit
    def test_ignores_non_csv_files(self, mock_config, temp_archive_dir):
        """Test that non-CSV files are ignored."""
        from results_archive import load_all_archived_results
        import config

        # Create CSV file
        df = pd.DataFrame([{'sim_id': 1, 'w_r': 400e-9}])
        df.to_csv(os.path.join(config.RESULTS_ARCHIVE_DIR, "result.csv"), index=False)

        # Create non-CSV file
        with open(os.path.join(config.RESULTS_ARCHIVE_DIR, "notes.txt"), 'w') as f:
            f.write("Some notes")

        combined = load_all_archived_results()

        assert len(combined) == 1

    @pytest.mark.unit
    def test_empty_archive_dir(self, mock_config, temp_archive_dir):
        """Test with empty archive directory."""
        from results_archive import load_all_archived_results

        combined = load_all_archived_results()

        assert combined.empty


# ============================================================================
# ensure_archive_dir tests
# ============================================================================

class TestEnsureArchiveDir:
    """Tests for ensure_archive_dir() function."""

    @pytest.mark.unit
    def test_creates_directory(self, mock_config, tmp_path, monkeypatch):
        """Test that archive directory is created if missing."""
        from results_archive import ensure_archive_dir
        import config

        new_archive_dir = str(tmp_path / "new_archive")
        monkeypatch.setattr(config, 'RESULTS_ARCHIVE_DIR', new_archive_dir)

        ensure_archive_dir()

        assert os.path.exists(new_archive_dir)

    @pytest.mark.unit
    def test_no_error_if_exists(self, mock_config, temp_archive_dir):
        """Test no error if directory already exists."""
        from results_archive import ensure_archive_dir

        # Directory already exists from fixture
        ensure_archive_dir()  # Should not raise


# ============================================================================
# archive_current_results tests
# ============================================================================

class TestArchiveCurrentResults:
    """Tests for archive_current_results() function."""

    @pytest.mark.unit
    def test_creates_archive_file(self, mock_config, temp_csv_dir, temp_archive_dir, sample_results_df):
        """Test that archive file is created."""
        from results_archive import archive_current_results
        import config

        sample_results_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        archive_path = archive_current_results()

        assert archive_path is not None
        assert os.path.exists(archive_path)

    @pytest.mark.unit
    def test_archive_contains_data(self, mock_config, temp_csv_dir, temp_archive_dir, sample_results_df):
        """Test that archived file contains the data."""
        from results_archive import archive_current_results
        import config

        sample_results_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        archive_path = archive_current_results()
        archived_df = pd.read_csv(archive_path)

        assert len(archived_df) == len(sample_results_df)

    @pytest.mark.unit
    def test_returns_none_if_no_results(self, mock_config):
        """Test returns None if no result.csv exists."""
        from results_archive import archive_current_results

        result = archive_current_results()

        assert result is None

    @pytest.mark.unit
    def test_description_in_filename(self, mock_config, temp_csv_dir, temp_archive_dir, sample_results_df):
        """Test that description is included in filename."""
        from results_archive import archive_current_results
        import config

        sample_results_df.to_csv(config.RESULTS_CSV_FILE, index=False)

        archive_path = archive_current_results(description="test_run")

        assert "test_run" in archive_path
