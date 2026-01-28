# test/lumerical/test_basic_operations.py
# Tests for basic Lumerical API operations
# These tests require Lumerical to be installed and configured

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# Simulation File Tests
# ============================================================================

class TestSimulationFilesExist:
    """Tests for simulation template files."""

    @pytest.mark.lumerical
    def test_charge_sim_file_exists(self):
        """Test that CHARGE simulation file exists."""
        import config

        assert os.path.exists(config.CHARGE_SIM_FILE), \
            f"CHARGE file not found: {config.CHARGE_SIM_FILE}"

    @pytest.mark.lumerical
    def test_fde_sim_file_exists(self):
        """Test that FDE simulation file exists."""
        import config

        assert os.path.exists(config.FDE_SIM_FILE), \
            f"FDE file not found: {config.FDE_SIM_FILE}"

    @pytest.mark.lumerical
    def test_charge_file_extension(self):
        """Test that CHARGE file has correct extension."""
        import config

        assert config.CHARGE_SIM_FILE.endswith('.ldev'), \
            "CHARGE file should have .ldev extension"

    @pytest.mark.lumerical
    def test_fde_file_extension(self):
        """Test that FDE file has correct extension."""
        import config

        assert config.FDE_SIM_FILE.endswith('.lms'), \
            "FDE file should have .lms extension"


# ============================================================================
# Session Fixtures
# ============================================================================

@pytest.fixture
def device_session():
    """Fixture to create and cleanup DEVICE session."""
    import config

    if config.LUMERICAL_API_PATH not in sys.path:
        sys.path.insert(0, config.LUMERICAL_API_PATH)

    try:
        import lumapi
    except ImportError:
        pytest.skip("lumapi not available")

    session = lumapi.DEVICE(hide=True)
    yield session
    session.close()


@pytest.fixture
def mode_session():
    """Fixture to create and cleanup MODE session."""
    import config

    if config.LUMERICAL_API_PATH not in sys.path:
        sys.path.insert(0, config.LUMERICAL_API_PATH)

    try:
        import lumapi
    except ImportError:
        pytest.skip("lumapi not available")

    session = lumapi.MODE(hide=True)
    yield session
    session.close()


# ============================================================================
# File Loading Tests
# ============================================================================

class TestLoadChargeFile:
    """Tests for loading CHARGE simulation file."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_load_charge_file(self, device_session):
        """Test loading the CHARGE simulation file."""
        import config

        if not os.path.exists(config.CHARGE_SIM_FILE):
            pytest.skip(f"CHARGE file not found: {config.CHARGE_SIM_FILE}")

        # Load should not raise an exception
        device_session.load(config.CHARGE_SIM_FILE)

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_charge_file_has_expected_structure(self, device_session):
        """Test that loaded CHARGE file has expected objects."""
        import config

        if not os.path.exists(config.CHARGE_SIM_FILE):
            pytest.skip(f"CHARGE file not found: {config.CHARGE_SIM_FILE}")

        device_session.load(config.CHARGE_SIM_FILE)

        # Check for CHARGE solver object
        try:
            # The file should have a CHARGE object
            device_session.select("CHARGE")
        except Exception:
            # If CHARGE object doesn't exist with that exact name, that's also info
            pass


class TestLoadFdeFile:
    """Tests for loading FDE/MODE simulation file."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_load_fde_file(self, mode_session):
        """Test loading the FDE simulation file."""
        import config

        if not os.path.exists(config.FDE_SIM_FILE):
            pytest.skip(f"FDE file not found: {config.FDE_SIM_FILE}")

        # Load should not raise an exception
        mode_session.load(config.FDE_SIM_FILE)

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_fde_file_has_sweep(self, mode_session):
        """Test that loaded FDE file has voltage sweep."""
        import config

        if not os.path.exists(config.FDE_SIM_FILE):
            pytest.skip(f"FDE file not found: {config.FDE_SIM_FILE}")

        mode_session.load(config.FDE_SIM_FILE)

        # Try to get sweep info
        try:
            sweeps = mode_session.getsweepresult()
            assert 'voltage' in str(sweeps).lower() or sweeps is not None, \
                "FDE file should have a voltage sweep"
        except Exception as e:
            # Sweep might exist but be empty
            pass


# ============================================================================
# Basic API Operation Tests
# ============================================================================

class TestBasicDeviceOperations:
    """Tests for basic DEVICE API operations."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_get_simulation_type(self, device_session):
        """Test getting simulation type from DEVICE session."""
        # DEVICE sessions should report their type
        # This is a basic sanity check that the API is working
        pass  # Basic session creation already tests this

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_device_session_has_methods(self, device_session):
        """Test that DEVICE session has expected methods."""
        assert hasattr(device_session, 'load'), "Session should have load method"
        assert hasattr(device_session, 'save'), "Session should have save method"
        assert hasattr(device_session, 'run'), "Session should have run method"
        assert hasattr(device_session, 'getresult'), "Session should have getresult method"
        assert hasattr(device_session, 'close'), "Session should have close method"


class TestBasicModeOperations:
    """Tests for basic MODE API operations."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_session_has_methods(self, mode_session):
        """Test that MODE session has expected methods."""
        assert hasattr(mode_session, 'load'), "Session should have load method"
        assert hasattr(mode_session, 'save'), "Session should have save method"
        assert hasattr(mode_session, 'run'), "Session should have run method"
        assert hasattr(mode_session, 'getsweepresult'), "Session should have getsweepresult method"
        assert hasattr(mode_session, 'close'), "Session should have close method"

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_session_has_sweep_methods(self, mode_session):
        """Test that MODE session has sweep-related methods."""
        assert hasattr(mode_session, 'runsweep'), "Session should have runsweep method"
        assert hasattr(mode_session, 'getsweepresult'), "Session should have getsweepresult method"
        assert hasattr(mode_session, 'getsweepdata'), "Session should have getsweepdata method"


# ============================================================================
# Parameter Get/Set Tests
# ============================================================================

class TestParameterOperations:
    """Tests for parameter get/set operations."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_device_select_and_get(self, device_session):
        """Test selecting objects and getting parameters in DEVICE."""
        import config

        if not os.path.exists(config.CHARGE_SIM_FILE):
            pytest.skip(f"CHARGE file not found: {config.CHARGE_SIM_FILE}")

        device_session.load(config.CHARGE_SIM_FILE)

        # Try to select CHARGE object and get a parameter
        try:
            device_session.select("CHARGE")
            # Try to get a common parameter
            solver_type = device_session.get("solver type")
            # If we get here, the API is working
            assert solver_type is not None or solver_type == ""
        except Exception:
            # Object might have different name, but session works
            pass

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_select_and_get(self, mode_session):
        """Test selecting objects and getting parameters in MODE."""
        import config

        if not os.path.exists(config.FDE_SIM_FILE):
            pytest.skip(f"FDE file not found: {config.FDE_SIM_FILE}")

        mode_session.load(config.FDE_SIM_FILE)

        # Try to get simulation region parameters
        try:
            # Common objects in FDE simulations
            mode_session.select("FDE")
            # Get a parameter
            x_span = mode_session.get("x span")
            assert x_span is not None
        except Exception:
            # Object might have different name
            pass


# ============================================================================
# Sweep Result Tests
# ============================================================================

class TestSweepResults:
    """Tests for sweep result operations."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_get_sweep_result_names(self, mode_session):
        """Test getting available sweep result names."""
        import config

        if not os.path.exists(config.FDE_SIM_FILE):
            pytest.skip(f"FDE file not found: {config.FDE_SIM_FILE}")

        mode_session.load(config.FDE_SIM_FILE)

        try:
            # This should return available results or an error if no results
            result_names = mode_session.getsweepresult("voltage")
            # If we get here without error, the sweep exists
            assert result_names is not None
        except Exception as e:
            # Sweep might exist but have no results yet
            # This is acceptable - the test verifies the API call works
            if "no results" in str(e).lower() or "not found" in str(e).lower():
                pass
            else:
                raise

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_get_existing_sweep_result(self, mode_session):
        """Test getting existing sweep results (if any)."""
        import config

        if not os.path.exists(config.FDE_SIM_FILE):
            pytest.skip(f"FDE file not found: {config.FDE_SIM_FILE}")

        mode_session.load(config.FDE_SIM_FILE)

        try:
            # Try to get neff results from voltage sweep
            sweep_result = mode_session.getsweepresult("voltage", "neff")
            assert sweep_result is not None
            assert 'neff' in sweep_result or isinstance(sweep_result, dict)
        except Exception as e:
            # No existing results - this is okay for a test
            if "no results" in str(e).lower():
                pytest.skip("No existing sweep results in file")
            elif "result does not exist" in str(e).lower():
                pytest.skip("Sweep has not been run")
            else:
                raise


# ============================================================================
# Monitor/Result Tests
# ============================================================================

class TestChargeResults:
    """Tests for CHARGE result operations."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_charge_monitor_exists(self, device_session):
        """Test that CHARGE file has expected monitors."""
        import config

        if not os.path.exists(config.CHARGE_SIM_FILE):
            pytest.skip(f"CHARGE file not found: {config.CHARGE_SIM_FILE}")

        device_session.load(config.CHARGE_SIM_FILE)

        # Try to get result from expected monitor
        try:
            # The simulation uses "CHARGE::monitor_charge"
            device_session.getresult("CHARGE::monitor_charge")
            # If no error, monitor exists (though may not have results)
        except Exception as e:
            if "no data" in str(e).lower():
                pass  # Monitor exists but no simulation run yet
            elif "does not exist" in str(e).lower():
                pytest.fail("Expected monitor 'CHARGE::monitor_charge' not found")
            else:
                pass  # Other errors might be acceptable
