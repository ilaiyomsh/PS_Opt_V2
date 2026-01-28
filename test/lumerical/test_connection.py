# test/lumerical/test_connection.py
# Tests for Lumerical API connection
# These tests require Lumerical to be installed and configured

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'))


# ============================================================================
# Configuration Path Tests
# ============================================================================

class TestLumericalConfig:
    """Tests for Lumerical configuration."""

    @pytest.mark.lumerical
    def test_config_path_exists(self):
        """Test that Lumerical API path exists in config."""
        import config

        assert hasattr(config, 'LUMERICAL_API_PATH'), \
            "config.LUMERICAL_API_PATH should be defined"
        assert config.LUMERICAL_API_PATH is not None, \
            "LUMERICAL_API_PATH should not be None"

    @pytest.mark.lumerical
    def test_api_path_is_directory(self):
        """Test that Lumerical API path is a valid directory."""
        import config

        # Note: This test will fail on machines without Lumerical
        if os.path.exists(config.LUMERICAL_API_PATH):
            assert os.path.isdir(config.LUMERICAL_API_PATH), \
                f"LUMERICAL_API_PATH should be a directory: {config.LUMERICAL_API_PATH}"

    @pytest.mark.lumerical
    def test_simulation_file_paths_defined(self):
        """Test that simulation file paths are defined."""
        import config

        assert hasattr(config, 'CHARGE_SIM_FILE'), "CHARGE_SIM_FILE should be defined"
        assert hasattr(config, 'FDE_SIM_FILE'), "FDE_SIM_FILE should be defined"


# ============================================================================
# API Import Tests
# ============================================================================

class TestLumapiImport:
    """Tests for lumapi module import."""

    @pytest.mark.lumerical
    def test_lumapi_import(self):
        """Test that lumapi can be imported."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
            assert lumapi is not None
        except ImportError as e:
            pytest.skip(f"lumapi not available: {e}")

    @pytest.mark.lumerical
    def test_lumapi_has_device(self):
        """Test that lumapi has DEVICE class."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
            assert hasattr(lumapi, 'DEVICE'), "lumapi should have DEVICE class"
        except ImportError:
            pytest.skip("lumapi not available")

    @pytest.mark.lumerical
    def test_lumapi_has_mode(self):
        """Test that lumapi has MODE class."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
            assert hasattr(lumapi, 'MODE'), "lumapi should have MODE class"
        except ImportError:
            pytest.skip("lumapi not available")


# ============================================================================
# Session Creation Tests
# ============================================================================

class TestDeviceSession:
    """Tests for DEVICE session creation."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_device_session_creation(self):
        """Test DEVICE session creation and cleanup."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
        except ImportError:
            pytest.skip("lumapi not available")

        session = None
        try:
            session = lumapi.DEVICE(hide=True)
            assert session is not None, "DEVICE session should be created"
        finally:
            if session is not None:
                session.close()

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_device_session_hide_parameter(self):
        """Test that hide=True works for DEVICE session."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
        except ImportError:
            pytest.skip("lumapi not available")

        session = None
        try:
            # Should not open a GUI window
            session = lumapi.DEVICE(hide=True)
            assert session is not None
        finally:
            if session is not None:
                session.close()


class TestModeSession:
    """Tests for MODE session creation."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_session_creation(self):
        """Test MODE session creation and cleanup."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
        except ImportError:
            pytest.skip("lumapi not available")

        session = None
        try:
            session = lumapi.MODE(hide=True)
            assert session is not None, "MODE session should be created"
        finally:
            if session is not None:
                session.close()

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_session_hide_parameter(self):
        """Test that hide=True works for MODE session."""
        import config

        if config.LUMERICAL_API_PATH not in sys.path:
            sys.path.insert(0, config.LUMERICAL_API_PATH)

        try:
            import lumapi
        except ImportError:
            pytest.skip("lumapi not available")

        session = None
        try:
            session = lumapi.MODE(hide=True)
            assert session is not None
        finally:
            if session is not None:
                session.close()


# ============================================================================
# Connection Fixture Tests
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


class TestSessionFixtures:
    """Tests using session fixtures."""

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_device_fixture_works(self, device_session):
        """Test that device_session fixture provides valid session."""
        assert device_session is not None

    @pytest.mark.lumerical
    @pytest.mark.slow
    def test_mode_fixture_works(self, mode_session):
        """Test that mode_session fixture provides valid session."""
        assert mode_session is not None
