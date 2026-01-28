# Test Suite Documentation

This directory contains the automated test suite for PS_Opt_V2.

## Structure

```
test/
├── conftest.py              # Shared fixtures and mocks
├── pytest.ini               # Pytest configuration (in project root)
│
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_data_processor.py   # Optical calculations (calc_alpha, calc_dneff, etc.)
│   ├── test_bo.py               # Bayesian optimization (cost function, duplicates)
│   ├── test_lhs.py              # Latin Hypercube Sampling
│   └── test_results_archive.py  # Results archive management
│
├── integration/             # Integration tests (filesystem, mocked Lumerical)
│   └── test_csv_operations.py   # CSV read/write operations
│
├── lumerical/               # Lumerical API tests (requires installation)
│   ├── test_connection.py       # API import, session creation
│   └── test_basic_operations.py # File loading, parameter operations
│
└── tools/                   # Development utilities (not tests)
    └── debug_sweep_structure.py # Debug script for FDE sweep inspection
```

## Running Tests

### Prerequisites

Activate the virtual environment:
```bash
source venv/bin/activate
```

### Run All Tests (excluding Lumerical)

```bash
pytest -m "not lumerical"
```

### Run Only Unit Tests

```bash
pytest -m unit
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Run Lumerical Tests (requires Lumerical installation)

```bash
pytest -m lumerical
```

### Run Specific Test File

```bash
pytest test/unit/test_data_processor.py -v
```

### Run with Coverage Report

```bash
pytest -m "not lumerical" --cov=system --cov-report=term-missing
```

### Run Tests in Parallel (requires pytest-xdist)

```bash
pip install pytest-xdist
pytest -m "not lumerical" -n auto
```

## Test Markers

Tests are tagged with markers for selective execution:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit tests with no external dependencies |
| `@pytest.mark.integration` | Integration tests (may use filesystem) |
| `@pytest.mark.lumerical` | Tests requiring Lumerical installation |
| `@pytest.mark.slow` | Tests that take > 1 second |

## Fixtures

Shared fixtures are defined in `conftest.py`:

### Sample Data Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_params` | Sample parameter dictionary |
| `sample_params_bounds` | Parameter bounds from config |
| `sample_voltage_array` | Voltage array (0 to 2.5V, 25 points) |
| `sample_neff_array` | Complex effective index array |
| `sample_neff_reaches_pi` | neff array that reaches π phase shift |
| `sample_neff_no_pi` | neff array that doesn't reach π |
| `sample_results_df` | Sample results DataFrame |

### Temporary File Fixtures

| Fixture | Description |
|---------|-------------|
| `temp_csv_dir` | Temporary directory for CSV files |
| `temp_results_csv` | Temporary results CSV file |
| `temp_params_csv` | Temporary params CSV file |
| `temp_archive_dir` | Temporary archive directory |

### Mock Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_lumapi` | Mock lumapi module |
| `mock_charge_session` | Mock CHARGE session with sample data |
| `mock_fde_session` | Mock FDE/MODE session with sample data |
| `mock_config` | Override config values for testing |

## Test Descriptions

### Unit Tests

#### `test_data_processor.py` (27 tests)
Tests pure calculation functions:
- `calc_alpha()` - Optical loss calculation (dB/cm)
- `calc_dneff()` - Effective index change
- `calc_dphi()` - Phase shift calculation
- `calculate_v_pi()` - V_π interpolation

#### `test_bo.py` (23 tests)
Tests Bayesian optimization functions:
- `calculate_loss_function()` - Cost function (success/penalty cases)
- `_is_duplicate_params()` - Duplicate parameter detection

#### `test_lhs.py` (12 tests)
Tests Latin Hypercube Sampling:
- Sample generation with correct bounds
- Reproducibility with fixed seed
- Space-filling properties

#### `test_results_archive.py` (17 tests)
Tests results archive management:
- `get_next_sim_id()` - ID sequencing
- `load_all_results_for_bo()` - Loading and merging results
- `archive_current_results()` - Archiving functionality

### Integration Tests

#### `test_csv_operations.py` (9 tests)
Tests CSV file operations:
- Creating and appending to result files
- Error logging to CSV
- Reading params.csv with units row

### Lumerical Tests

#### `test_connection.py`
Tests Lumerical API connectivity:
- lumapi import
- DEVICE/MODE session creation
- Configuration path validation

#### `test_basic_operations.py`
Tests basic Lumerical operations:
- Loading simulation files (.ldev, .lms)
- Getting/setting parameters
- Sweep result access

## Writing New Tests

### Example Unit Test

```python
import pytest
import numpy as np

class TestMyFunction:
    @pytest.mark.unit
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_value)
        assert result == expected_value

    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Using Fixtures

```python
@pytest.mark.unit
def test_with_sample_data(sample_params, sample_voltage_array):
    """Test using shared fixtures."""
    result = process_params(sample_params)
    assert len(result) == len(sample_voltage_array)

@pytest.mark.integration
def test_csv_operation(mock_config, temp_csv_dir):
    """Test with mocked config and temp directory."""
    import config
    save_result(config.RESULTS_CSV_FILE, data)
    assert os.path.exists(config.RESULTS_CSV_FILE)
```

## CI/CD Integration

For automated testing in CI pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    source venv/bin/activate
    pytest -m "not lumerical" --tb=short -q
```

## Troubleshooting

### Tests hang or are slow
- BO tests may be slow on first run due to library imports
- Use `-x` flag to stop on first failure: `pytest -x`

### Import errors
- Ensure virtual environment is activated
- Check that `system/` is in Python path

### Lumerical tests fail
- Verify Lumerical is installed
- Check `LUMERICAL_API_PATH` in `config.py`
- Ensure simulation files exist in `Lumerical_Files/`
