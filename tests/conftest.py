import pytest
from pathlib import Path

@pytest.fixture
def test_output_dir(tmp_path):
    """Provides a temporary directory for test outputs."""
    return tmp_path

@pytest.fixture
def mock_config(test_output_dir):
    return {
        "paths": {
            "output_dir": str(test_output_dir / "runs"),
            "db_path": str(test_output_dir / "runs" / "framegenx.db")
        }
    }
