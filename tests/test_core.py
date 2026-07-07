import pytest
import os
from src.core.context import RunContext
from src.core.db import RunTrackerDB

def test_run_context_initialization(mock_config):
    ctx = RunContext(prompt="Test prompt", config=mock_config)
    assert ctx.prompt == "Test prompt"
    assert ctx.run_id is not None
    assert ctx.is_successful is False
    assert ctx.repair_loops_executed == 0
    
def test_run_context_artifact_saving(mock_config):
    ctx = RunContext(prompt="Test prompt", config=mock_config)
    test_data = {"hello": "world"}
    ctx.save_artifact("test.json", test_data)
    
    file_path = ctx.run_dir / "test.json"
    assert file_path.exists()
    
def test_run_tracker_db(mock_config):
    db_path = mock_config["paths"]["db_path"]
    db = RunTrackerDB(db_path)
    
    ctx = RunContext(prompt="DB Test prompt", config=mock_config)
    ctx.is_successful = True
    ctx.repair_loops_executed = 2
    ctx.contract = {"objects": []}
    
    db.log_run(ctx)
    
    # Retrieve
    retrieved = db.get_run(ctx.run_id)
    assert retrieved is not None
    assert retrieved["prompt"] == "DB Test prompt"
    assert retrieved["is_successful"] is True
    assert retrieved["repair_loops"] == 2
    
    db.close()
    assert os.path.exists(db_path)
