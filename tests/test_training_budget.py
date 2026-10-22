from types import SimpleNamespace
import time

import pytest

from src.training.budget import TrainingBudget
from src.training.common import TrainingConfig


def sampled_budget(**overrides):
    budget = TrainingBudget(TrainingConfig(**overrides))
    budget.process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=500_000_000))
    budget.virtual_memory = lambda: SimpleNamespace(available=4_000_000_000)
    return budget


def test_budget_records_samples_and_does_not_claim_a_hard_memory_limit():
    budget = sampled_budget()
    budget.check()
    assert budget.report()["sampled_peak_process_rss_bytes"] == 500_000_000
    assert budget.report()["sampled_minimum_host_available_bytes"] == 4_000_000_000
    assert "not an OS hard limit" in budget.report()["sampling_scope"]


def test_low_host_memory_aborts_before_more_training():
    budget = sampled_budget()
    budget.virtual_memory = lambda: SimpleNamespace(available=100_000_000)
    with pytest.raises(MemoryError, match="Insufficient host RAM"):
        budget.check()


def test_process_memory_budget_and_total_deadline_are_separate():
    budget = sampled_budget(max_process_rss_gb=.2)
    with pytest.raises(MemoryError, match="process RSS"):
        budget.check()
    budget = sampled_budget(max_total_runtime_seconds=1)
    budget.started = time.monotonic() - 2
    with pytest.raises(TimeoutError, match="including preparation"):
        budget.check()
