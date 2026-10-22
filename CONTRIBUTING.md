# Contributing to FrameGenX

Thank you for your interest in contributing to FrameGenX! This guide will help you get oriented.

## Project Philosophy

FrameGenX is a **verification-first** system. Before contributing:

1. Read [`docs/architecture.md`](docs/architecture.md) for the stage graph and VRAM lifecycle
2. Read [`docs/research.md`](docs/research.md) for the research hypothesis and measurement methodology
3. Read [`AGENTS.md`](.agents/AGENTS.md) for the core engineering principles

### The Four Questions

Before proposing any feature, ask:

1. **Can it be measured?** → Every feature must have a CSR impact assessment
2. **Can it be benchmarked?** → It must be testable via `benchmark.py`
3. **Can it be reproduced?** → Fixed seeds, logged artifacts, deterministic outputs
4. **Can it run on 6GB VRAM?** → If not, reject the proposal

## Development Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (RTX 3050 or equivalent)
- CUDA 11.8+ and cuDNN

### Installation
```bash
git clone https://github.com/vaishak-v-nair/FrameGenX.git
cd FrameGenX
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

pip install -e ".[dev]"
```

### Running Tests
```bash
python -m pytest tests/ -v --tb=short
```

All tests must pass before submitting a PR. Current test coverage:
- `test_core.py` — RunContext and DB tests
- `test_stage_graph.py` — Pipeline, verification, and composition tests
- `test_intent_extraction.py` — Contract extraction accuracy tests
- `test_retrieval.py` — BM25, layout blueprint, and style preset tests

### Running the Benchmark
```bash
python benchmark.py --config full --seed 42
python benchmark.py --config baseline --seed 42
python scripts/benchmark_report.py
```

## How to Add a New Stage

1. Create your stage in the appropriate `src/` subdirectory
2. Inherit from `Stage` and decorate with `@register_stage`
3. Implement the `run(ctx: RunContext)` method
4. Add it to `config.yaml` pipeline stages list
5. Add tests in `tests/`

```python
from src.core.stage import Stage, register_stage
from src.core.context import RunContext

@register_stage
class MyStage(Stage):
    name = "my_stage"

    def run(self, ctx: RunContext) -> None:
        # Your logic here
        # Access contract: ctx.contract
        # Save artifacts: ctx.save_artifact(name, data)
        pass
```

## How to Add a New Verifier

1. Create your verifier in `src/verification/`
2. Follow the CPU-resident pattern (models load to CPU, execute on GPU, flush after)
3. Add config toggle in `config.yaml` under `verification:`
4. Wire it into `verify_stage.py`
5. Test with the ablation benchmark

### VRAM Discipline Rules

- **Never** load two GPU-heavy models simultaneously
- **Always** call `free_memory()` after GPU work
- **Always** use fp16 precision for vision models
- Document VRAM usage in your module docstring
- Test on 6GB hardware before declaring "it works"

## How to Extend the Benchmark

1. Add prompts to `benchmark_dataset.json` with a `category` field
2. Ensure balanced category coverage (use `scripts/curate_subset.py`)
3. Run all 4 ablation configs
4. Report per-category CSR breakdown
5. Update `results.md` with `scripts/make_results.py`

### Benchmark Categories
- `object_count` — Numerical accuracy (e.g., "three apples")
- `color_constraints` — Attribute binding (e.g., "red car, blue house")
- `spatial_relations` — Positional accuracy (e.g., "beside", "on top of")
- `attribute_binding` — Multi-attribute objects
- `text_rendering` — Text in generated images
- `multi_object_scenes` — Complex compositions (>3 objects)

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for all public functions
- Preserve existing comments and docstrings
- Use `ruff` for linting: `ruff check src/ tests/`

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New code has test coverage
- [ ] VRAM usage documented
- [ ] Benchmark impact assessed (if applicable)
- [ ] No new `DeprecationWarning`s introduced
- [ ] Docstrings added for public APIs
