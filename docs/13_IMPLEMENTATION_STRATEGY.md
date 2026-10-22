# 13_IMPLEMENTATION_STRATEGY.md — Engineering Architecture, Priorities, and Quality Standards

---

## 1. Core Implementation Principles

To transition FrameGenX from an academic concept into an industrial-strength open-source platform, all engineering decisions adhere to six foundational priorities:

```
+─────────────────────────────────────────────────────────────────────────────+
|                         CORE IMPLEMENTATION PRIORITIES                      |
+─────────────────────────────────────────────────────────────────────────────+
|  1. Scientific Validity       Claims are paid for in measurements. No       |
|                               unmeasured performance assertions.            |
+─────────────────────────────────────────────────────────────────────────────+
|  2. Reproducibility           Fixed seeds, deterministic random states, and  |
|                               pinned dependency checksums across runs.      |
+─────────────────────────────────────────────────────────────────────────────+
|  3. Benchmarkability          Every feature must be benchmarkable via the   |
|                               automated ablation suite before merging.      |
+─────────────────────────────────────────────────────────────────────────────+
|  4. Modularity                Stage-graph decoupled architecture; stages    |
|                               communicate only via in-memory RunContext.    |
+─────────────────────────────────────────────────────────────────────────────+
|  5. Resource Efficiency       Strict 6GB VRAM budget; zero concurrent GPU    |
|                               allocations; aggressive CUDA memory flushes.  |
+─────────────────────────────────────────────────────────────────────────────+
|  6. Open-Source Ergonomics    Single-command setup, intuitive CLI/API, and   |
|                               comprehensive glass-box web observability.    |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Architectural Module Implementation Blueprint

### 2.1 Stage-Graph Decoupling
Every functional module inherits from the abstract base class `Stage` (`src/core/stage.py`):
```python
class Stage(ABC):
    name: str = "base_stage"
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    def enabled(self) -> bool:
        return self.name in self.config.get("pipeline", {}).get("stages", [])
    @abstractmethod
    def run(self, ctx: RunContext) -> None:
        """Mutate ctx in place and emit intermediate artifacts."""
        ...
```
- **Pre-Loop Stages (`intent`, `retrieval`)**: Executed once at run initialization.
- **Loop Stages (`plan`, `adapt`, `generate`, `verify`, `reflect`)**: Executed iteratively up to `max_repair_loops`.

### 2.2 Memory Lifecycle Strategy (6GB VRAM)
```
       STAGE MEMORY LIFECYCLE PROTOCOL
       
  [Stage N Complete]
          │
          ▼
  [Explicit Output Serialization] ──> Save intermediate PNG / JSON to disk
          │
          ▼
  [Module De-referencing]         ──> Release model weights from active scope
          │
          ▼
  [CUDA Garbage Collection]       ──> gc.collect() + torch.cuda.empty_cache()
          │
          ▼
  [VRAM Health Check]             ──> Verify allocated VRAM < 200MB before next Stage
          │
          ▼
  [Stage N+1 Allocation]          ──> Load next model into clean GPU memory
```

---

## 3. Engineering Quality Gates & CI/CD Standards

No pull request or architecture change is merged without satisfying three mandatory gates:

### Gate 1: Automated Unit & Integration Tests (`tests/`)
```powershell
python -m pytest tests/ -v
```
- **Schema Validation**: Tests that `ContractV2` strictly validates valid JSONs and rejects malformed coordinate inputs.
- **Memory Leak Tests**: Asserts that running 5 consecutive pipeline cycles does not increase baseline VRAM allocation by $>50\text{MB}$.
- **Stage Isolation Tests**: Validates that mocking any individual stage allows downstream stages to execute without crashing.

### Gate 2: Benchmark Regression Gate (`benchmark.py`)
```powershell
python benchmark.py --config full --limit 15 --seed 42
```
- **Regression Constraint**: Any PR that decreases the baseline Constraint Satisfaction Rate (CSR) on the balanced test suite is automatically blocked.

### Gate 3: Glass-Box Artifact Integrity
Every generation run must produce a verifiable on-disk audit trail in `runs/<id>/`:
- `contract_v2.json`
- `plan.json`
- `attempt_k/draft.png`
- `attempt_k/dino_detections.json`
- `attempt_k/vqa_results.json`
- `attempt_k/decision.json`
- `receipt.json`

---

## 4. Tradeoffs & Architectural Invariants

| Dimension | Decision | Rationale |
|---|---|---|
| **Latency vs. Verifiability** | We prioritize verifiability over raw speed. | An unverified 3-second generation that requires 60 seconds of human QA is net slower than a 25-second verified generation. |
| **Aesthetics vs. Logic** | We prioritize constraint adherence over aesthetic saturation. | A beautiful image with 4 apples when 3 were requested is a compilation error. |
| **Monolith vs. Stage Graph** | We choose a decoupled stage graph. | Allows independent upgrading of generators (SDXL $\to$ Flux), detectors (DINO $\to$ Florence-2), or NLP parsers without systemic refactoring. |
| **Local vs. Cloud-Only** | We enforce local consumer GPU feasibility. | True open-source longevity requires accessibility on standard developer laptops (6GB VRAM). |
