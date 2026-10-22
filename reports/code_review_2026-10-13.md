# FrameGenX code quality review — 2026-10-13

## Verdict

Follow-up implementation and live evidence are recorded in
`reports/research_image_generation_2026-10-13.md` (updated 2026-10-14).
It adds per-requirement coverage, conservative duplicate suppression,
fingerprinted benchmark suites, disabled default refinement, immutable
planning, provider and offline-cache fixes, valid previews, and owned Milvus
connections. The remaining-risk table below is this earlier review's snapshot.

FrameGenX has a functioning modular generation pipeline and useful recorded
evidence, but it is not yet justified to call the code or the product
"perfect." The default acceptance decision currently proves only the
implemented hard checks: counted object detections and supported spatial
relationships. It does not prove every constraint written into the contract.
See `reports/qa_2026-10-13.md` for the earlier integrated GPU/API assessment.

## Current system map

| Layer | Current implementation |
| --- | --- |
| User interface | React/Vite studio, run history, and ablation dashboard in `frontend/src/` |
| HTTP service | FastAPI in `src/api.py`; one in-process GPU job at a time, polling, previews, history and stats |
| Orchestration | `src/core/stage.py` owns intent/retrieval once and plan/adapt/generate/verify/reflect retries |
| Contract | LLM extraction plus Pydantic fields in `src/intent/`; spatial relationships and negations are appended after model validation |
| Retrieval | Local Milvus/caption index, layout/style data, optional web fallback and failure memory |
| Generation | SDXL-Turbo, optional LoRA and regional attention; current default skips ControlNet; optional post-acceptance img2img |
| Verification | GroundingDINO count and box geometry are hard checks; CLIP and external vision QA are diagnostics |
| Persistence | Per-run files under `runs/` and SQLite run/benchmark metadata |

Primary deployment target is the local RTX 3050 Laptop GPU with 6GB VRAM and
16GB RAM. The API and Docker Compose bind to loopback by default. A public
deployment would need an authentication and access-control design.

## Confirmed defects fixed in this review

1. **Missing verification could produce success.** An empty stage list had
   returned `accepted=True` without a contract or verification report.
   `Pipeline.run` now requires generation and verification stages and requires
   a fresh hard-failure report for every attempt. Direct `ReflectStage` calls
   also fail closed on a missing report.
2. **Spatial predicates could pass without being checked.** A saved accepted
   run recorded `predicate: on`, `passed: true`, with reason `unknown predicate
   'on': skipped`. Unsupported predicates and the under-specified binary
   `between` relation now fail. Supported `on` variants are normalized and
   checked for contact. The same saved boxes now fail the contact check.
3. **Detector noun matching could count unrelated classes.** `cat` previously
   matched `caterpillar` by substring in both count and spatial matching.
   Whole-word phrase matching now prevents this; a real offline GroundingDINO
   replay still detected one cat and one chair in the saved image.
4. **Analysis reports invented diagnoses.** Missing signal names defaulted to
   zero, so an accepted run with no failures acquired three negative "root
   causes." Reports now leave cause inference empty and derive review actions
   from recorded failure types only.
5. **Benchmark errors contaminated pass rates and resume.** Provider/model
   failures were stored as rejected images, included in the denominator, and
   skipped on resume. Error rows are now retryable and excluded from scored
   rows; the CLI and launchers report incomplete execution. Benchmark LoRA
   variants explicitly enable one configured adapter and reject scoring if
   activation fails. Their current live LoRA performance remains unmeasured.
6. **Cache reuse crossed code/config changes.** The API could serve an old
   accepted image after a verifier change. New interactive rows store a
   fingerprint of active configuration, backend source, and dependency pins;
   the API reuses only matching rows. An old accepted row exists for the saved
   red-cat/blue-chair prompt but is ineligible under the current fingerprint.
7. **The run-files static mount exposed the whole SQLite database.** The
   `/runs` route now serves only validated preview PNG names under canonical
   run UUIDs. The history UI distinguishes execution errors from visual
   rejection.
8. **Configuration and launch instructions disagreed with execution.** The
   planner had ignored `generation.control_image_mode` while config named
   `canny` as the default; it now honors explicit modes and defaults to the
   previously effective `none`. Both benchmark launchers now resolve the
   checkout from their own location. `TESTING.md` no longer points to the
   nonexistent `E:\framegenx`, missing evidence images, or the GPU pressure
   test as a CPU-only command. User-facing benchmark labels now say
   count/spatial pass rate; historical result pages are marked as such.

## Remaining release risks, ranked

| Severity | Finding | Evidence and required next step |
| --- | --- | --- |
| Critical | Full contract satisfaction is unproven. Colors, many attributes, negations, and uncounted presence are recorded but are not hard-gated. | `src/intent/extractor.py` records them; `src/verification/verify_stage.py` hard failures come from counts and spatial boxes. Ten of thirty curated subset prompts are color prompts. Add explicit per-constraint supported/verified/unavailable states and a calibrated attribute/negation gate before claiming full CSR. |
| High | Refined output is generated after acceptance and is never reverified. | `src/generation/refine_stage.py` writes `refined_final.png`; `src/api.py` exposes the verified draft. Decide whether refinement is a separate unverified derivative or reverify it before presenting it as accepted. |
| High | Box heuristics remain weak for multi-instance and occlusion cases. | `src/verification/spatial_verifier.py` uses the first box for each class; `src/verification/qa.py` does not perform the NMS described in the design specification. Evaluate against a labeled spatial/count set and calibrate thresholds before stronger claims. |
| High if network-exposed | There is no user authentication or tenant isolation. | Loopback defaults reduce local exposure, but `/api/runs` lists prompts and `/artifacts/{run_id}` exposes recorded contracts. Do not publish the service without an access model. |
| Medium | Benchmark rows lack a complete experiment version. | Resume keys on config label, prompt and seed, but not source/model revision, dataset snapshot, or adapter revision. Mixed experiments can enter one table after code or model changes. Add a manifest/fingerprint per experiment and segregate reports. |
| Medium | API and standalone benchmark do not share a process-wide GPU lock. | `src/api.py` uses a thread lock; `benchmark.py` runs in another process. Simultaneous use can exhaust 6GB VRAM. Add a cross-process job lease or shared queue. |
| Medium | Release gates are manual. | No repository CI workflow was found; the local environment lacks Ruff/Pyright despite optional Ruff metadata. Add enforced unit, API, frontend, packaging and benchmark-provenance gates. |

## Verification and limits

- `venv\Scripts\python.exe -m pytest tests --ignore=tests/pressure_test.py -q --basetemp .pytest_tmp_quality_full`: **160 passed** after the changes above.
- `venv\Scripts\python.exe -m pytest tests/pressure_test.py -q -k 'not full_pipeline' --basetemp .pytest_tmp_quality_pressure`: **12 passed, 1 deselected**. This was run before the final noun-matching/API changes; the real GroundingDINO noun-matching path was replayed afterward against a saved image, with expected and detected counts both `{cat: 1, chair: 1}`.
- `npm run lint` and `npm run build` in `frontend/`: passed after the UI changes.
- Scoped `git diff --check` with CRLF accepted and PowerShell parser validation for `scripts/continue_full30.ps1`: passed. The unscoped diff check reports pre-existing staged CRLF/cache artifacts in the dirty worktree.
- The earlier report records one full local GPU generation/refinement run and `pip check`. A new full generation and a controlled 30-prompt/LoRA benchmark were not run after this review's changes.
- Browser automation was rejected by the browser security policy for the local URL; visual/mobile/keyboard/WCAG checks remain unverified. The Docker daemon was unavailable in the earlier assessment; container startup remains unverified.
- The worktree had many pre-existing staged artifacts. Review changes were scoped to application, tests, scripts, and documentation. The `.pytest_tmp_quality*` directories remain because the prior automatic approval review rejected recursive cleanup of test temp directories.
