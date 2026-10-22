# TODOS

## ~~Glass-box UI panels~~ — DONE 2026-07-18 & 2026-08-15
- Shipped Generation Studio (per-attempt cards, live stage timeline, draft-vs-final compare, reward history).
- Shipped `AblationDashboard.jsx` (live comparative CSR, reward, CLIP, VQA, and per-category leaderboards from `GET /stats`).
- Shipped `RunHistory.jsx` (paginated runs table, filtering by config / acceptance, and one-click artifact inspector from `GET /runs`).

## ~~Reflect Stage τ-Sweep & Hard-Only Mode~~ — DONE 2026-08-15
- Diagnosed reflect regression via `scripts/reflect_diagnosis.py` and `scripts/tau_sweep.py`.
- Replaced detrimental τ reward-gating with `mode: hard_only` default in `config.yaml` and `src/reflection/reflector.py`.
- Acceptance now reliably requires zero hard failures (counts + spatial constraints).

## ~~Spatial Verification Pipeline~~ — DONE 2026-08-15
- Shipped `src/intent/relationships.py` (NLP extraction of 12 spatial predicates and negations).
- Shipped `src/verification/spatial_verifier.py` (pure CPU geometry, 0 VRAM).
- Integrated spatial violations into `src/verification/verify_stage.py` as first-class hard failures.
- Added comprehensive unit and geometry test suite (109 tests passing).

## ~~Wire Advanced Vision Verifiers as Optional Plugins~~ — DONE 2026-08-25
- Shipped `verification.plugins.scene_comprehension` + `easyocr` config gates, wired in `src/verification/verify_stage.py`.
- **Architecture pivot:** Florence-2 retired — its HF checkpoint is incompatible with transformers 5.x. Scene comprehension rides the LLM vision router (same path as VQA): 0 VRAM, frontier-grade models.
- **Soft-signal semantics:** both plugins populate `verification_report.signals` and NEVER hard-reject — a vision model's phrasing ("a feline" = cat) or incidental background signage must not burn repair loops. Word-boundary, plural-aware matching (no "cat"-in-"category" traps).
- **Latency contract:** both run once per run, on the FINAL attempt only.
- Tests: `tests/test_plugins.py` (9 cases) + repair_policy fragment tests. Suite: 130 passed.
- **Not yet measured:** CSR impact on `text_rendering` / `complex_scenes` categories — enable + run `benchmark.py` before promoting either to a hard gate.

## Next: Regional Latent Conditioning (Phase 2 gate)
- **What:** BoxDiff/MultiDiffusion-style cross-attention steering so each counted entity's adjective tokens attend only inside its planner mask (attack `object_count` CSR, currently the weakest category at 40–60%).
- **Why:** prompt-level corrections have hit their ceiling; masks already exist from the plan stage.
- **After that:** re-run the 15-prompt ablation on current code (results.md predates the 2026-08-24 quality revisions), then the full 30-prompt suite, multi-seed.

**Status clarification, 2026-10-16 planning review:** The regional controller now exists as an experimental explicit profile in `configs/constraint_control.yaml` and `src/generation/controlled.py`. The saved 2026-10-15 comparison reports local count/position 1/8 -> 4/8 and chromatic checks 1/6 -> 3/6; full-contract CSR is unverified and artifacts/binding errors persist. See `reports/constraint_reduction_2026-10-15.md`. The historical percentages above are not current qualified accuracy. Keep calibration, disjoint multi-seed evaluation and qualification work open; do not mark the feature fully qualified from code presence or this note.

## Autoplan proposals — not approved for implementation

Collected from CEO, Design, DX and Engineering review, started 2026-10-15 and continued 2026-10-16. Source: `docs/plans/STACK_EVOLUTION_PLAN.md`, A72. This section preserves the original technology proposal and records conditional follow-ups; it does not authorize adoption, publication, paid calls or deployment. Priority applies only after the stated entry gate. Human and CC estimates exclude empirical study/annotation time and owner decisions.

### F4 — Hosted audience, budget, identity and restore decisions

**What:** Select the hosted task/audience, region, provider-key ownership, monthly spending cap, data retention and support responsibilities, then rehearse a private hosted profile.

**Why:** A deployable service requires ownership and recoverable data, rather than merely public URLs for the local API.

**Pros:** Concrete admission/tenant/storage requirements and a cost-bounded rollout.

**Cons:** Recurring operational duty, remote GPU cost and identity/privacy scope.

**Context:** Current use is a local singleton with SQLite and one 6 GB GPU. PostgreSQL is a conditional hosted repository; Redis is optional wakeup transport, not job truth. The owner has not specified a customer or budget. Begin with the P5 decision checklist and full database-plus-artifact restore protocol; do not expose today's API publicly.

**Effort:** L; human ~5-8d / CC ~2-4d for a bounded pilot after owner decisions; recurring operations excluded.
**Priority:** P2
**Depends on:** P0 qualification/usefulness evidence, P2 durable jobs/artifacts, owner audience/budget/region/identity decisions, licensing and tested restore.

### F5 — Owner-approved license, notices and distribution guidance

**What:** Record project licensing/contribution terms, dependency/model/dataset notices, changelog and a redacted reproducibility issue template before redistribution.

**Why:** The repository has no root license authority; a package or container cannot imply permission or support terms that the owner has not chosen.

**Pros:** Clear reuse terms, reviewable release contents and safer failure reports.

**Cons:** Owner policy work and maintaining a notice inventory; no license is invented by an agent.

**Context:** Reuse CONTRIBUTING and the existing verifier extension guide. New reports should request version/profile/run IDs plus a synthetic reproducer, never `.env` or private prompts. Inspect built wheel/sdist/container contents for private runs/config/cache and bind them to the tested source manifest. This follows DX X4 and Engineering distribution findings.

**Effort:** M; human ~1-2d / CC ~0.5-1d after the owner's policy choice.
**Priority:** P2, mandatory before publication
**Depends on:** Owner-selected terms, P0 package/source inventory and an explicit publication decision.

### F1 — Optional framework and retrieval comparisons

**What:** Compare Streamlit prototypes, LangChain adapters, CrewAI/ADK, LlamaIndex, ChromaDB, RAGAS and Graph RAG against existing direct stages, LangGraph and PyMilvus only for a named missing task.

**Why:** The proposed tools overlap; adding them all does not establish more accurate pixels or a better developer task.

**Pros:** Evidence of useful interoperability, retrieval or prototyping improvements without discarding the baseline.

**Cons:** More dependencies, labeling, migration and maintenance cost; some tools may provide no incremental value.

**Context:** Current retrieval already rewrites, reranks and grades local documents with optional web fallback. Freeze the task, corpus/version, privacy policy, time/call/memory budget and independent labels first. Graph RAG requires a real cross-document reasoning task. RAGAS assesses retrieval/text behavior; image CSR needs separate labeled image evaluation. A Streamlit prototype consumes the same service/evidence contract and does not become a second production interface by default. Retain or reject each candidate based on the same baseline and rollback evidence.

**Effort:** L per comparison; human ~1-3d / CC ~0.5-2d, plus separately budgeted empirical runs/labels.
**Priority:** P3
**Depends on:** Named task and owner priority, P0 labels/protocol, P3 permission/budget interfaces, laptop resource gate.

### F2 — Optional voice with transcript-to-contract confirmation

**What:** Add ElevenLabs or another admitted speech path only for a useful accessibility/hands-free journey, with editable transcript, explicit microphone state and the same contract confirmation.

**Why:** Speech is an input modality; inaccurate transcription must not silently change visual requirements or start generation.

**Pros:** An accessible alternative when users demonstrate value.

**Cons:** Audio privacy/retention, permission errors, provider cost and another quality evaluation surface.

**Context:** No selected voice audience or consent/retention policy exists. Begin with a text-equivalent prototype and denied/cancelled/empty/mistranscribed/offline cases. Keep providers off until configured and permitted. No automatic voice cloning, external publication or notification.

**Effort:** M; human ~3-5d / CC ~1-3d for a bounded prototype; speech/image evaluation separately budgeted.
**Priority:** P3
**Depends on:** Named user task, P1 contract editor, explicit audio/provider consent and budget, resource/quality gates.

### F3 — Scoped n8n / A2A workflows

**What:** Prototype one useful external completion workflow or remote-agent task with scoped identities, signed events and replay-safe delivery.

**Why:** Distributed delivery can duplicate or lose events; remote agents must not bypass contract/owner/budget policy.

**Pros:** Measurable integration value for an identified workflow.

**Cons:** Authentication, recipient permission, retry/reconciliation and deployment complexity.

**Context:** Begin with read-only status/evidence through the shared API/MCP boundary. Any generation tool reuses confirmed submissions and device ownership. Test duplicate/out-of-order events, invalid signatures, unavailable endpoints and exhausted budgets with synthetic recipients. Explicit authorization is required before sending messages or publishing images; a proposal is not that authorization.

**Effort:** L; human ~4-8d / CC ~2-4d per bounded vertical slice.
**Priority:** P3
**Depends on:** Named workflow/recipient authorization, P2 durable submission/job service, P3 scoped tools and P5 identity where remote.

### F6 — At most one dedicated trace backend after measurement

**What:** Compare existing MLflow/local structured records with Phoenix, LangSmith and Opik on the same failed-run debugging task before choosing a dedicated trace backend.

**Why:** An extra observability server should solve a demonstrated task gap without exporting private prompts/images or exceeding 16 GB RAM.

**Pros:** Better cross-stage diagnosis if the current records are insufficient.

**Cons:** Setup, retained data, paid-service/export permissions and another operational component.

**Context:** T4 recommends MLflow/local records first and treats local Phoenix as a candidate; alternatives remain valid. Measure setup time, peak combined RAM, data exported and time to diagnose a representative failed run. Collector outage must not block generation or qualification. Keep payload export off by default and retain a rollback switch. No backend is selected by this TODO.

**Effort:** M; human ~1-2d / CC ~0.5-1d, plus observed debugging/resource tasks.
**Priority:** P3
**Depends on:** A named debugging gap, P3 redaction/ID conventions, owner privacy/cost choice and combined resource gate.

**Mandatory work tracked in the plan, not deferred here:** P1 accessibility/focus/mobile verification, shared design tokens/component reference, truthful qualification states, canonical setup docs and prerequisite-aware task measurement. Optional features do not defer those core release requirements.
