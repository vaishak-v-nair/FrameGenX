<!-- /autoplan restore point: C:/Users/vaish/.gstack/projects/vaishak-v-nair-FrameGenX/main-autoplan-restore-20261015-195627.md -->
# FrameGenX Stack Evolution Plan

Status: APPROVED ROADMAP WITH RUNTIME GATES — user selected A on 2026-10-16; implementation proceeds in separate planned work packages.
Branch/base: main/main. Snapshot: 38eb0f0 plus existing working-tree changes.
Mode: SELECTIVE EXPANSION. Scope: planning artifacts only.
Review started 2026-10-15; continued 2026-10-16 after interruption. Current checked HEAD remains 38eb0f0. User approval and its exact pre-approval plan hash are recorded in `reports/planning/APPROVAL_2026-10-16.md`. The new I-HallA/VSM/Jupyter/integration/testing request is separated before coding in `docs/plans/RESEARCH_TRAINING_EXECUTION_PLAN.md`. Approval accepts the recommendations; empirical qualification and conditional activation gates remain.

## Original proposal

$autoplan Everything is working perfectly. I was thinking, what if we bring these things?:
Frontend: React, TypeScript, Streamlit for rapid prototyping
Backend: Python, FastAPI, PostgreSQL, Redis
LLM integration: LangChain, OpenAI, Gemini, Claude
Multi-agent: LangGraph, CrewAI, Google ADK, A2A
Knowledge: LlamaIndex, ChromaDB, RAGAS, Graph RAG
Tool integration: MCP servers, n8n
Voice and multimodal: ElevenLabs, multimodal LLMs
Evaluation: LangSmith, Phoenix or Opik
ML and fine-tuning: PyTorch, Hugging Face PEFT, LoRA/QLoRA, MLflow
Deployment: Docker, CI/CD, hosted backend and frontend.


## Product objective

Increase measurable visual constraint satisfaction and make runs understandable, reproducible and recoverable within the RTX 3050 Laptop 6GB VRAM / 16GB RAM constraint. Evaluate every named technology without treating overlapping frameworks as cumulative requirements. Owner preferences and objections remain pending the final gate.

## Evidence baseline

- React/JSX + Vite frontend, FastAPI backend and custom stage executor already exist.
- SQLite tracks runs; job state and GPU lock currently live in one API process.
- Retrieval already uses LangGraph, MiniLM, BM25 and PyMilvus, with cloud and web-search fallback paths.
- PyTorch, PEFT, QLoRA and MLflow are implemented. The 90-image, 48-step trained adapter remains experimental and disabled by default.
- The controller comparison reported 4/8 local count/position passes and 3/6 chromatic-binding passes, with uncalibrated verifiers and full CSR unmeasured.
- The saved bottle CLI run found two green bottles but retained background artifacts and unverified requirements.
- Default web generation uses SDXL-Turbo; the tested NF4 SDXL regional controller is a separate explicit CLI profile.
- No tracked GitHub workflow directory was found at intake; Docker files exist, with a host-specific model-cache mount.
- The working tree contains extensive pre-existing source, data, run and generated-file changes. This review preserves those changes.

## Assumptions requiring owner validation

1. Image-contract accuracy remains the primary product outcome.
2. The laptop/local workflow remains supported when hosted services are introduced.
3. Hosted multi-user use, budget, data residency and voice are proposed capabilities, not yet settled requirements.
4. Everything working perfectly is a premise to check against saved evidence, not a quality qualification.

## Proposed roadmap skeleton

- P0: Calibrated evaluation, truthful requirement coverage and durable submission identity; establish clean delivery evidence.
- P1: TypeScript, shared API contracts and a user-editable contract workflow exposing explicit supported profiles.
- P2: Durable jobs and worker recovery with local SQLite support; PostgreSQL and Redis as a hosted deployment profile.
- P3: Provider adapters, local-first tracing and read-first MCP integration.
- P4: Bounded comparative experiments for retrieval, agents and fine-tuning.
- P5: Hosted rollout and optional workflow/voice integrations after product and cost gates.

## Decision Audit Trail

| ID | Phase | Decision | Classification | Principle | Rationale | Rejected / pending |
|---|---|---|---|---|---|---|
| A01 | Intake | Review before implementation | Mechanical | Explicit | User invoked autoplan; changes remain planning artifacts | Application implementation pending approval |
| A02 | Intake | Base branch main | Mechanical | Reproducibility | GitHub default branch and local branch both main | None |
| A03 | Intake | Review UI and developer experience | Mechanical | Completeness | Proposed frontend, API, MCP and deployment changes affect both | None |
| A04 | Intake | Preserve every proposed technology in a decision matrix | Mechanical | Completeness | Avoid silently dropping any original option | Adoption and deferral need final review |
| A05 | CEO 0A | Replace the perfect-quality premise with an experimental baseline | User Challenge UC1, pending | Explicit | Saved images and diagnostics retain reproducible failures | Owner's premise stands until final response |
| A06 | CEO 0C-bis | Recommend staged adoption, with alternative frameworks kept in experiments | User Challenge UC2, pending | DRY / Completeness | Addresses all proposed capabilities while avoiding duplicate runtime owners | Installing the complete stack remains an original option |
| A07 | CEO 0D | Include reproducibility command and manifest in the run view | Mechanical | Boil lakes | Direct component scope, small change, no new infrastructure | None |
| A08 | CEO 0D | Check profile eligibility before GPU submission | Mechanical | Completeness | Reuses controller validation and preserves unsupported requirements | None |
| A09 | CEO 0D | Include durable refresh recovery in P2 | Taste T1 | Completeness | One process restart currently loses job handles | Minimal local-only patch remains viable |
| A10 | CEO 0D | Show immutable contract beside repair changes | Mechanical | Explicit | A repair must not weaken the user's requirements | None |
| A11 | CEO 0D | Include redacted evidence bundle after schema versioning | Taste T2 | Completeness | Makes run evidence portable; expands several files | Defer export until real collaboration demand |
| A12 | CEO 0D | Propose voice and remote workflow follow-ups with entry gates | Taste T3, subject to UC2 | Pragmatic | New privacy and operational scope requires a named journey | Owner may prioritize accessibility or integration research |
| A13 | CEO 1 | Keep deterministic stages and one supervisor per physical GPU | Mechanical | Explicit | Process-local API locks cannot enforce device ownership | Framework-wide rewrite rejected |
| A14 | CEO 2 | Distinguish missing, pending and corrupt artifacts | Mechanical | Completeness | Broad JSON exception handling currently hides integrity failures | Silent absence rejected |
| A15 | CEO 3 | Gate hosted exposure on identity, ownership, quotas and restore | Mechanical | Completeness | Current loopback API has no hosted tenant security boundary | Direct public exposure rejected |
| A16 | CEO 4 | Separate execution completion from qualification and unknown evidence | Mechanical | Explicit | Full-contract claims exceed current diagnostics | Boolean-only verdict rejected |
| A17 | CEO 4 | Persist idempotency and lease ownership; quarantine a still-live stale GPU worker | Mechanical | Completeness | Database fencing cannot stop old CUDA work | Exactly-once execution claim rejected |
| A18 | CEO 5 | Generate types from the authoritative schema; validate at runtime | Mechanical | DRY | Avoid hand-maintained Python/TypeScript contract drift | Parallel schema definitions rejected |
| A19 | CEO 6 | Use held-out human labels and equal-budget paired comparisons | Mechanical | Completeness | Loss and proxy scores cannot qualify exact-count reduction | Uncalibrated promotion rejected |
| A20 | CEO 7 | Measure total GPU/system memory with optional services included | Mechanical | Explicit | Torch allocation is a narrower measurement | Tool-only VRAM claim rejected |
| A21 | CEO 8 | Compare MLflow tracing before adding one dedicated trace server | Taste T4 | Pragmatic / DRY | Existing tracking may cover needs; local Phoenix is a candidate | LangSmith and Opik remain viable |
| A22 | CEO 9 | Use additive migrations and preserve historical result semantics | Mechanical | Completeness | Rollback must not rewrite old evidence or accepted identities | Destructive data rollback rejected |
| A23 | CEO 10 | Keep adapters reversible and review unused experiments | Mechanical | Explicit | Prevent research frameworks becoming hidden runtime obligations | None |
| A24 | CEO 11 | Preserve current React visual language and expose contract/evidence states | Mechanical | Explicit | Main user decision is requirement satisfaction | Aesthetic redesign deferred |
| A25 | CEO required outputs | Initially keep deferred TODO proposals in this plan until approval | Mechanical, superseded by A72 | Explicit | Preserve this initial choice in the audit; Engineering's required backlog collection supersedes location only | Historical TODOs and implementation approval remain untouched |
| A26 | CEO spec review | Record valid capability rejections before execution eligibility | Mechanical | Completeness | Preserve the unconditional denominator | Malformed requests reported separately |
| A27 | CEO spec review | Audit prompt-to-contract preservation separately from image CSR | Mechanical | Explicit | Extracted-contract quality cannot establish original intent retention | None |
| A28 | CEO spec review | Separate terminal cancellation, recovery and new retry identity | Mechanical | Explicit | The initial diagram was ambiguous | Cancelled-to-queued transition rejected |
| A29 | CEO spec review | Split local, hosted and optional module exit gates | Mechanical | Pragmatic | Optional Postgres/Redis/tenant dependencies must not block SQLite releases | None |
| A30 | CEO spec review | Budget GPU/annotation time and protect a bounded six-month sequence | Mechanical | Explicit | Coding estimates exclude measured experiment costs | None |
| A31 | CEO CLI voice | Add verifier-over-existing-generators as a fourth alternative | Taste T5 | Completeness | Independent CLI reviewer identified a smaller product boundary | Integrated local generation remains the draft recommendation |
| A32 | CEO CLI voice | Validate usefulness thresholds against an observed owner/user task | Taste T6 | Explicit | Provisional yield/time defaults lack demand evidence | Do not invent a target customer or commercial requirement |
| A33 | CEO CLI voice | Require power/inference preflight before a qualification study | Mechanical | Completeness | A fixed small cohort may only establish feasibility | Insufficient power stays inconclusive |
| A34 | CEO CLI voice | Add incremental-value ablations and a two-study review point | Mechanical | Completeness | Test the value added by verification/controller/repair | A forced product pivot is not auto-approved |
| A35 | Design 1 | Keep Studio, History and Benchmarks as persistent navigation; use shareable view/job URLs | Mechanical | Explicit | Sketch anchors were not a navigation replacement | No new router framework required solely for three views |
| A36 | Design 2 | Separate generator eligibility, installed availability and verifier coverage | Mechanical | Completeness | Both design voices found conflicting unsupported-requirement behavior | No silent requirement removal or client-controlled bypass |
| A37 | Design 2 | Define visible actions for every job/result state | Mechanical | Explicit | Backend states alone do not tell a user what to do | Fake progress and unsafe retry rejected |
| A38 | Design 3 | Give contract review explicit count/color/relation editing and immutable confirmation | Mechanical | Completeness | Fixed rows cannot resolve omitted or invented requirements | None |
| A39 | Design 4 | Use task-specific workspace hierarchy with diagnostics secondary | Mechanical | Explicit | Image/evidence decisions should lead; framework names do not earn main-screen space | Decorative dashboard redesign rejected |
| A40 | Design 5 | Preserve existing typefaces/palette; define shared semantic components | Mechanical | DRY | Respect project design language and avoid binary verdict reuse | Generic skill typeface replacement rejected for this project |
| A41 | Design 6 | Specify labeled mobile rows, focus containment and bounded announcements | Mechanical | Completeness | Both voices found lost mobile context and general accessibility promises | Browser proof remains pending |
| A42 | Design 7 | Recommend inline contract review within Studio | Taste T7 | Explicit | Preserves workspace context and visible original prompt | Separate guided stepper is viable for complex contracts |
| A43 | Design 7 | Keep user image selection separate from automated qualification | Mechanical | Explicit | Retaining an image cannot alter CSR/evidence status | Manual pass override rejected |
| A44 | DX 0 | Use local Python researcher/contributor as the review persona | Mechanical | Bias toward action | README, Windows launcher and hardware contract support this inference | Owner may override; no customer invented |
| A45 | DX 1 | Ship a dependency-free evidence inspector and real example before GPU setup | Mechanical | Explicit | Concrete first value without torch/downloads/keys | New hosted playground deferred |
| A46 | DX 2 | Require explicit profile and network policy in v1 automation; record omitted seeds | Taste T8 | Explicit | Reproducibility and permitted provider use should be visible | Prompt-first cloud-assisted default is viable with explicit provider permission |
| A47 | DX 2 | Keep legacy generate semantics and add opt-in strict qualification exit behavior | Mechanical | Completeness | Existing scripts must not silently change meaning | Exit zero alone is not a qualification signal |
| A48 | DX 3 | Publish one versioned error envelope with safe recovery actions | Mechanical | DRY | API, CLI and MCP need the same actionable failure | Raw trace/credential output rejected |
| A49 | DX 4 | Make one compatibility table and linear setup route authoritative | Mechanical | Explicit | Conflicting docs and owner-specific paths create avoidable uncertainty | Independent copied setup instructions rejected |
| A50 | DX 5 | Retain v0/legacy readers through one stable release plus 90 days before removal | Mechanical | Completeness | Give users a documented migration and downgrade window | No automatic destructive evidence rewrite |
| A51 | DX 6 | Test inspection, CPU and GPU profiles separately on Windows/Linux | Mechanical | Completeness | Inspecting artifacts must not require CUDA; GPU portability must be measured | All Python >=3.10 support not assumed |
| A52 | DX 7 | Require owner-approved license/notices before redistribution promises | Mechanical | Explicit | No root license file found; dependency/model terms are separate | No invented permissive license or support SLA |
| A53 | DX 8 | Measure prerequisite-aware onboarding locally without exporting private inputs | Mechanical | Completeness | Roleplay timestamps and competitor examples are not measured results | Hidden telemetry rejected |
| A54 | Eng architecture | Enforce device ownership before CUDA use across every entrypoint | Mechanical | Completeness | API-only locking misses CLI, benchmark, training and standalone verification | Uncoordinated direct GPU execution rejected |
| A55 | Eng architecture | Make durable submission identity precede optional job creation and P1 clients | Mechanical | Explicit | Ledger-only rejections also need idempotency and response-loss recovery | Job-only request-key lookup superseded |
| A56 | Eng architecture | Define cancellation/commit ordering and finalizer-loss reconciliation | Mechanical | Completeness | Leased/finalizing/reconciliation states had missing recovery transitions | Cancelled work never requeued automatically |
| A57 | Eng architecture | Back up database, manifests and referenced artifact payloads at one cutoff | Mechanical | Completeness | Hash manifests cannot restore lost images or observations | Manifest-only backup rejected |
| A58 | Eng code quality | Carry stable entity and requirement IDs through observations and evidence | Mechanical | Completeness | Class-level deduplication loses separate object/color obligations | Ambiguous assignments remain unverified |
| A59 | Eng code quality | Use one versioned canonicalizer and distinguish semantic/request/artifact hashes | Mechanical | DRY | Regenerated UUIDs/timestamps can invalidate confirmation and retries | No semantic requirement normalization by prompt rewriting |
| A60 | Eng code quality | Enforce all promotion gates through one versioned typed policy evaluator | Mechanical | Completeness | Existing registry gates are weaker than P0 and cannot be reused unchanged | Loss-only and positive-overall-interval-only promotion rejected |
| A61 | Eng code quality | Centralize typed outcomes and keep adapters independent of ML initialization | Mechanical | DRY | Existing broad exception and import-time pipeline seams need bounded responsibilities | New universal agent base class rejected |
| A62 | Eng code quality | Make CPU tests hermetic with explicit fixtures and outbound denial | Mechanical | Reproducibility | Current conftest loads developer .env even with mocked extractor tests | No live credential dependency in normal CI |
| A63 | Eng CLI security | Recommend a local session/capability boundary in addition to Origin/Host checks | Taste T9 | Explicit | Subagent judged security sufficient; CLI identified missing hostile-browser details | Reusing a full hosted identity provider locally is viable but heavier |
| A64 | Eng CLI security | Allowlist exported evidence fields and identify redacted derivatives | Mechanical | Completeness | Raw prompts/config/metadata can contain embedded secrets | No automatic export of unrestricted raw provenance |
| A65 | Eng CLI deployment | Bind tested releases and qualification evidence to a complete frozen source/build manifest | Mechanical | Reproducibility | Initial commit does not identify extensive working-tree execution changes | Dirty-tree commit identity alone is insufficient |
| A66 | Eng tests | Require behavior/error tests for all 32 new flow groups with each feature | Mechanical | Completeness | Existing 288-test evidence does not cover proposed durability/v1 paths | Conditional features carry their tests when admitted; no test-only success claim |
| A67 | Eng tests | Evaluate intent, provider, retrieval, repair and image policies separately | Mechanical | Explicit | A schema pass, RAGAS result or denoising loss is not visual correctness | Missing labels/power remain inconclusive |
| A68 | Eng performance | Enforce one total deadline including queue, retries, finalization and stop | Mechanical | Completeness | Per-provider timeout alone permits an unbounded user wait | Unstoppable worker quarantines the device; never claim it stopped |
| A69 | Eng performance | Measure peak and percentile GPU/process/system memory with all active services | Mechanical | Explicit | p95 and torch-only counters can miss fatal transient peaks | Memory fit is a measured gate, not a promise from a watchdog |
| A70 | Eng performance | Bound queries, queues and caches using explicit indexes and revision identities | Mechanical | Completeness | More workers or a second store cannot increase one GPU's capacity | No unmeasured N+1 incident claimed |
| A71 | Eng reliability | Define platform-specific flush and failure-durability guarantees before qualification | Mechanical | Completeness | Atomic rename is not a complete power-loss durability guarantee | Unsupported filesystems cannot inherit tested durability claims |
| A72 | Eng required outputs | Append contextual unapproved follow-ups to TODOS.md and retain history | Mechanical | Explicit | Autoplan Eng explicitly requires collecting deferred scope from all phases | Supersedes A25's location choice, not approval state or original scope |
| A73 | Eng amendment check | Include owner-only local bootstrap and authentication on every HTTP example call | Mechanical | Completeness | Source reviewer found the complete example omitted the new access boundary | Bearer CLI transport and browser cookie/CSRF transport remain explicit |
| A74 | Eng amendment check | Restrict v1 submission to the exact confirmed contract; prepare intent separately | Mechanical | Explicit | An earlier generic omitted-contract paragraph contradicted no post-confirm extraction | Legacy behavior remains scoped and cannot enter v1 implicitly |

## 1. CEO review: strategy and scope

### 1.1 Named premises and product outcome

| Premise | Assessment | Decision |
|---|---|---|
| Everything works perfectly | Contradicted by the saved four-case controller cohort: count, binding, shape and verifier errors persist. A successful execution is a narrower result. | UC1: ask the owner at the final gate to replace this premise with an explicitly experimental quality baseline. |
| More AI frameworks will reduce visual hallucination | Unproven. Agent coordination, retrieval, tracing and speech have different jobs from spatial generation and independent visual measurement. | Measure the effect of each addition separately; preserve all proposed options for review. |
| The proposed technologies are all absent | False for React, Python, FastAPI, LangGraph, PyTorch, PEFT, QLoRA, MLflow and Docker. Gemini also has an existing provider path. | Extend the existing seams before adding a parallel implementation. |
| A hosted product is the next business goal | Not established. The proposal includes hosting, but does not specify audience, usage, budget or data residency. | Plan the complete hosted prerequisite chain; make activation an owner decision. |
| Multiple agents should own acceptance | Inconsistent with contracts as source of truth. An agent can propose a repair; it must not change the contract or mark its own work qualified. | Keep a deterministic acceptance policy over recorded evidence and explicit unknown states. |
| Local development must work on the stated laptop | Explicit project authority. Extra containers and CPU models also consume the 16 GB system memory budget. | One physical GPU worker, sequential model residency, optional services stopped by default. |
| More training examples guarantee improvement | Unsupported by the 90-image pilot. Denoising loss is not an exact-count or binding metric. | Curate disjoint, reviewed data and promote weights only after a held-out paired evaluation. |

The main user journey is: express intent, inspect and approve an editable contract, choose a supported generation profile, see progress, inspect the generated image and requirement evidence, then accept a qualified result or understand why the system rejected or could not verify it. Developers need the same journey through CLI, API and later MCP. Adding more frameworks without preserving this journey increases setup burden without demonstrating improvement.

Doing nothing preserves a useful local research tool, its known quality limits, and transient job state. The most direct improvement is a trustworthy acceptance boundary and a recoverable contract workflow. A hosted multi-user platform is a valuable possible destination, with a larger operational and security scope than the current local application.

### 1.2 What already exists: reuse map

| Sub-problem | Existing implementation | Plan treatment |
|---|---|---|
| Browser interface | `frontend/src/App.jsx`, `runState.js`, history and ablation components | Keep React and design language; split by responsibility while adding types and contract review. |
| Typed contract | `src/intent/models.py`, relationship validation, `src/core/context.py` | Version and export the authoritative schema; generated frontend types are consumers. |
| API and polling | `src/api.py`: `/jobs`, job artifacts, run history, ablation; legacy `/generate` | Evolve the asynchronous API; retain documented compatibility until a deprecation window is approved. |
| Pipeline orchestration | `src/core/stage.py`, `src/pipeline.py` | Reuse deterministic stages. Introduce durable scheduling around the pipeline rather than replace every stage with an agent. |
| Local run storage | `src/core/db.py`, per-run artifact directories | Add explicit migrations and job records; preserve existing run IDs and archived evidence. |
| Controlled generation | `src/generation/{controlled,regions,conditioning,fusion}.py`, `configs/constraint_control.yaml` | Expose only allowlisted profile IDs; retain eligibility checks and exact sampler parity tests. |
| Requirement evidence | `src/verification/contract_evidence.py`, local count/color/spatial checks | Separate operational verdict, verified requirement coverage and full-contract qualification. |
| Repair | `src/reflection/reflector.py`, bounded stage loop | Keep bounded attempts and recorded seeds; prohibit contract weakening during repair. |
| Provider fallback | `src/core/llm_router.py` | Add capability-aware OpenAI/Claude adapters and improve existing Gemini path behind the same interface. |
| Retrieval / agents | `src/retrieval/{agent,retriever,memory}.py`, LangGraph, PyMilvus, MiniLM, BM25 | Benchmark existing retrieval before replacing it or adding another store. |
| Training | `src/training/*`, `configs/training_pilot.yaml`, `requirements-ml.txt` | Reuse preflight, prepared data, checkpointing, evaluation and registry; keep adapter disabled until promoted. |
| Experiment records | MLflow tracking and hash manifests in `reports/` | Join model/dataset/quality records with run IDs, not a second competing truth store. |
| Packaging | `pyproject.toml`, pinned requirements, Dockerfile, Compose, Windows launcher | Define tested compatibility profiles and CI from the existing packaging. |

### 1.3 Dream state and alternatives

```text
CURRENT                         THIS PLAN                         12-MONTH IDEAL
Local research pipeline  --->    Measured contract workflow --->   Supported local product
Uncalibrated diagnostics         Calibrated, bounded claims         + optional hosted service
Experimental controller          Durable jobs and typed API        Reproducible qualification
In-memory job handles            Capability-aware integrations     Clear coverage and recovery
Small adapter pilot              Evidence-driven experiment gates  Qualified model catalog
```

| Approach | Coverage / effort estimate | Reuse and benefits | Costs / risks |
|---|---|---|---|
| A. Qualify the local core only | 7/10 for the whole proposal; human 2-3 engineer-weeks / CC-assisted 5-8 active days plus measured GPU and labeling time | Fastest route to credible accuracy; existing UI, DB and stages remain familiar. | Does not deliver hosting, remote tools or speech. Still needs later durable service work. |
| B. Staged local core plus optional hosted profile (recommended) | 10/10 coverage of evaluation and adoption decisions, not mandatory installation of every alternative; human 8-12 engineer-weeks / CC-assisted 3-5 weeks plus owner and hardware gates | Complete end-to-end architecture, incremental releases, shared contracts, one runtime per concern. Local release remains usable. | More migrations, API compatibility and operations work. Optional modules require explicit entry criteria. |
| C. Install the full proposed stack at once | 10/10 tool inclusion; human 4-6 engineer-months / CC-assisted 6-10 weeks plus provider/infrastructure setup | Broad interoperability laboratory; useful if the owner's main objective is framework research. | Several overlapping orchestrators, databases and trace stores; larger test matrix and 16 GB RAM pressure; visual accuracy still unproven. |
| D. Contract verifier and evidence tools over existing generators | 8/10 of the original platform proposal; human 3-5 engineer-weeks / CC-assisted 1-2 weeks plus calibration | Accept externally generated images through a bounded local import and return requirement evidence; smaller generation/hosting ownership. | Import validation and model-independent calibration still needed; does not deliver the integrated repair experience or every proposed integration. |

Estimates are planning ranges, not delivery promises. CC-assisted means implementation with coding agents and human review. Dataset annotation, GPU throughput, third-party access and product decisions do not become faster merely because code is generated. Approach B is the draft recommendation because it addresses every proposed capability without requiring all alternative implementations to run together. UC2 at the final gate asks whether this staged interpretation matches the owner's intent; the original list remains intact until then.

### 1.4 Scope expansion scan and time interrogation

The full proposal touches well over eight files and introduces more than two services if adopted wholesale. Split the work into independently releasable vertical slices with one acceptance gate each. The smallest complete first slice is a calibrated evaluation release plus an honest contract/evidence view; it does not depend on Postgres, Redis or a new agent framework.

The larger product opportunity is a portable evidence bundle: a user or tool can inspect the exact contract, model revision, seed, attempts, verifier versions and qualification status without trusting the live service. That bundle supports CLI, browser, MCP, evaluation and incident recovery. It gives future integrations a shared foundation and makes a failed run useful evidence.

| Expansion / moment for the user | Human / CC-assisted estimate | Draft resolution | Why |
|---|---|---|---|
| Copy a run's reproducibility command and manifest | 2-4h / 1-2h, S | Include in P1 | Direct run-view scope, no new infrastructure; user can reproduce the result. |
| See an unsupported requirement before spending GPU time | 4-8h / 2-4h, S | Include in P1 | Reuses controller eligibility; prevents a misleading submission. |
| Return to a saved job after refreshing | 2-3d / 1-2d, M | Include in P2; taste choice | Several files and persistent state; completeness requires recovery. |
| Read the exact changed requirement between attempts | 3-5h / 1-3h, S | Include in P1 | Show immutable contract and separate repair changes, not an unexplained score. |
| Download a bundle with redacted metadata and checksums | 1-2d / 0.5-1d, M | Include after P1 schema; taste choice | Valuable evidence portability; paths and secret scrubbing need tests. |
| Hear results spoken aloud and submit a voice prompt | 3-5d / 1-3d, M | Defer to conditional P5 | Requires consent, privacy and transcription confirmation; does not establish visual accuracy. |

Other considered opportunities are retrieval ingestion, cross-agent delegation and n8n callbacks; their explicit follow-up gates appear in the technology matrix and backlog. No item above is represented as implemented.

| Review / implementation checkpoint | Decision to settle then | Expected output |
|---|---|---|
| Hour 1 | Preserve evidence and enumerate supported requirements | Baseline manifest, claim vocabulary, contract coverage map. |
| Hours 2-3 | Define calibration labels and release gates | Frozen splits, rubric, failure examples and human review protocol. |
| Hours 4-5 | Design typed contract/job API and recovery semantics | Schema, state transitions and compatibility fixtures. |
| Hour 6+ | Prove the first slice, then choose hosted/provider experiments | Bounded pilot report before broad installation. These checkpoints are ordering aids, not a promise of six-hour delivery. |

Mode is **SELECTIVE EXPANSION**. Small direct-scope improvements are included. Overlapping runtime additions and changes to the user's stated direction remain recommendations for the final gate.

## 2. Technology decision matrix

Every technology from the original proposal has an explicit disposition. “Proposed” means pending approval and capability validation. Existing does not imply production qualification. Documentation links establish capabilities; the adoption decisions are this review's inferences.

| Technology | Current project state | Proposed role and phase | Gate / reason |
|---|---|---|---|
| React | Existing production UI code | Keep, P1 | Preserve navigation, history and artifact views. |
| TypeScript | JSX today; React types installed | Add incrementally, P1 | Shared API schema; strict checking for converted code; remove unchecked API boundary values. [React guide](https://react.dev/learn/typescript). |
| Streamlit | No application path found | Optional experiment UI, P4 | Only for a concrete researcher prototype; import the same service/contract library and never run a competing GPU worker. [Docs](https://docs.streamlit.io/). |
| Python | Existing pipeline and CLI | Keep | Resolve the declared >=3.10 range against the actual pinned 3.14 environment in CI. |
| FastAPI | Existing API | Keep and split lifecycle/worker responsibilities, P2 | API worker replication must not replicate GPU model residency. [Worker docs](https://fastapi.tiangolo.com/deployment/server-workers/). |
| PostgreSQL | Not current run database | Hosted persistence profile, P2 | Only activate with hosted/concurrent-user requirement; migrations, ownership, backups and restoration tests required. [Locking](https://www.postgresql.org/docs/current/explicit-locking.html). |
| Redis | Not current scheduler | Optional hosted notification/queue transport, P2 | DB is job truth; lost wakeups are recoverable by DB scan. Redis delivery does not guarantee one execution. [Streams](https://redis.io/docs/latest/develop/data-types/streams/). |
| LangChain | Not the primary orchestration layer | Optional provider/tool adapter helper, P3 | Adopt a small integration only if it removes code; do not rewrite deterministic stages into an agent loop. [Stack roles](https://www.langchain.com/oss-overview). |
| OpenAI | OpenAI-compatible SDK used for other providers | Optional named provider, P3 | Distinguish SDK compatibility from an actual OpenAI account/model. Select pinned supported models after capability and cost probes. |
| Gemini | Existing router path | Harden existing adapter, P3 | Validate structured responses, refusals, missing content, timeouts and privacy settings. [Structured output](https://ai.google.dev/gemini-api/docs/structured-output). |
| Claude | No current router provider found | Optional named provider, P3 | Use Anthropic's actual API adapter; do not assume OpenAI request compatibility. [Structured output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). |
| LangGraph | Existing retrieval agent | Keep the current bounded graph; optional checkpoint experiment, P4 | Checkpointed execution can replay nodes: external side effects still require deduplication. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence). |
| CrewAI | Not wired | Isolated comparison, P4 | Compare the same task set and budgets with LangGraph; ship only a demonstrated capability gap. [Introduction](https://docs.crewai.com/en/introduction). |
| Google ADK | Not wired | Isolated comparison, P4 | Consider if Google ecosystem integration is a real requirement. Keep one graph owner per workflow. [ADK](https://google.github.io/adk-docs/). |
| A2A | No remote-agent protocol | Conditional interoperability experiment, P5 | Agent-to-agent delegation only for a named remote collaborator; contracts, auth, task identity and cancellation required. [Protocol](https://a2a-protocol.org/latest/). |
| LlamaIndex | Existing custom retrieval code instead | Conditional ingestion experiment, P4 | Need a document ingestion use case and benchmark before adding a second agent/retrieval framework. [Framework](https://developers.llamaindex.ai/python/framework/). |
| ChromaDB | Existing PyMilvus store | Alternative-store benchmark, P4 | Choose one store per deployment. Compare Windows portability, restart behavior, licensing and retrieval quality; no silent corpus migration. [Chroma](https://docs.trychroma.com/docs/overview/introduction). |
| RAGAS | No current evaluation path | Optional retrieval/LLM metrics, P4 | Calibrate custom metrics. RAG answer faithfulness does not measure pixels; visual CSR stays separate. [Ragas](https://docs.ragas.io/en/stable/). |
| Graph RAG | No current knowledge-graph retrieval | Conditional research, P4 | A text knowledge graph is different from the image's scene graph. Need cross-document reasoning demand and measurable retrieval gain. [Microsoft GraphRAG](https://microsoft.github.io/graphrag/). |
| MCP servers | No current public tool server | Read-only local evidence tools first, P3 | MCP connects clients to tools/resources; it does not supply visual correctness. Generation tools later share API authorization and job budgets. [MCP](https://modelcontextprotocol.io/docs/getting-started/intro). |
| n8n | No current automation | Conditional completion workflow, P5 | Signed callbacks, replay-safe delivery and secrets outside workflow exports; no auto-publishing. [Queue mode](https://docs.n8n.io/hosting/scaling/queue-mode/). |
| ElevenLabs | No current voice workflow | Optional speech interface, P5 | Explicit recording consent, deletion rules, visible transcript and contract confirmation. [Transcription](https://elevenlabs.io/docs/overview/capabilities/speech-to-text). |
| Multimodal LLMs | Optional external vision QA exists | Diagnostic judge and contract assistance, P3/P4 | Compare to human labels, permit abstention; never sole acceptance authority. |
| LangSmith | Not current trace backend | Candidate in one tracing spike, P3 | Framework integrations and managed operations may suit a team; evaluate retention/cost/permissions. [Observability](https://docs.langchain.com/langsmith/observability). |
| Phoenix | Not current trace backend | Preferred local tracing candidate, P3; taste choice | OpenTelemetry/OpenInference and local hosting fit the draft local-first direction. Measure RAM and verify full redaction first. [Phoenix](https://arize.com/docs/phoenix/). |
| Opik | Not current trace backend | Alternative in the same spike, P3 | Compare against Phoenix and LangSmith. Current self-host docs distinguish local and production deployment and note user-management limits. [Self-host](https://www.comet.com/docs/opik/self-host/overview). |
| PyTorch | Existing inference and training | Keep | Sequential model residency; whole-process and system memory gates. |
| Hugging Face PEFT | Existing | Keep, P4 training | Reuse adapter validation/registry; do not combine unqualified adapters. |
| LoRA/QLoRA | NF4/rank-4 pilot exists | Curated, held-out training experiments, P4 | Quantized adapter training reduces trainable memory; it does not establish correctness. [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization). |
| MLflow | Existing local experiment tracking | Keep for dataset/weights/quality lineage | Evaluate existing tracing before adding a dedicated server; link trace IDs and run IDs. [Tracing](https://mlflow.org/docs/latest/genai/tracing/). |
| Docker | Dockerfile and Compose exist | Portable optional profile, P0/P2 | Remove hardcoded host cache assumption through configuration; validate Windows/WSL2 GPU behavior separately. [GPU support](https://docs.docker.com/compose/how-tos/gpu-support/). |
| CI/CD | No GitHub workflow found at intake | Add verified checks and gated delivery, P0/P5 | CPU CI on every change; separate GPU qualification runner, release provenance, deployment rollback. |
| Hosted backend / frontend | Local loopback API and static UI today | Conditional private hosted pilot, P5 | Auth, tenant isolation, quotas, durable artifacts, TLS, backups, budget and residency decision before exposure. |

### Research implications for image accuracy

[GenEval](https://github.com/djghosh13/geneval) and [T2I-CompBench](https://arxiv.org/abs/2307.06350) motivate separate evaluation of counts, positions and attribute binding rather than a single aesthetic score. Use pinned benchmark versions and category subsets plus independent human labels; do not compare a custom subset with a published leaderboard total. Their published results do not qualify this application's current controller.

The controller already adapts spatial diffusion ideas documented in `reports/constraint_reduction_2026-10-15.md`. The next experiments should isolate region geometry/background artifacts, detector reflection errors, binding and repair behavior before attempting a bigger agent architecture. Compare at equal inference budgets as well as equal seeds; a 25-step controlled sampler versus a four-step Turbo default is a product profile comparison, not an isolated causal test of the controller.

### Competitive and audience check

The repository's `docs/10_COMPETITOR_ANALYSIS.md` and `docs/WHY_NOT_COMFYUI.md` contain historical assertions about sole-platform uniqueness, high verified numeracy and competitors lacking automated integration. Current [ComfyUI documentation](https://docs.comfy.org/) includes API workflow execution, agent tools, MCP and cloud options. The first release must replace unsupported exclusivity and quality assertions with dated, bounded comparisons; the existing prose is not evidence of a market advantage.

The defensible product hypothesis is portable contracts and calibrated evidence on constrained hardware. The audience is not validated by this repository: a local researcher, an automation developer and a hosted creative team may need different products. Before expanding P3-P5, run three to five task-based sessions with intended users: can they create a contract, interpret unknowns, recover a run and use the evidence bundle? Record task completion, confusion, time and whether an integration solves a task they actually have. This is a demand gate, not a claim that users have already been interviewed.

The six-month regret scenario is maintaining four agent frameworks and three data stores while the verifier still counts reflections as objects. Another is treating a prettier or more capable foundation model as proof that FrameGenX adds value. Compare both the existing local baseline and a permitted stronger generator under the same contract, budget and independent judging protocol; cost and privacy differences stay visible.

Before committing interface expansion or external integrations, observe an intended user or the owner completing one recurring task using the current CLI/UI and their current substitute. Record failure cost, acceptable inspection time, acceptable wait and whether the contract evidence changes the decision. The three to five sessions are a proposed validation study; a private research tool can use an owner-defined research task instead of inventing commercial customers. Demonstrated local reliability and integrity fixes remain justified by reproducible failures and do not wait for customer discovery.

A bounded incremental-value study compares current generation plus manual review, the same generator plus independent FrameGenX verification, then controller and bounded repair additions. Fix the task distribution and total generation budget, and report time/cost per qualified output including unsuccessful submissions, human inspection and GPU occupancy. A stronger generator comparison is separate from this causal ablation. An external-image verifier is an alternative worth testing, not an approved pivot or a demand claim.

## 3. Releases and measurable exit criteria

All numeric thresholds below are **proposed release criteria**, not measured achievements. Freeze them before the evaluation, and revise them only with an audit note and a new evaluation version. No supported-domain claim extends to arbitrary scenes.

### P0: Establish a qualification boundary and reproducible delivery

1. Freeze contract/evidence schema versions and enumerate supported predicates per profile: count, named color, supported spatial relations, negation, style, background, text and other attributes. Mark unsupported or unmeasured requirements explicitly. Preserve every user requirement, including requirements the model cannot satisfy or the verifier cannot assess.
2. Build a licensed annotation manifest with image/contract IDs, source lineage, disjoint scene/prompt/seed families, split hashes and reviewer rubric. Start with 120 calibration images spanning normal cases, reflections, occlusion, cropped objects and color ambiguity; a different locked 120-contract evaluation cohort uses four preregistered seeds per profile. This is a bounded starting study; report uncertainty rather than claim universal performance.
3. Label every required predicate on the calibration set using two blinded human reviewers, adjudicate disagreement and record agreement by category. Fit thresholds using calibration only. Add a separate background/shape artifact rubric and false-rejection review. Benchmark final labels may be collected later by reviewers blinded to profile identity; never use them for tuning the same release.
4. Report strict full-contract CSR (all requirements human-confirmed), per-predicate satisfaction, acceptance precision, false acceptance, false rejection, abstention, requirement coverage, first-attempt quality, quality after bounded repair, latency and memory. Unknowns never become successes; unresolved human labels keep CSR an interval or unavailable with denominators. Publish both unconditional satisfaction over all submitted contracts and the narrower supported-domain figure to prevent cherry-picking by rejection.
5. Proposed promotion gate: no confirmed false accept in a minimum 100 independently labeled accepted cases, with the confidence bound and sample count displayed; this does not establish a zero error rate. For a reduction claim, the preregistered paired 95% confidence interval for full-contract CSR difference must exceed zero, with no >5 percentage-point category degradation and no unsupported-category omission. Cluster resampling by contract family, not correlated seeds. If power is inadequate, label inconclusive and enlarge the next frozen cohort.
6. Preserve pinned versions, model hashes, source snapshot, seed streams, raw detector outputs, masks and final pixels. Separate quality failure from infrastructure error. Never silently remove failed runs from denominators.
7. Establish CPU CI with typed schemas, unit/integration tests, frontend checks, dependency/security checks and an offline test boundary. A GPU qualification lane is explicit and opt-in; its actual status must be recorded before a release. Review source/artifact ignore rules without deleting or staging the current working tree.

For the prompt route, score intent preservation separately from image-to-contract satisfaction. Human reviewers enumerate original prompt requirements before seeing the extracted contract; report retained, omitted, invented and changed requirements, by predicate. A confirmed/supplied-contract CSR claim is explicitly scoped to that contract and cannot establish preservation of the original prompt. The first frozen study includes 120 independently reviewed prompt-to-contract pairs from disjoint evaluation families; calibration/tuning never uses their labels.

Prevent a high-precision system from becoming useless through abstention. Proposed usefulness gates, to freeze with the owner before evaluation, are >=30% qualified yield over the preregistered supported task distribution, <=50% unverified/abstained results, and a 600-second total job deadline with at most two repairs. Latency quantiles and achieved yield remain measured outputs; these targets do not imply feasibility or current attainment. If either qualification integrity or usefulness misses its target, retain experimental status and investigate rather than narrow the task distribution after seeing results.

Record every schema-valid unique confirmed submission before capability eligibility, including unsupported valid contracts, into a durable submission/evaluation ledger. It contains owner/singleton identity, unique request key, original request digest, immutable contract/profile identities, selected seed, evaluation version, admission status/reason, persisted response/outcome and optional linked job ID. This P0 foundation precedes P1 clients; durable execution and recovery complete in P2. Repeated requests with the same key/digest count once; valid capability rejections count as non-satisfied in the unconditional operational denominator without spending GPU time. Malformed/schema-invalid and unauthorized requests are excluded from the image-quality denominator and reported separately without recording private unauthorized payloads. The ledger preserves submitted intent under the configured retention policy; the execution job is created only for an eligible confirmed revision. Preview validation is a separate read/pure operation and never enters this denominator.

| Study workload | Frozen starting size | Compute / labeling budget |
|---|---|---|
| Verifier calibration | 120 reviewed images, two reviewers | Prefer existing licensed images plus <=120 synthetic outputs; threshold tuning only here. |
| Prompt extraction audit | 120 prompt/contract pairs, two reviewers | CPU/provider experiment within a permitted call budget; no image-quality shortcut. |
| Locked image comparison | 120 contracts x 4 seeds x 2 profiles = 960 first outputs | Rough estimate 32.7 GPU hours at 122.5s/output, not a throughput forecast. Bounded repair can increase this substantially. |
| Labeling | 960 outputs x 2 reviewers, plus adjudication | Plan 32-64 person-hours at 1-2 minutes per image per reviewer, with a pilot to measure actual time; complex predicate sets may take longer. |
| Initial execution cap | 48 GPU hours and the stated annotation allocation | Stop at the cap, preserve incomplete cells and report inconclusive. Do not silently shrink the frozen study or claim reduction from a partial convenience sample. A separately versioned larger study needs a new budget. |

The 100-accepted-case precision gate may need more outputs than the initial cohort produces. If that threshold is not reached, report the actual count/bound and keep the profile experimental. Schedule evaluation runs across laptop availability; no concurrency assumption or coding-time estimate absorbs annotation and GPU work.

Statistical preflight comes before the locked study: define a minimum worthwhile improvement (proposed +10 percentage points), estimate paired discordance using a separate small calibration pilot, and compute required independent contract-family count for 80% power at the preregistered error level. Define category comparisons before seeing results: promotion requires point improvement >=the chosen minimum and a positive overall paired 95% interval; each claimed category must also meet a preregistered non-inferiority test with a five-point margin and multiplicity control. If the allocated 120 families/category counts cannot resolve those decisions, label the initial cohort a feasibility/estimation study and plan a separately budgeted qualification study. Never reinterpret a point estimate as statistical evidence of no regression.

The provisional >=30% yield, <=50% abstention and 600-second defaults are guardrails to discuss, not user-derived usefulness evidence. Freeze final values from the observed task and owner priorities before the relevant evaluation. No promotion or hosting decision may treat those proposed defaults as an approved service promise.

P0 exits when calibration and benchmark artifacts exist, coverage is honest, operational verdicts cannot masquerade as full qualification, CPU CI succeeds remotely for a reviewable change and the intended GPU profile has recorded evidence. An inconclusive accuracy result can still ship as experimental; it cannot be promoted as a reduction.

### P1: One contract workflow in React and CLI/API

Add TypeScript incrementally around API responses and run state, then convert components as they are changed. Generate the TypeScript contract from versioned Pydantic/OpenAPI output. Make the contract editor an inspect/validate/confirm step before generation; support a pasted validated contract so the local path does not require a cloud intent service. Profile choices come from server capabilities, not arbitrary YAML paths.

Show operational status and requirement evidence separately in studio, history, demo replay and downloads. Required counts/colors/relations cannot be silently discarded to gain acceptance. Save contract revisions immutably and use their hashes in cache and run identities; edited contracts create new submissions. Preserve keyboard behavior, responsive evidence views and existing navigation.

Exit: the same frozen contract/profile/seed identity appears in CLI, API and UI artifacts; a supported offline example works without provider credentials; unknown requirements are visible on every verdict surface. CPU tests cover schema compatibility and UI state; browser journeys and accessibility checks are required during implementation, subject to the available approved browser environment.

Implementation dependency: P0 submission storage and identity must land before P1 submission clients. Contract review/read-only UI can be developed earlier, but the new live v1 submission route is enabled only after P2 local worker/recovery gates pass. The roadmap labels are work packages, not permission to expose a client before its durable backend exists. If T1 selects a minimal legacy patch, keep v1 submission inactive and document the narrower v0 guarantees.

### P2: Durable jobs and optional hosted persistence

Move execution out of the FastAPI request process. In local mode, use SQLite plus a single worker and durable job table. In hosted mode, use PostgreSQL for job/run ownership and transactions; Redis is optional transport, never the sole record of accepted work. Keep payloads small and store image artifacts by reference.

Use a database uniqueness constraint over `(owner_id, idempotency_key)` on the submission ledger, not only on jobs. The same key/digest returns the same submission and persisted outcome, including a rejection with no job. This is idempotency: a repeated request has the same intended effect. Return 409 on the same key with a different original request digest. Distinguish omitted seed in the original request from the chosen execution seed; choose and commit the latter once, after key ownership is resolved. Persist the job link and queued row in one transaction, or leave an explicit pending-admission record recoverable by a scanner after interruption. A uniqueness constraint on submission-to-job linkage prevents a second job. Store the immutable contract hash, profile digest, provider/model revisions, attempt number, worker lease, monotonically increasing generation token, timestamps, status and artifact manifest digest. Resolve uncertain responses through `GET /api/v1/submissions?request_key=<key>`, whether the submission is pending, rejected or linked to a job; a job ID is optional, never invented.

A lease authorizes a live worker for a bounded time; a generation token prevents a worker with an expired lease from committing late results. It does **not** stop a stale process from occupying VRAM. Assign exactly one supervisor/worker to each physical GPU. Before reassigning that device after missed heartbeats, terminate and confirm the old process has exited; if that cannot be proved, quarantine the device and show recovery required. Do not advertise automatic concurrent replay safety from database locking alone.

Enforce the same device owner at the GPU construction boundary for API workers, CLI/library generation, benchmark, training, CUDA preflight, standalone verification and retrieval models that request CUDA. Acquire the shared host-runtime device gate before model loading or CUDA initialization; direct library execution must join that gate or fail with `FGX_DEVICE_BUSY`. The supervisor holds a process-lifetime OS lock keyed by physical device identity, with its owner/process recorded; its authorized child is the only process allowed to load models for that lease. Duplicate supervisors fail before loading. Native Windows and Docker/WSL cannot be presumed to share a file lock: one explicitly selected runtime owns the physical GPU, and concurrent cross-runtime FrameGenX workers are unsupported and must fail admission. A central broker is required before supporting cross-runtime sharing. External non-cooperating GPU applications remain outside this exclusion guarantee; preflight checks actual headroom and reports contention rather than claiming hardware reservation. Test cross-entrypoint contention, not just two API requests.

Write artifacts to a temporary attempt directory, hash them, atomically rename within the same filesystem, then transactionally commit the manifest under the current state/token/cancellation guard. A reconciler detects a renamed directory with no DB commit and repairs the reference or quarantines it; a DB reference to missing/corrupt artifacts becomes an infrastructure failure, never an accepted cache hit. Use content hashes and a uniqueness constraint to make repeated commits harmless. Finalization remains leased; worker loss during finalization enters `reconciliation_required`. The detailed transition and cancellation rules in 4.4 are authoritative, including uncertain DB commit outcomes.

Local exit: forced API restart, worker kill, duplicate submission, stale lease, cancellation and disk-full tests preserve one authoritative outcome and no concurrent GPU residency using SQLite and one supervised worker. Local mode has an explicit singleton owner and does not claim multi-user access control. Existing archived runs stay readable. Conditional hosted exit: PostgreSQL parity/migration, tenant isolation, backup/restore and, only when Redis is selected, lost/duplicate transport message tests. Neither profile requires copying production data into tests.

### P3: Controlled providers, evidence tools and tracing

Keep `LLMRouter` as the integration seam. Declare capabilities per provider/model: text contract extraction, schema support, vision, context limits, request IDs, timeout, cost budget and credential source. Do not send private data to a fallback provider without the user's permitted provider set. Record the actual provider/model used; reject semantic contract drift even when JSON is syntactically valid. Refusal, malformed JSON, empty response and exhausted budget have separate statuses.

Prototype read-only MCP tools for schema/capabilities, run status and authorized evidence listing. Do not allow arbitrary shell commands, arbitrary filesystem paths or unbounded model downloads. Later generation tools call the same job service with scopes, confirmation policy and duplicate protection. Treat tool descriptions, retrieved documents and agent messages as untrusted data rather than instructions that can redefine acceptance.

Before introducing another evaluation server, compare existing MLflow tracing with Phoenix, LangSmith and Opik on the same representative failed run. Choose at most one dedicated trace backend. Default to structured local logs plus redacted IDs; enabling exported prompts, images or transcripts is a deliberate configuration change. A trace collector outage must not block generation or corrupt qualification.

Local exit: offline supplied-contract flow still works; selected provider fixtures cover the fault matrix; local logs link request/job/run/attempt/MLflow IDs without secrets. A named developer must demonstrate a read-only MCP task improves on the existing CLI/API before that optional adapter is built. If chosen, its local tools enforce the singleton owner; hosted cross-user denial tests activate only with the hosted identity profile. The tracing spike records actual RAM, setup burden and data exported; no dedicated collector is required if MLflow/local logs suffice. No paid provider calls are necessary for normal CPU CI.

### P4: Bounded accuracy, retrieval and framework experiments

Use existing PyTorch/PEFT/MLflow modules. Expand licensed and reviewed datasets by observed failure category, with synthetic counts and per-object color swaps plus natural-image contrast cases. Reject mislabeled, duplicate, ambiguous, corrupted and evaluation-overlapping samples. Keep ground-truth masks/boxes and source licenses; model-generated labels do not qualify themselves. Start with small balanced subsets and increase only after the held-out signal justifies it.

Train rank/learning-rate/conditioning experiments with one GPU job, fixed compute budgets, checkpoint recovery and dataset hashes. Keep the unqualified adapter off by default. Compare denoising loss as a training diagnostic and CSR as the promotion outcome; artifact and category regressions block promotion. A run can finish successfully while its model remains rejected or experimental.

Compare retrieval on/off, current Milvus versus a Chroma alternative, and current bounded LangGraph against CrewAI/ADK only for a named task gap. Use the same contract tasks, time/call budgets and frozen labels. RAGAS evaluates relevant retrieval/text behavior; independent image evaluation determines whether that behavior changes pixels correctly. Graph RAG is admitted only after a real cross-document reasoning task exists.

Exit for any addition: a reproduced improvement in its declared metric, no reduction in acceptance integrity, acceptable laptop memory and latency, documented failure behavior and a rollback switch. A framework experiment can be rejected without invalidating the core product.

### P5: Hosted and optional multimodal workflows

Before hosting: settle intended users, provider spending limits, artifact retention/deletion, model and dataset licenses, data residency and whether users supply their own provider keys. Implement authentication, per-owner access checks, quotas, request budgets, TLS, secrets rotation, recoverable artifact storage and tested backups. Public URLs cannot expose the current local API directly.

Choose one hosting target through a cost and GPU-availability spike; no cloud vendor is selected by this draft. A hosted browser can use a separately deployed frontend and GPU worker service. A browser hosted remotely cannot silently access a user's loopback laptop server; any laptop companion connection needs a separate threat model and owner approval.

Voice, n8n and A2A are conditional vertical slices. Voice shows transcript and extracted contract for confirmation, provides a text equivalent and explicit microphone state, and never fabricates a voice clone consent. Workflows receive scoped signed completion events with retry limits; they never publish images or send notifications without authorization. Remote agents receive least-privilege tasks and cannot override the contract policy.

Exit: synthetic two-user isolation, quota, restore, deletion, provider-outage and cancel/retry journeys pass in staging; actual hosting cost is bounded and approved; rollback is rehearsed. Voice and inter-agent features require their own usefulness and privacy evidence before activation.

## 4. Architecture and deep CEO review

### 4.1 Architecture: boundaries, scaling and coupling

The current API constructs a pipeline at import, stores `JOBS` in memory and uses a `threading.Lock` for GPU exclusion. Replicating that process multiplies job maps and locks, so FastAPI worker count cannot be used as GPU capacity. The proposed boundary keeps API processes lightweight and delegates all GPU ownership to one supervisor per physical device.

Before, the browser is coupled to a default prompt-only pipeline and transient job ID. After, the immutable contract, public profile ID and durable job identity join every interface to one service layer. Provider adapters, retrieval experiments, trace exporters and speech remain optional consumers; none can redefine the acceptance policy or require simultaneous GPU model residency.

```text
BEFORE
React UI -> FastAPI [JOBS + GPU_BUSY + Pipeline] -> SQLite / runs/
                                      |-> providers / retrieval

PROPOSED
React TS / CLI / MCP(read-first)
            |
            v
FastAPI: validation + identity + capability policy
            |                 |-> allowed provider adapters (optional)
            v
Job service -> SQLite(local) OR PostgreSQL(hosted)
            |                       ^
            |-> Redis wakeup? -------|  DB scan repairs missed wakeups
            v
GPU supervisor (one process owner per physical device)
            |
Existing deterministic stages -> generate -> verify -> repair policy
            |                                  |
            v                                  v
Atomic artifact bundle                  Requirement evidence
            |                                  |
            +---------> qualification policy <--+
                             |
                        run/job records
                             |
                    UI / export / metrics

All stages -> redacted logs -> optional ONE trace backend
Training/eval -> MLflow + dataset/model manifests -> promotion gate
```

At 10x load, the GPU queue grows first, not throughput; at 100x, queue admission and per-user budgets reject excess work. Adding Redis cannot make the laptop generate 100 images concurrently. Single points of failure are the GPU worker, local disk and database; local mode reports outages and recoverable jobs, while hosted mode needs backups and durable artifact storage before service promises.

### 4.2 Error and rescue map

Current `_read_json` returns `None` on every exception, making absent and corrupt artifacts indistinguishable. `_run_job` converts any exception to an error string but has no persistent recovery record. Retain a final process-boundary catch for sanitization, while typed failure categories guide retries, abstention and recovery; log contextual IDs and sanitized diagnostics rather than raw arguments or credentials.

The following registry covers the planned critical service boundaries. Classes marked “proposed” are small domain exceptions to define during implementation; library exception families must be confirmed against the pinned dependency versions. “Planned” is not a claim that current code rescues the condition.

| Method / codepath | Failure / exception class | Current treatment | Planned rescue and user outcome |
|---|---|---|---|
| Contract validate / compile | `pydantic.ValidationError`, `ValueError` | Validation exists; web contract submission absent | No retry; field/predicate errors and unsupported profile explanation; no GPU charge. |
| Provider request | `TimeoutError`, `httpx.TimeoutException`, SDK timeout | Fallback exists | Deadline and at most two transient retries inside one total budget; permitted fallback only; visible provider unavailable. |
| Provider quota/auth | SDK `RateLimitError`, `AuthenticationError`, `PermissionDeniedError` | Provider-dependent fallback | Respect retry delay only within budget; never retry bad auth; redact key; actionable server config status. |
| Provider body | `json.JSONDecodeError`, `ValidationError`, proposed `ProviderRefusal` / `EmptyResponse` | Parsing/fallback paths vary | At most one constrained re-request; then explicit intent unavailable/refused; editable supplied contract path. |
| Job submit | Proposed `IdempotencyConflict`, `QueueFull`, DB integrity error | Memory handle, 409 if GPU busy | Same request returns same durable submission/outcome, possibly no job; changed payload 409; full queue 429 with retry hint and resolvable ledger outcome. |
| DB transaction | `sqlite3.OperationalError`, SQLAlchemy `OperationalError` / `IntegrityError` | Shared SQLite connection | Bounded retry only for recognized lock/transient conflicts; rollback; no phantom successful job response. |
| Lease / final commit | Proposed `StaleLeaseError` | No durable lease | Reject stale commit; quarantine GPU if old process persists; recovery status rather than double execution. |
| Redis wakeup | Redis `ConnectionError`, `TimeoutError` | Not present | DB remains queued; scanner recovers; warn operator; user keeps stable job ID. |
| GPU generation | `torch.cuda.OutOfMemoryError`, `RuntimeError` | Terminal thread error | Release process/model resources; no automatic resolution downgrade; record attempt and configured retry budget. |
| Verifier | Model load / corrupt checkpoint / unavailable evidence | Some diagnostic unknown states | Preserve unavailable/unknown; strict gate abstains; never infer pass from missing verification. |
| Artifact read/write | `FileNotFoundError`, `PermissionError`, `OSError`, `JSONDecodeError` | Missing/corrupt may both be `None` | Distinguish pending, missing, corrupt and disk-full; atomic finalization; quarantine incomplete output. |
| MCP tool | Protocol validation / scope / timeout errors | Not present | Structured tool error with stable code; no arbitrary path or shell fallback. |
| Trace export | Export timeout / connection error | No dedicated trace backend | Bounded non-blocking buffer; count dropped spans; generation continues with local provenance. |
| Speech / callback / A2A | Auth, malformed task, duplicate event, timeout | Not present | Conditional modules off; typed failure, bounded retry and visible alternative; no duplicate side effect. |

```text
error -> classify -> validation/auth/refusal? -> stop with useful code
               |-> transient dependency? -> retry within deadline -> allowed fallback?
               |-> missing evidence? -> unverified / abstain
               |-> stale worker / corrupt artifact? -> quarantine + reconcile
               `-> unexpected? -> failed job + sanitized incident ID
```

### 4.3 Security and threat model

Local loopback access is the current deployment assumption, not user authentication. Hosting adds untrusted callers, cross-user artifact access and expensive GPU work, so public exposure is blocked until identity, ownership and quotas exist. Parameterized history queries and UUID path checks are useful existing controls; they do not supply tenant isolation.

Prompt injection means instructions hidden in untrusted retrieved text or tool content. Keep that content outside privileged system policy, use allowlisted tools and validate structured results against the original contract. Secrets, prompts, images, voice and dataset licenses have separate handling requirements; telemetry or a hosted trace product must not silently export them.

| Threat | Likelihood / impact if exposed | Planned mitigation / gate |
|---|---|---|
| Another user reads run/artifact IDs | High / High in naive hosting | Owner scope on every query, signed short-lived artifact access and synthetic two-user negative tests. |
| Anonymous compute exhaustion | High / High | Auth, per-user queue/rate/compute budgets, payload limits and one supervised GPU worker. A rate limit bounds requests per time window. |
| Secret / private prompt leakage in traces | Medium / High | Redaction before instrumentation/export, default payload omission, retention/deletion controls and fixture secret tests. |
| Retrieved instructions call tools or weaken contract | Medium / High | Untrusted-data boundary, tool allowlist, immutable contract, deterministic policy. |
| Remote URL tools reach internal services | Medium / High | No arbitrary fetch tool; explicit host allowlist, address validation and redirect limits for any later downloader. |
| Artifact path traversal / malformed archive | Medium / High | Server-resolved IDs, path containment, bounded archive extraction and checksums; never trust client filenames. |
| Browser session abuse | Medium / High | Choose bearer or secure-cookie auth explicitly. Cookie mode requires CSRF protection (cross-site requests must prove origin) and strict same-site settings. CORS (allowed browser origins) is narrow and does not replace auth. |
| Unsafe model / dependency artifacts | Medium / High | Pinned revisions, safetensors, hash verification, no unreviewed remote code, dependency scan and license manifest. |
| Forged / repeated completion event | Medium / Medium | Signature, timestamp tolerance, event ID deduplication, retry cap and no unapproved outbound actions. |

### 4.4 Data and interaction edge cases

Separate an omitted optional field from an empty user requirement and an unavailable dependency. Permitted intent parsing happens only through an explicit preparation operation or a documented legacy flow. V1 job submission requires the exact confirmed canonical contract; a missing/empty contract returns 422 before nondeterministic extraction or GPU work, never an implicit parse or unconstrained success. Persist the submission ledger after schema validation and before capability eligibility; persist an execution job only after capability validation and confirmation. A UI reconnect reads authoritative state rather than resubmitting.

The same rules apply to the proposed text, speech, agent and tool inputs. Speech cannot skip contract confirmation; a remote agent cannot submit an arbitrary profile file. A transient network error after submission is an uncertain response and must be reconciled through the request key, not translated into a new GPU job.

```text
HAPPY: intent/contract -> validate+confirm -> immutable revision -> job -> evidence -> qualified/failed/unverified
NIL:   no text + no contract -------------------------------> 422 field error
       no optional retrieval/provider ---------------------> eligible local path or explicit unavailable
EMPTY: whitespace / zero requirements / empty transcript ----> field error, no submission
ERROR: provider/DB/GPU/disk/transport ------------------------> typed error -> bounded rescue -> stable status

Voice -> transcript -> user edits -> SAME contract confirmation -> SAME job service
Tool/agent -> scoped structured request -> SAME validation -----> SAME job service
Retrieval -> provenance+data-only context -> validated proposal -> contract never mutated
Telemetry -> redaction -> bounded buffer -> optional collector; outage does not gate result
```

| Interaction | Current gap / planned behavior | Required verification |
|---|---|---|
| Rapid double-click / repeated HTTP request | UI disabling is insufficient; stable request key plus unique DB constraint | Concurrent duplicate submit returns one submission/outcome and at most one job; changed payload conflicts. |
| Refresh / new tab / server restart | Current in-memory job may disappear; persist job and reconnect by authorized ID | Kill API after commit, reconnect and observe same outcome. |
| Back navigation / closing contract dialog | Keep editable draft locally; no execution until explicit confirm | No unexpected request or lost confirmed revision. |
| Edit while a job runs | New immutable revision; running job keeps old contract | Correct hashes in history, cache and evidence. |
| Cancel near completion | DB transition decides winner; cancelled attempt never becomes qualified | Both commit-before-cancel and cancel-before-commit schedules. |
| Worker heartbeat expires | Confirm process exit or quarantine physical GPU | No concurrent VRAM ownership under injected delay. |
| Empty history / 10,000 runs | Helpful first run; stable bounded pagination | Deterministic ordering with equal timestamps and concurrent inserts. |
| Missing / partial / corrupt artifacts | Distinct pending versus broken states | Corrupt JSON/PNG cannot produce success or unsafe cache reuse. |
| Slow provider / offline browser | Deadline, explicit reconnect state and saved job link | Reconnect without duplicate call or indefinite spinner. |
| Expired hosted session | Reauthenticate, then regain authorized job | No other user's metadata visible; preserved draft and no resubmit. |
| Generation-incompatible structure / invalid values | Reject before model load; verifier coverage gaps follow the explicit experimental policy in Design Pass 2 | 0/negative/huge counts, Unicode, long input and unsupported generation relations. |
| Imported old run | Archived schema version and explicit narrower evidence | No retroactive full-contract qualification from old boolean. |

```text
PREVIEW: draft -> canonical validation/coverage -> user confirmation (no ledger)
SUBMISSION: confirmed/schema-valid POST -> durable recorded -> pending admission
                                                        |-> rejected (no job)
                                                        `-> admitted + linked queued job

JOB STATE MACHINE (proposed)
queued -> leased -> running -> finalizing -> completed
   |         |         |           `-> reconciliation_required -> completed / failed
   |         |         `-> recovery_pending -> queued (old process confirmed stopped)
   |         `-> recovery_pending
   `-> cancelled (terminal)
leased / running / finalizing / recovery_pending / reconciliation_required
   -> cancel_requested -> cancelled (only after process stopped and refs settled)
any nonterminal -> failed (recorded unrecoverable execution failure)

completed contains result_status = qualified | constraints_failed | unverified
failed means infrastructure/execution failure, not a bad image
terminal states do not return to running; retry creates a linked job/attempt
transition update compares expected state + lease token; stale updates are rejected
```

| From | Event / precondition | To | Invalid transition prevention |
|---|---|---|---|
| queued | live supervisor atomically claims eligible device | leased | Compare expected state and increment lease token. |
| leased | process started and device owner confirmed | running | Owner/token check; no parallel launch. |
| queued | cancellation transaction wins | cancelled | Claim/cancel serialize; no worker is started after this terminal state. |
| leased/running | worker lost | recovery_pending | Confirm old process exit before requeue; otherwise quarantine the device with an explicit reason. |
| recovery_pending | supervisor confirms old process exit, no cancellation, budget remains | queued | Reclaim through a new token; retain interrupted attempt and never auto-retry a terminal job. |
| leased/running/finalizing/recovery_pending/reconciliation_required | cancellation transaction wins before terminal commit | cancel_requested | Durable cancel guard rejects final commits and automatic requeue; stop any live process. |
| cancel_requested | process stopped and artifact references reconciled | cancelled | Terminal; retain any pixels as cancelled/unverified evidence, with no recovery edge back to queued. |
| running | complete intact attempt, valid lease, no cancel | finalizing | Compare-and-update; rejected stale worker cannot commit. |
| finalizing | manifest and DB transaction succeed | completed | Qualification is a separate result field. |
| finalizing | finalizer dies or commit result is uncertain | reconciliation_required | First read authoritative DB state: an already completed job stays completed. Renew/check the lease while finalizing; never assume a timeout means rollback. |
| reconciliation_required | reconciler claims exclusive recovery, old process stopped, bundle valid and cancel absent | completed | New reconciler token authorizes attaching only the recorded lost attempt's exact bundle/token lineage; no newer attempt may exist. Recheck state and cancel in the commit transaction. |
| reconciliation_required | missing/corrupt bundle or irrecoverable storage state | failed | Durable integrity error; retain diagnostics, never qualify partial evidence. |
| completed/failed/cancelled | late/repeated cancellation | unchanged terminal state | Return the recorded outcome and `cancel_applied=false`/too-late reason; do not rewrite success as cancelled. |
| nonterminal | unrecoverable execution/storage failure | failed | Durable error code and retained evidence. |
| terminal | user explicitly retries | new linked job | Original terminal record is immutable. |

The terminal DB commit is the cancellation linearization point: whichever transaction first persists cancellation or completion determines the valid next state. A renamed artifact directory alone is not completion. A reconciler may adopt a lost finalizer's bundle only under its own exclusive recovery authority and verified predecessor-attempt lineage; the old worker cannot use that authority. An uncertain cancellation response is resolved by status read, without promising that the device has already stopped. A DB outage prevents new admission; a worker that cannot renew within the lease/deadline stops cooperatively, then the supervisor confirms termination or quarantines. This specification is tested with controlled schedules and real process termination; DB fencing alone is insufficient evidence.

### 4.5 Code quality and compatibility

Reuse the existing stage protocol and service functions; a new universal agent layer would duplicate control flow and error policy. Split `src/api.py` by service responsibility when moving job execution, and split `App.jsx` when adding contract/profile/evidence controls. Generated types need runtime response validation at trust boundaries; TypeScript alone cannot validate server data.

The packaging currently declares a broad Python range while its verified local environment is specific. Introduce compatibility profiles and lock/constraint files based on real clean installs rather than promise all >=3.10 environments. Keep public schema adapters and legacy read paths explicit; avoid requiring a simultaneous React rewrite, database migration and provider replacement in one change.

### 4.6 Test review and evidence map

The saved suite pass is bounded evidence for deterministic behavior, not current visual accuracy or a newly executed suite in this planning session. New schema, state and policy behavior should have many fast unit tests, fewer repository/provider integration tests and a small set of complete user journeys. GPU and human-labeled evaluation remain separate required evidence for model promotion.

For the 2am release check, submit a supplied contract, kill the API mid-job, reconnect, finish verification and prove the final displayed image, contract hash, decision and manifest agree. A hostile test submits the same request concurrently, delays the worker beyond its lease, fills the artifact volume and attempts another user's evidence URL. Chaos testing is deliberate interruption of dependencies to verify recovery; it should use synthetic jobs and test directories.

```text
NEW UX: contract edit/profile/evidence/reconnect -> unit state + API integration + browser E2E/a11y
NEW DATA: revision/job/lease/manifest ------------> schema + DB transaction/migration tests
NEW CODE: capability/refusal/unknown policy ------> fixtures + property/boundary tests
ASYNC: cancel/retry/reconcile/queue --------------> fake-clock schedules + real two-process worker tests
EXTERNAL: providers/MCP/traces -------------------> protocol fixtures + opt-in bounded sandbox probes
ML: dataset/train/promote -----------------------> split/hash/resume tests + paired frozen GPU/human study
ERRORS: registry rows ---------------------------> named injected fault + exact user/status expectation
DEPLOY: clean install/backup/restore ------------> isolated CI matrix + staging rehearsal
```

Avoid flaky wall-clock sleeps and live paid-provider tests in normal CI. Pin seeds and package/model versions, but record that cross-device floating-point behavior may differ. Browser and accessibility execution is required during implementation; earlier browser automation approval was rejected, so this planning review does not claim new browser observations or bypass that restriction.

### 4.7 Performance and resource review

The latest saved supplied-contract run took about 122.5 seconds overall and reserved about 2.563 GB in PyTorch; these are a single profile/run measurement, not a p99 service guarantee or total GPU usage. Training, generation and verification must not share the physical GPU concurrently. The biggest new slow paths are generation plus bounded repair, provider fallback and large retrieval/trace startup.

Proposed budgets are process-level p95 memory <=5.0 GB GPU with at least 1.0 GB physical headroom, FrameGenX tree private working set <=10 GB and system committed memory <=14 GB on the laptop during a recorded representative run. Measure with browser and any optional services included, not torch counters alone. Trace collectors, vector stores and local LLMs are off unless their combined footprint passes the budget; use remote optional providers without silently changing privacy permissions.

For local mode, admit one active and at most two waiting jobs initially; hosted pilot limits are per approved capacity. Backpressure means refusing or delaying new work when capacity is full. Use bounded event lists, paginated history, small job records, artifact retention quotas and indexed queries over owner/status/time; avoid loading base64 images into queue messages. Benchmark 20 representative end-to-end runs before setting p95/p99 latency promises; the plan makes no measured percentile claim today.

### 4.8 Observability and debugging

Every request carries request, job, run, attempt and trace IDs, along with contract/profile/model/verifier/dataset versions. Log queue wait, stage durations, allowed fallback, retry reason, evidence coverage, final policy result, stale-lease rejection and reconciliation events. Do not put raw provider errors, keys or unrestricted prompt/image payloads in user-visible messages or exported spans.

Dashboard panels separate execution completion, strict qualification, predicate quality, abstention, false acceptance, queue age, GPU memory and provider cost. Alert on orphaned jobs, lease loss, artifact integrity failures, repeated verifier unavailability, queue age beyond budget and unexpected export volume. Runbooks must reconstruct a run from the manifest and show how to recover a worker without scheduling a second process onto the same GPU.

### 4.9 Deployment, migration and rollback

Local installs continue without Postgres, Redis, speech or a tracing server. Hosted rollout is a separate profile and does not inherit local trust assumptions. Use additive schema migration first, deploy readers that accept old and new records, then switch new writes; compare migrated row counts and hashes, preserve old evidence and test backup restoration before switching traffic.

A feature flag is a configuration switch for enabling a new path. Separate flags control strict qualification, durable jobs, controller profile exposure, new provider adapters, trace payload export and remote tools. Rolling back a model/profile must preserve the historical identities and result semantics; do not relabel prior unverified runs as qualified or return an old cache result under a new digest.

```text
DEPLOY
backup+restore test -> additive migration -> compatible API -> worker canary
  -> synthetic contract smoke -> limited UI profile -> observe -> expand

ROLLBACK
new error/quality budget breached?
  yes -> stop admission -> drain or quarantine worker -> disable new profile/integration
      -> prior compatible API/worker -> reconcile artifacts/jobs -> verify old read paths
      -> keep additive DB columns; restore backup only in a rehearsed maintenance window
  no  -> retain evidence -> widen only within capacity
```

The first five minutes after deployment check health/readiness, schema version, queue ownership, one synthetic job and access denial. The first hour checks error/abstention/queue/resource budgets, restores a test artifact and samples trace redaction. Readiness must distinguish a healthy API from an unavailable GPU or invalid model cache; liveness alone is insufficient.

### 4.10 Long-term trajectory

The proposed architecture creates shared contracts, durable evidence and an explicit experiment boundary. Those capabilities make later MCP, agents, voice and hosting easier without committing the core to one vendor. A new engineer should be able to run a no-key supplied-contract example, inspect an archived failed result and understand why a model was not promoted.

Reversibility is 4/5 for provider, trace and framework adapters, 3/5 for typed API version evolution and 2/5 for hosted data migration after real users arrive. Introduced debt includes maintaining SQLite/Postgres parity, compatibility adapters, dataset annotation and a larger optional dependency matrix. Review unused integrations quarterly and remove experiments that never meet their entry gate; do not let research prototypes become undocumented production paths.

| Horizon | Capacity allocation / decision | Stop or defer rule |
|---|---|---|
| Months 1-2 | Prioritize P0/P1 and local P2, labeling and one frozen study; reserve 48 laptop GPU-hours initially. | If qualification or useful yield is inconclusive, publish experimental scope and spend the next cycle on diagnosed failures. No hosted/voice expansion consumes the qualification budget. |
| Months 3-4 | Only after a target user completes the core journey, admit at most one provider/tool/tracing spike at a time. | If it does not improve a named task within its call/memory/cost budget, retain existing code and archive the experiment. |
| Months 5-6 | Choose between another accuracy study and a private hosted pilot using observed demand and operational capacity. | Hosting stays deferred without owner budget, region, identity, restore and usefulness evidence. No simultaneous agent-framework migration. |

This is a sequencing envelope for an owner plus one implementation stream, not a staffing or delivery commitment. Review the allocation after each frozen report; do not convert optional features into calendar promises.

After at most two frozen studies within the approved compute/annotation budget, hold an explicit owner review. Continue the integrated accuracy roadmap only if the observed task has a worthwhile improvement signal or a specific, testable failure diagnosis justifying another bounded study. Otherwise retain research-only status, narrow the supported domain through a new declared protocol, or consider the verifier-only alternative; the owner chooses. Hosting requires repeat usefulness/demand and operational readiness, and cannot substitute for an inconclusive core result.

### 4.11 Design and user journey

The interface should show the contract before GPU execution and show the image beside its requirement evidence afterwards. Current studio/history/ablation navigation can support that progression; framework names belong in diagnostics or settings only when the user needs them to make a choice. Preserve the existing visual language while making unsupported predicates, provider use and unverified evidence legible.

The emotional journey is uncertainty about the prompt, confidence about the confirmed contract, patient waiting with real stage progress, then either evidence-supported acceptance or a useful explanation of failure. Do not replace honest unknowns with a green completion badge. Mobile users need a single-column image/evidence flow, and keyboard/screen-reader users need field errors, focus return and status announcements.

```text
Studio empty -> prompt or supplied contract -> review/edit -> capability check
  -> confirm -> queued/running (reconnect/cancel) -> image + evidence
  -> qualified / constraints failed / unverified / execution failed
  -> history -> replay exact contract/profile -> export evidence or create new revision
```

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Contract editor | Parsing with deadline | Example supplied contract | Field/predicate error | Confirmed immutable revision | Unsupported fields retained and explained |
| Job | Queued/running stage | No job submitted | Retryable vs terminal code | Execution completed | Recovery/reconnect/cancel pending |
| Evidence | Verifier active | No observations yet | Missing/corrupt evidence | Requirements qualified | Unknowns remain visible |
| History | Bounded list load | First-run guidance | Retry with filter preserved | Authorized saved results | Archived version / missing bundle |
| Provider settings | Capability probe | No cloud key required | Provider unavailable | Allowed capabilities recorded | Some capabilities unavailable |

## 5. Failure modes, non-scope and follow-up work

### Failure Modes Registry (current exposure -> required outcome)

“Test” below describes planned coverage unless an existing test file is named. A planned mitigation does not close a current gap. Critical current gaps are highlighted where lost state or missing evidence could be silent.

| Codepath | Failure mode | Current rescued? | Test / required evidence | User sees now -> planned | Logged? |
|---|---|---|---|---|---|
| API process | Restart loses JOBS handle | N | Restart/reconnect integration | Unknown job -> stable recovered job | Memory only -> persistent events |
| Multi-process API | Two locks allow two GPU jobs | N | Two-process/supervisor fault test | Possible OOM -> queued or capacity error | Partial -> lease + device owner |
| Artifact reader | Corrupt JSON becomes absent | N (CRITICAL GAP) | Corrupt/truncated artifact fixtures | Missing panel -> corrupt evidence warning | No -> typed incident |
| Qualification UI | Hard pass confused with complete evidence | Partial | `tests/test_contract_evidence.py` + all UI surfaces | Narrow pass -> explicit coverage/unknown |
| Worker | Dies before DB commit | N | Kill before/after atomic rename | Lost status -> reconciled or failed | Partial -> durable event |
| Verifier | Reflections counted as objects | N for quality | Frozen reflection/occlusion labels | Wrong diagnostic -> calibrated/abstain | Saved raw outputs |
| Color verification | Ambiguous mask / wrong noun | Partial unknown support | Binding/classification human rubric | Unknown/failure -> same, calibrated |
| Provider | Refusal/empty/schema/timeout | Partial | Provider fault matrix | Generic fallback -> typed permitted recovery | Partial -> redacted trace |
| Retrieval | No local hits / web outage | Partial | Offline/empty/hostile-content fixtures | Optional failure -> no contract drift | Provenance + safe diagnostics |
| Redis (planned) | Lost/duplicate wakeup | Planned DB scan | Drop/duplicate transport message | Durable queued state | Event IDs |
| Stale lease (planned) | Old process continues using GPU | Planned quarantine | Paused worker and failed termination | Recovery required, no second worker | Supervisor incident |
| Hosted routes (planned) | Cross-owner run access | Not currently a hosted identity system | Two-user negative endpoint/tool tests | 404/403 without data disclosure | Security event |
| Training | Data leakage / loss-only promotion | Pilot remains disabled | Split hashes + locked promotion | Experimental or rejected | MLflow + manifest |
| Trace backend (planned) | Collector unavailable / secret capture | Planned bounded local export | Down collector + secret fixture | Generation proceeds; export warning | Dropped-count metric |
| Voice/callback/A2A (conditional) | Transcript drift / forged event / repeated side effect | Not implemented | Consent/confirm/signature/dedup fixtures | Clear failure and text/manual path | Scoped audit |

Current critical silent gaps: corrupt artifacts treated as absent and lost transient execution state after interruption. Multi-process GPU ownership is a critical deployment risk if naive API replication is introduced. No new component in this draft is described as currently tested or repaired.

### NOT in scope for the first qualified local release

1. Mandatory installation of every overlapping agent/retrieval framework: UC2 asks whether the intended goal is a laboratory or a maintained product.
2. Public SaaS launch before identity, quotas, restore and residency decisions: complete hosted prerequisites remain in P5.
3. ElevenLabs voice, n8n notification delivery and A2A deployment before a named user journey: conditional follow-ups preserve the original options.
4. A second production Streamlit interface: prototype only when a researcher needs it.
5. Replacement of the current vector store without a comparative migration benchmark: keep evidence and corpus lineage.
6. Training on scraped or unlicensed images, or accepting generated labels as independent truth: data review is required.
7. Large local LLMs, simultaneous agent models or distributed training on the 6 GB GPU: require measured compatibility or a separately approved remote compute experiment.
8. Aesthetic redesign, a framework rewrite or a new workflow engine without a demonstrated requirement.
9. Claims of hallucination elimination, universal correctness or qualification from training loss.

### Follow-up TODO proposals (planning backlog; do not mark implemented)

| ID | What / why / context | Pros / cons | Effort human -> CC-assisted | Priority / dependencies |
|---|---|---|---|---|
| F1 | Run the optional agent/retrieval comparison against the existing LangGraph/Milvus baseline for a named task. | Broader interoperability evidence / maintenance and labeling cost. | L -> M | P3; P0 labels and P3 budgets. |
| F2 | Add voice only after transcript-to-contract confirmation has a useful target audience. | Accessibility and hands-free input / audio privacy, cost and error handling. | M -> S-M | P3; P1 contract editor and explicit owner gate. |
| F3 | Prototype n8n/A2A external workflows with scoped identities and event deduplication. | Integrations / deployment and security complexity. | L -> M | P3; P2 durable jobs and P5 auth. |
| F4 | Choose hosting region, audience, provider key ownership and monthly budget; rehearse restore. | Concrete deployable service / recurring operational duty. | L -> M | P2; P2 storage and owner decisions. |

These remain unapproved proposals. Engineering A72 collects them into a clearly labeled planning section in `TODOS.md`, preserving every historical item and the owner's original proposal. This explicitly supersedes A25's initial location choice; writing a planning backlog does not authorize implementation or adopt any proposed cut. Each includes enough context to revisit without assuming a hosted business model.

### Dream state delta and diagram audit

After P0-P3 the product has a supported local workflow, honest qualification, typed contracts, recoverable jobs and controlled integrations. P4 may improve visual accuracy only if its paired studies succeed. P5 adds a hosted service and optional voice/workflows after owner and operational gates; none is a prerequisite for locally measured quality.

Existing architecture documents mix historical and current descriptions. This planning change only adds diagrams in this plan, so no application diagram is silently replaced. Implementation must reconcile `docs/architecture.md`, `docs/04_SYSTEM_ARCHITECTURE.md`, `docs/14_INDEX_AND_RETRIEVAL_ARCHITECTURE.md` and `docs/16_CONSTRAINT_CONTROL.md` with the chosen scope; in particular, Turbo defaults and the optional regional controller must remain distinct.

## 6. CEO implementation tasks

These are future implementation tasks synthesized from findings, not completed work. Priority P1 blocks its applicable release, P2 follows within the selected phase and P3 is conditional follow-up. Effort excludes GPU runtime, annotation turnaround and owner decisions.

- [ ] **C1 (P1, human: 3-5d / CC: 1-3d)** - evaluation - Freeze requirement coverage, human labeling rubric and disjoint calibration/benchmark manifests. Surfaced by 1.1 and P0: current proxy results cannot qualify reduction. Files: `src/verification/contract_evidence.py`, `src/training/evaluation.py`, `configs/evaluation_pilot.json`, evaluation scripts and docs. Verify: split/hash tests, blinded label audit, frozen evaluation report.
- [ ] **C2 (P1, human: 2-4d / CC: 1-2d)** - qualification - Separate operational outcome, evidence coverage and strict qualification in schemas, storage and reporting. Surfaced by 4.4: missing evidence can be confused with acceptance. Files: core context, verification, reflection, API, run state and report modules. Verify: unknown/corrupt/legacy fixtures cannot emit qualified status.
- [ ] **C3 (P1, human: 2-3d / CC: 1-2d)** - delivery - Define tested dependency profiles, CPU CI, explicit GPU gates and artifact policy. Surfaced by 4.5/4.9: broad Python declaration and no current remote checks. Files: requirements, pyproject, frontend package, Docker/Compose, proposed CI workflow and docs. Verify: clean Windows/Linux installs, remote CI URLs, recorded GPU gate.
- [ ] **C4 (P1, human: 4-6d / CC: 2-3d)** - interface - Add typed contract review, allowlisted profiles and identical CLI/API/UI identity. Surfaced by P1/4.11: web cannot submit supplied contracts or choose tested controller. Files: intent models, API/service, CLI, frontend API/types/editor/run views. Verify: offline contract journey, identity/cache parity, unsupported-before-GPU and accessibility checks.
- [ ] **C5 (P1, human: 5-8d / CC: 2-4d)** - jobs - Add durable submission, worker supervision, cancellation and recovery. Surfaced by 4.1/4.4: memory jobs and process-local GPU lock. Files: API, core DB, proposed job service/supervisor, migrations and tests. Verify: duplicate delivery, restart, stale process and cancel/commit races with one GPU owner.
- [ ] **C6 (P1, human: 2-4d / CC: 1-2d)** - artifacts - Finalize and reconcile checksummed bundles; distinguish pending/missing/corrupt. Surfaced by 4.2: JSON corruption becomes absence. Files: artifact service, API readers, persistence and export tests. Verify: disk full, corrupt JSON/image, rename-before-commit and stale cache fixtures.
- [ ] **C7 (P1 for hosting, human: 5-8d / CC: 2-4d)** - hosting - Add optional PostgreSQL/Redis profile with owner scopes, quotas, migrations and backup restoration. Surfaced by 4.3/4.9: local trust assumptions do not support public exposure. Files: repository/config, auth/API service, deployment/migration/runbook and isolation tests. Verify: two-user access denial, lost wakeup, restore rehearsal; no hosted activation before owner gate.
- [ ] **C8 (P2, human: 3-5d / CC: 1-3d)** - providers - Extend the existing router with capability and permission policies, structured output and bounded rescue. Surfaced by 4.2/P3: provider fallback can change capability and privacy context. Files: router/adapters, extraction/QA callers, config and provider fixtures. Verify: refusal/empty/schema/auth/timeout/budget matrix without paid CI calls.
- [ ] **C9 (P2, human: 2-3d / CC: 1-2d)** - observability - Compare existing MLflow and one tracing candidate; instrument redacted stage/job metrics. Surfaced by 4.8: fragmented incident reconstruction. Files: tracking, pipeline hooks, API middleware/config and a spike report. Verify: failed-run correlation, collector outage, secret fixtures and laptop RAM measurement.
- [ ] **C10 (P2, human: 2-4d / CC: 1-2d)** - tools - Expose read-only MCP schema/status/evidence tools through the same access policy. Surfaced by P3: external tools need scope and stable IDs. Files: proposed MCP adapter, shared services and tool contract tests. Verify: unauthorized IDs, invalid paths, timeout and no GPU side effects.
- [ ] **C11 (P2 conditional, human: 4-8d / CC: 2-4d)** - ML - Curate failure-focused datasets and run compute-bounded LoRA/QLoRA promotion experiments. Surfaced by P4: 90-image loss-only pilot remains experimental. Files: existing training/data/evaluation/registry modules, dataset manifests/configs and reports. Verify: license/split/duplicate checks, resume equivalence, frozen CSR and artifact-regression study.
- [ ] **C12 (P3, human: 1-3d per experiment / CC: 0.5-2d)** - optional integrations - Evaluate Streamlit, CrewAI/ADK, LlamaIndex/Chroma/RAGAS/GraphRAG, speech, n8n and A2A only against named entry gates. Surfaced by matrix/F1-F3: overlapping frameworks need evidence of value. Files: isolated experiment scripts/configs/reports; core service only if a benchmark supports adoption. Verify: same task/budget baseline, resource use and rollback; no implied production installation.
- [ ] **C13 (P1, human: 1-2d / CC: 0.5-1d)** - product evidence - Correct unsupported competitor/quality claims and validate the next integration with task-based user sessions. Surfaced by the competitive and audience check: old comparison prose claims advantages without current evidence. Files: `docs/10_COMPETITOR_ANALYSIS.md`, `docs/WHY_NOT_COMFYUI.md`, release/research docs and a user-study report. Verify: every comparative claim has a dated primary source or matched experiment; observed user tasks determine optional-feature priority.

## 7. CEO independent voices and completion

### CODEX SAYS (CEO: document-only strategy challenge)

The CLI review completed with exit 0 after a repository-read attempt stalled and was interrupted. The successful review consumed the frozen embedded document and made no independent source-verification claim. Its six concerns were early task discovery, a verifier-only alternative, task-derived usefulness thresholds, statistical power, incremental-value ablations and an explicit core-product stop decision. The draft incorporates each as a planning correction; customer demand and measured quality remain unknown.

### INDEPENDENT CODEX SUBAGENT (CEO: source-grounded strategy and spec)

No Claude model was available through the collaboration interface, so a fresh-context Codex subagent supplied the requested independent review role. Round 1 found six specification issues (7/10); they were amended. Round 2 returned PASS on completeness, consistency, clarity, scope and feasibility, overall 9/10, with no blocking document issues. The plan is not implementation-authorized or empirically qualified by those scores.

| Dimension | Subagent initial | CLI frozen review | Consensus / disposition |
|---|---|---|---|
| Premises valid | OK | OK | CONFIRMED: experimental baseline and no framework-count accuracy claim. |
| Right problem | OK | CONCERN | DISAGREE: T6 preserves research value while adding observed-task usefulness gates. |
| Scope calibration | CONCERN | CONCERN | CONFIRMED concern: local/hosted gates split and expansion conditional. |
| Alternatives explored | OK | CONCERN | DISAGREE: T5 adds verifier-only alternative without forcing a pivot. |
| Competitive/market risks | CONCERN | CONCERN | CONFIRMED concern: dated claims and task observation required. |
| Six-month trajectory | CONCERN | CONCERN | CONFIRMED concern: compute allocation, stop rules and owner review added. |

Consensus is 4/6 matched judgments and two strategic disagreements, not six passes. Reviews saw successive documented revisions; the final specification re-check is subagent-only and marks all six amended strategic dimensions OK subject to owner gates. Both independent instances are from the Codex family; this is independent context, not cross-model diversity. Raw/summarized evidence is in `reports/planning/ceo_codex.md` and `ceo_subagent.md`.

### CEO completion summary

| Required output | Result |
|---|---|
| Mode selected | SELECTIVE EXPANSION; Approach B draft, alternatives A/C/D preserved. |
| System audit | Existing React/FastAPI/LangGraph/ML stack reused; transient jobs, uncalibrated evidence and stale comparisons identified. |
| Step 0 | Seven named premises; reuse map; dream diagram; four alternatives; six adjacent opportunities; temporal decisions. |
| Section 1: architecture | API/worker boundary and duplicate runtime ownership addressed in plan; scaling limits explicit. |
| Section 2: errors | Fourteen critical service-boundary rows; missing/corrupt and lost-state gaps called out. |
| Section 3: security | Eight threat rows; seven high-impact scenarios if exposed without controls; conditional hosted gates. |
| Section 4: data/UX | Twelve interaction rows, four data paths, submission/job state machine and transition table. |
| Section 5: quality | Shared schemas, stage reuse and tested runtime profiles planned. |
| Section 6: tests | Eight test-flow branches, fault specifications and independent GPU/human evidence gates. |
| Section 7: performance | GPU/system-memory, queue, percentile measurement and experiment budgets specified. |
| Section 8: observability | Identity correlation, redaction, metrics, alerts and recovery runbooks specified. |
| Section 9: deployment | Additive migrations, conditional profiles, rollout/rollback diagrams and post-deploy checks. |
| Section 10: future | Reversibility 2-4/5 by boundary; six-month allocation and two-study owner review. |
| Section 11: design | Full contract-to-evidence journey and loading/empty/error/success/partial matrix; deeper review follows. |
| NOT in scope | Nine first-release exclusions/conditional boundaries written. |
| What already exists | Twelve-row reuse map written. |
| Dream state delta | P0-P3 local outcome and P4/P5 conditional delta written. |
| Error/rescue registry | Fourteen boundaries, two critical current integrity/recovery gaps. |
| Failure modes | Fifteen rows; two current silent critical gaps; multiprocess GPU exposure separately flagged. |
| TODOS updates | Four contextual follow-up proposals retained here and collected into the unapproved backlog by Engineering A72; approval pending. |
| Scope proposals | Six presented; five draft includes, one conditional deferral; taste decisions retained. |
| CEO plan | External mirrored artifact written and synchronized. |
| Outside voices | Codex subagent then CLI sequentially; CLI successful pass document-only. |
| Lake Score | 7/7 enumerated coverage choices select complete handling: replay command, eligibility, refresh, contract diff, bundle, calibration coverage, interrupted-run test. This is review choice coverage, not product performance. |
| Diagrams produced | Dream state, before/after architecture, four data paths, error flow, job states, test coverage, rollout/rollback and user journey. |
| Stale diagrams | Historical architecture and competitor descriptions identified; no application document silently rewritten. |
| Unresolved decisions | UC1-UC2, T1-T6 and conditional owner budget/audience/threshold choices remain at the final gate. |
| Spec loop | Two iterations; six initial spec issues fixed; six CLI strategy corrections incorporated; final spec 9/10 PASS. |

Phase 1 complete. Future implementation tasks C1-C13 and a phase JSONL artifact are written. Completion means the planning review ran; current application risks and empirical gaps remain open.

## 8. Design review

### Step 0: scope and evidence

This is an application workspace, with contract review, profile selection, progress/recovery, image evidence, history and benchmark interpretation in scope. No `DESIGN.md` was found. The initial design completeness is 6/10: the CEO plan covers broad states and intent but leaves screen hierarchy, exact action labels, responsive navigation, focus management and profile-exception handling unspecified. A complete design specification would make those decisions explicit while preserving the current React interface.

Reuse `frontend/src/index.css` dark neutrals, existing Inter/JetBrains Mono typography, studio/history/ablation navigation, score/evidence primitives, localStorage reconnect behavior and the existing image preview. This is a deliberate project-consistency exception to the generic skill's preference for a new typeface: replacing typography is not necessary to solve the requested product issues. Add locally served fonts if needed for offline parity rather than introduce another external dependency.

The installed gstack designer was probed and generation failed before any image request: `No OpenAI API key found.` No key was requested or installed. The fallback artifact is `C:/Users/vaish/.gstack/projects/vaishak-v-nair-FrameGenX/designs/contract-workspace-20261015/workspace-sketch.html`, with an unchanged copy of the saved experimental bottle PNG. It is an unapproved static wireframe, not a rendered/browser-tested mockup. No browser automation was attempted because the earlier approval rejection still applies.

Primary focus: separate execution from qualification, put contract confirmation before GPU use, keep unknown requirements visible, define recovery/cancel behavior and make the same task usable on narrow screens and keyboard. Independent design voices run next, followed by all seven passes.

### Design outside voices: source and litmus scorecard

The fresh-context Codex subagent inspected the document, HTML and selected frontend source; the CLI independently reviewed the embedded frozen document/HTML with CEO context only. Both completed before the seven passes. They each found six gaps: capability-policy distinction, editable contract details, selection versus qualification, actionable recovery states, mobile/accessibility semantics and shared hierarchy/navigation. No rendered inspection or cross-model diversity is claimed.

| Litmus check | Subagent | CLI | Consensus before amendments |
|---|---|---|---|
| Brand unmistakable | NO | NO | CONFIRMED concern; add a contract/evidence-specific first screen, not a marketing redesign. |
| One strong visual anchor | NOT SPEC'D | NOT SPEC'D | CONFIRMED missing specification; choose the contract before submit and image/verdict afterward. |
| Scannable headings | YES | YES | CONFIRMED specification intent. |
| Each section has one job | YES | YES | CONFIRMED specification intent. |
| Cards actually necessary | YES | YES | CONFIRMED semantic grouping; avoid a mosaic of score cards. |
| Motion improves hierarchy | NOT SPEC'D | NOT SPEC'D | CONFIRMED missing policy; motion is not required for task success. |
| Usable without decorative shadows | YES | YES | CONFIRMED structural intent. |

Consensus: 7/7 matching specification judgments, including three concerns/missing specifications; this is not seven visual passes. Hard rejection risk was generic presentation and inherited binary badges, resolved in the specification below. The raw CLI review is `reports/planning/design_codex.md`; a subagent finding record is `reports/planning/design_subagent.md`.

### Pass 1: Information architecture (6 -> 9/10)

The current app's Studio, Run History and Ablation Benchmark are useful orientation points. The wireframe's internal anchors illustrate sections and do not replace that navigation; the implementation retains three persistent views labeled Studio, History and Benchmarks. Add URL state such as `?view=studio&job=<id>` and `?view=history&page=2`, relative to the deployment base, with back/forward and authorized reload handling. A historical run opens read-only and never overwrites an unsaved draft; “Create draft from this contract” makes a copy deliberately.

Before submission, rank the original request plus editable contract first, profile eligibility and confirmation second, technical provenance third. During execution, rank the current job state/action first, latest image/evidence second when available, stage/attempt detail third. After execution, rank qualification plus next action first, final image beside requirement evidence second, earlier attempts/diagnostics/provenance third. Use one primary workspace, not a stack of equally weighted pipeline cards.

```text
FrameGenX | Studio | History | Benchmarks | Settings (only when settings exist)
Studio
  Draft: original intent -> editable requirements -> profile/coverage -> confirm
  Active run: state + safe action -> current image/evidence -> attempt/stage details
  Result: qualification + next action -> image | required evidence -> provenance
History -> filters + labeled execution/qualification rows -> read-only run
Benchmarks -> protocol/scope + denominators/uncertainty -> category results -> diagnostics
```

This makes FrameGenX recognizable through its confirmed contract and evidence ledger rather than through a new decorative brand system. A 10/10 rating needs an approved rendered reference and user task observation; the static plan alone cannot establish that.

### Pass 2: Interaction states and capability policy (6 -> 9/10)

The two design voices found a real contradiction: generator incompatibility was mixed with missing verification coverage. Define three independent capability fields from the backend: installed/available, generation-compatible, and verification coverage/calibration. The client displays those facts and the server repeats validation; no client switch may disable required checks or remove a requirement.

| Profile/requirement condition | Before submit | Allowed execution / qualification ceiling |
|---|---|---|
| Model/profile not installed or GPU unavailable | Show named profile disabled with “Unavailable on this machine” and the actual setup requirement. | No GPU scheduling; preserve draft and capability-rejection ledger. |
| Contract structurally incompatible with generator | Inline reason next to offending requirement; offer compatible profile or editing a new draft. | Block with typed eligibility error; never auto-delete the requirement. |
| Generator can attempt, verifier has unsupported/uncalibrated predicate | Show requirement as “Cannot verify with this profile”; summarize before confirmation. | Only a server-defined experimental policy may permit “Run experimental image”; resulting gaps remain unverified. Strict qualification unavailable. |
| Mandatory verifier temporarily unavailable | “Required verification is unavailable. Your draft is saved.” | Strict profile submission blocked; retry capability check. If failure happens mid-run, execution fails with any produced image retained as unverified. |
| Optional diagnostic unavailable | Explain in diagnostics; no false score. | Continue within profile policy; missing requirement evidence still prevents full qualification. |
| All required predicates supported and independently calibrated | Normal contract confirmation, including profile quality status. | Qualification still depends on actual observations and final policy, not preflight alone. |

| Visible state | User sees | Primary action | Other action / disabled reason |
|---|---|---|---|
| Empty Studio | “Describe an image or open an example contract.” | Open supplied-contract example | Paste JSON; no-key path explained at point of use. |
| Parsing / validating | “Preparing your contract…” with actual stage and deadline | Cancel preparation | Generate unavailable until a valid revision exists. |
| Invalid draft | Summary and field-specific error; original text retained | Fix first invalid field | No job submission; examples explain supported values. |
| Ready | Requirements, coverage and exact revision/profile summary | Confirm contract and generate | If coverage gaps are allowed: explicit experimental action wording and acknowledgement. |
| Submission response uncertain | “Checking whether your request was received.” | Check existing request | Reuse request key; do not offer a new-job retry yet. |
| Queued | “Queued. One GPU job runs at a time.” | Cancel queued job | Show known queue position only if authoritative; no invented ETA. |
| Running / automatic repair | Stage, attempt n of maximum, elapsed time and latest actual evidence | Cancel run | “Repair attempt n” only after it starts; no cosmetic percent progress. |
| Disconnected | “Connection lost. The last known run may still be active.” | Reconnect | Preserve last confirmed state, URL and request key; do not resubmit. |
| Cancel requested | “Stopping the worker safely…” | View existing evidence | Start/retry disabled until process stop confirmed; cancellation is not instantaneous. |
| Recovery pending / GPU quarantined | “Run interrupted. Recovery needs the local worker to stop.” | Refresh status | Local operator diagnostic link; no launch onto the occupied device. Hosted users contact operator through the selected support mechanism. |
| Finalizing / reconciling | “Saving and checking the evidence bundle.” | Inspect available evidence | New retry waits for authoritative completion or failure. |
| Completed + qualified | “Qualified for the confirmed contract” plus exact scope/version | Download image | Download evidence; create a new draft. No guarantee beyond recorded scope. |
| Completed + constraints failed | “Required constraints failed. Repair budget exhausted.” | Inspect failed requirements | Try new seed as a linked new job within budget, or create a revised draft. |
| Completed + unverified | “Image generated. Some requirements remain unverified.” | Review unverified requirements | Download labeled unverified output or create a new draft; never a green qualified badge. |
| Execution failed | Specific sanitized cause and stable incident/job ID | Retry only when backend marks safe | Retain any produced unverified image; distinguish provider, GPU, verification and storage failure. |
| Cancelled | “Run cancelled. No further attempts will run.” | Create a new draft / linked retry | Original job remains terminal. |
| Missing/corrupt bundle | “Evidence is incomplete or damaged.” | Retry evidence retrieval / diagnostics | Image download may remain available but is labeled unverified; no qualified cache reuse. |
| Empty history | First run: “No runs yet” plus Open example. Filtered list: “No runs match these filters” plus Clear filters. | Context-specific action | Do not equate zero results with a backend error. |
| History/benchmark load error | Retained filters and “Could not load records” | Retry read | No generation side effect. |
| Benchmark has no comparable suite | “No comparable qualified study is recorded.” | Open protocol / run instructions | No fabricated leaderboard rank, zero percentage or full-CSR label. |

Every result surface uses the same execution and qualification vocabulary. Legacy records display “Archived operational verdict” plus their missing coverage, rather than being upgraded during migration. Completion and “Save selection” have no authority to change qualification: the user may retain or download an image while its evidence stays failed/unverified. Human benchmark annotations are a separate audited process with reviewer identity and adjudication, not an “Accept anyway” shortcut.

### Pass 3: User journey and contract editing (5 -> 9/10)

The original prompt remains visible while editing requirements so an omission or invented object is discoverable. Use grouped object rows with stable instance IDs: changing bottle count from 2 to 3 adds an explicit instance and validation preview; editing its color changes the per-object attribute; selecting “car left of bus” uses existing subject/object IDs and a supported relation selector. Adding/removing a predicate is explicit, recorded in the revision diff and requires reconfirmation; no background extraction overwrites manual edits.

Provide structured controls with a JSON view/paste alternative. A pasted negative count, duplicate entity ID, unresolved relation target or malformed JSON shows a field/path error, preserves the pasted text and focuses the first error after a summary. Schema-valid values are not silently coerced or rounded; profile-specific 2-6 foreground limits and relation restrictions are validated against current server capabilities. Confirmation shows the exact revision hash, profile and experimental ceiling; editing afterwards makes a new draft and invalidates that confirmation.

| Step | User action | Likely feeling / design support |
|---|---|---|
| First 5 seconds | Identify FrameGenX, Studio and a contract example | Orientation from a single task heading, not a framework tour. |
| First minute | Enter intent or paste a supplied contract | Confidence from visible original text, editable rows and no-key example. |
| Before GPU use | Correct extraction mistakes and inspect coverage | Control from explicit errors, profile limits and immutable confirmation. |
| First 5 minutes | Wait, reconnect or cancel | Real stages/elapsed time and safe next action; no fake ETA or duplicated job. |
| Result | Compare expected and observed requirements | Trust from failed/unknown items first and image evidence beside them. |
| Failed/unverified | Decide whether to revise, retry or retain the image | A useful next action and intact history, no forced “success.” |
| Long-term return | Restore a run, replay identity, inspect a model update | Confidence from versioned evidence and compatibility, not retroactive relabeling. |

A running job retains its confirmed revision. “Create a new draft” copies into a distinct draft without altering that job; submission obeys the current queue limit. Preserve unsaved draft state separately from the selected historical run, and warn only when an explicit navigation/reset would discard edits; offer restoration rather than a repeated confirmation dialog on ordinary navigation.

### Pass 4: Specific product UI and generic-design risk (5 -> 9/10)

Classify this as APP UI, not a marketing landing page. The strong visual anchor is the editable contract while preparing a run, and the generated image with qualification heading when reviewing it. Secondary diagnostics live in one inspector with disclosure sections; no icon-card grid or giant hero is added. Name headings Contract, Generation profile, Requirement evidence, Attempts and Provenance; use one primary action for the current state.

Replace the existing “VQA (Precision)” label with “Vision QA diagnostic”; uncalibrated model answers are not statistical precision. CLIP alignment and composite reward remain diagnostics with method/scope, never a proxy green verdict. A requirement row has Requirement, Expected, Observed, Status, Method/version and Evidence link; failed/unknown rows are initially visible. For a calibrated example: count expected 2, observed 2 -> passed; color expected green, observed blue -> failed; surface expected white with no verifier -> unverified. With an uncalibrated method, an apparent operational pass is labeled diagnostic and the qualification field stays unverified.

Use status words plus icons, not color alone. Motion is limited to a short non-blocking disclosure transition and an optional stage activity indicator; reduced motion uses a static “Running” label. Do not add motion to satisfy a generic style checklist. The typography-preservation decision is scoped to this existing product, with critical body/evidence text at least 16px and meaningful hierarchy.

### Pass 5: Design system and component mapping (6 -> 8/10)

No DESIGN.md exists, so the current stylesheet and explicit plan specifications are the working reference, not an invented approved design system. Preserve black/near-black surfaces (`--bg-primary`, `--bg-secondary`), light text, restrained white primary actions and existing semantic green/amber/red tokens. Add explicit neutral/unverified and archived variants and measure their contrast in implementation; do not reuse a binary `accepted` class as the qualification component.

| Component | Existing leverage | Required semantics |
|---|---|---|
| WorkspaceNavigation | Existing studio/history/ablation controls | Persistent three-view orientation, URL state, current-view indication and keyboard behavior. |
| ContractEditor / ProfileEligibility | Existing contract display, authoritative schema/compiler | Original intent, editable stable IDs, validation, coverage and immutable confirmation. |
| ExecutionStatus / QualificationStatus | Existing status pills and runState helpers | Separate fields with text/icon and no automatic legacy upgrade. |
| RequirementEvidenceRow | Existing evidence details and count chips | Expected/observed/status/method plus source link; every required predicate retained. |
| AttemptInspector | Existing image/score/repair components | Immutable contract plus actual attempt edits, seed and diagnostic scope. |
| HistoryList / BenchmarkComparison | Existing history and ablation components | Labeled statuses, denominators, comparable protocol and confidence/coverage. |
| ImagePreviewDialog | Existing Escape/focus-return behavior | Modal focus containment, inert background and explicit close action. |

Use an 8px spacing base with 16/24/32px grouping, consistent 4-8px control radius and existing restrained panel radius where reused. Controls need a visible boundary distinct from the background and a strong focus outline; removing shadows must preserve hierarchy. Plan a small design-token reference as part of P1, with screenshots after implementation, instead of an unrelated redesign project.

### Pass 6: Responsive behavior and accessibility (5 -> 9/10)

At desktop >=1200px, use a maximum 1280px workspace with a flexible image/contract column and a 360-440px evidence/settings column. At tablet 768-1199px, keep persistent wrapping navigation and place the confirmation summary after the contract; avoid a cramped two-column form. At 320-767px, use one column with the current decision first, a full-width image and visibly labeled expected/observed/status groups. History becomes labeled rows with separate Execution and Qualification fields; preserve header associations or use semantic definition lists, rather than hide headers and leave context-free values.

Target [WCAG 2.2 AA](https://www.w3.org/WAI/WCAG22/quickref/), including keyboard access, visible and unobscured focus, text alternatives, error association, contrast and reflow. Use 44x44px interaction targets as a stricter product choice; WCAG AA target-size rules have their own sizes/exceptions, so this is not a claim that AA universally mandates 44px. Require 4.5:1 normal-text and 3:1 large-text/non-text control contrast where applicable, text resizing to 200%, and reflow at 320 CSS pixels / 400% desktop zoom without losing actions or evidence. Test desktop, narrow portrait, tablet, wide screen, zoom and reduced motion during implementation.

After validation failure, focus an error summary linked to the invalid fields; mark those fields with `aria-invalid` and `aria-describedby`. After confirmation, keep focus on the submitted action/state heading and announce that the existing job was created; do not move focus on every poll. On explicit navigation to a completed result, focus its qualification heading. Announce only meaningful stage/status/attempt changes in a polite live region; do not announce elapsed seconds or every 1.5-second fetch. Cancellation keeps focus on the status region until a safe next action is available.

For image preview, follow the [WAI modal pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/): move focus inside, contain Tab/Shift+Tab, make background inert, support Escape, then return focus to the invoker. An image's generated alt text describes the requested task and attempt without asserting that the output actually satisfies it. Chart information also has a textual/table representation. These are requirements for testing, not observed accessibility compliance; browser execution remains unavailable in this review.

### Pass 7: decisions, unresolved items and design non-scope (4 -> 8/10)

The draft resolves navigation ownership, qualification authority, capability policy, contract identity and state-action wording. T7 recommends inline review within Studio, preserving the original request and active run context; a guided multi-step editor is a viable owner preference for complex scenes. If chosen, it must retain the same validation, immutable confirmation, back/forward and recovery behavior. Do not let taste change the acceptance policy or erase unsupported predicates.

| Pending choice | Screens / gate affected | Draft recommendation / consequence |
|---|---|---|
| UC1 quality premise | All verdicts and benchmark claims | Experimental evidence vocabulary until qualification study passes. |
| UC2 staged stack | Settings, tools, speech and hosted surfaces | Only activated capabilities appear; no empty framework dashboards. |
| T2 evidence export | Run actions and sharing | Full local bundle or clearly redacted derivative; redaction cannot claim complete reproducibility. |
| T5 integrated vs verifier-only boundary | Studio versus future image-import view | Preserve integrated workflow; test verifier-only alternative as an experiment. |
| T6 task/usefulness thresholds | Queue/latency expectations and promotion | Freeze from observed owner/user task before the relevant study. |
| T7 inline versus guided contract review | Studio editor and navigation | Inline by default, no modal for a large editable contract. |
| Hosted audience/privacy/budget | Login, settings, retention and support | No hosted UI before the conditional service gate. |

Design NOT in scope: a new marketing site, brand/font replacement, a second production Streamlit UI, unsupported advertised profile options and voice controls before a voice use case is selected. Existing UI reuse is mapped above. Proposed design TODO: record a compact token/component reference and verify browser/a11y behavior during P1; it is mandatory for that release, not deferred accessibility debt. No mockup has been approved; static sketches remain draft.

### Design implementation tasks

- [ ] **D1 (P1, human: 2-3d / CC: 1-2d)** - interface - Specify and implement contract/profile capability review without dropping requirements. Surfaced by Design Passes 1-3: eligibility and verification coverage conflicted; editing interactions were absent. Files: `frontend/src/App.jsx`, proposed editor/profile components, shared API types, `src/api.py`. Verify: edit count/color/relation, invalid pasted JSON, unavailable generator and explicit experimental coverage routes; original contract remains immutable.
- [ ] **D2 (P1, human: 2-3d / CC: 1-2d)** - evidence UI - Implement shared execution/qualification/evidence states and correct diagnostic labels. Surfaced by Passes 2/4/5: binary accepted badges and “VQA Precision” mislead. Files: run-state helpers, App, RunHistory, AblationDashboard and shared status components. Verify: qualified/failed/unverified/error/archived fixtures agree across all surfaces and downloads.
- [ ] **D3 (P1, human: 2-3d / CC: 1-2d)** - recovery UI - Add authoritative job action map, URL restoration and separate editable drafts. Surfaced by Passes 1-3: reconnect and historical run selection lacked ownership rules. Files: App/run-state, navigation and job client. Verify: response loss, refresh/back/new tab, cancel, quarantine, exhausted repairs and safe linked retry.
- [ ] **D4 (P1, human: 2-3d / CC: 1-2d)** - accessibility - Implement labeled responsive evidence, focus management and meaningful status announcements. Surfaced by Pass 6 and both voices: mobile context and modal/validation behavior were unspecified. Files: frontend components/styles and browser/accessibility tests. Verify: keyboard-only complete journey, modal containment, screen-reader announcements, 320px/400% reflow, contrast and reduced motion in an approved browser environment.

### Design completion summary

| Output | Result |
|---|---|
| System audit | No DESIGN.md; existing dark React workspace and components reviewed through source. |
| Step 0 | Initial 6/10; all seven dimensions reviewed. |
| Pass 1: information architecture | 6 -> 9; hierarchy, navigation and draft/run distinction explicit. |
| Pass 2: states | 6 -> 9; capability matrix and twenty visible state rows. |
| Pass 3: journey | 5 -> 9; editing/validation/confirmation and seven-step emotional arc. |
| Pass 4: generic-design risk | 5 -> 9; contract/image anchors, specific row fields and diagnostic terminology. |
| Pass 5: design system | 6 -> 8; current tokens mapped; approved visual reference remains absent. |
| Pass 6: responsive/accessibility | 5 -> 9; precise requirements; execution proof pending. |
| Pass 7: unresolved decisions | 4 -> 8; six mechanical choices resolved, T7 and shared owner gates pending. |
| NOT in scope / reuse | Five exclusions; component/token mapping written. |
| TODOS | One token/reference follow-up integrated into mandatory P1; no deferred accessibility gap. |
| Mockups | One generation attempt failed for missing key; one static HTML sketch; zero generated/approved image mockups. |
| Decisions made | A35-A43; T7 surfaced at final gate. |
| Overall planning score | 6 -> 8.7/10 (mean of seven final scores, rounded); not rendered quality. |
| Outside voices | Six concerns each; litmus 7/7 matching judgments including three concerns; no disagreements. |
| Remaining proof | Approved visuals, user tasks and live browser/accessibility verification. |

Phase 2 complete with bounded static-design evidence. The updated sketch reflects the final hierarchy and visible mobile labels; it is still unapproved. Pass to DX before Engineering.

## 9. Developer experience review

### Step 0A: product type, persona and source audit

Primary type: a local research CLI/API with a React interface, plus conditional hosted/MCP surfaces. It is not primarily a shipped Codex skill; the `.agents` guidance is contributor instruction, so a skill-package UX appendix does not replace API/CLI review. Mode: **DX POLISH**. Initial DX planning completeness is 5/10; earlier broad API/recovery requirements do not yet specify first-run packaging, exact error formats or upgrade semantics.

```text
TARGET DEVELOPER PERSONA (inferred, owner may override)
Who:       Python developer/researcher or contributor reproducing FrameGenX locally.
Context:   Clone on Windows, inspect evidence, then run a supplied contract on a 6 GB GPU.
Tolerance: Wants useful feedback in one terminal session; no measured abandonment data.
Expects:   Supported Python/Node versions, a no-key example, clear GPU/cache requirements,
           reproducible artifacts and a distinction between execution and qualification.
```

Evidence: README points new users to TESTING, has PowerShell setup and a native launcher; CLI supports `generate --config --contract --seed`; CONTRACT/control docs provide supplied-contract execution. The API already exposes interactive docs and tests contain useful fixtures. The audit found conflicting version promises: `pyproject.toml` says >=3.10, CONTRIBUTING says 3.10+/CUDA 11.8+, README setup says >=3.13, while the saved tested environment is 3.14 with its pinned CUDA stack. CONTRIBUTING links AGENTS to a nonexistent `.agents/AGENTS.md`; top-level LICENSE and CHANGELOG.md were absent in this inspection. No license is inferred or chosen for the owner.

### Step 0B: Developer Perspective (source-grounded roleplay, not an executed test)

I open the README and see contract-driven generation with first-class verification. The experimental quality note helps me understand what the tool can and cannot establish. I follow the link to TESTING because it promises a step-by-step path. It tells me to use PowerShell in the project directory and to create a virtual environment before I run anything.

I then compare the installation guide with CONTRIBUTING and the package metadata. They name different Python requirements, and I do not know which combination is supported. The normal install includes the machine-learning stack even if I only want to inspect a saved result. I need to understand CUDA, model downloads and a retrieval index before trying the full prompt path. The short generate command also leaves me wondering whether my prompt will use a cloud provider.

The supplied-contract example in the control guide is closer to the reproducibility task I came for. I can see a path that avoids intent parsing, but I still need the right cached models and configuration. After a run, I want a machine-readable answer explaining execution, qualification and unknown requirements separately. I do not want to infer that distinction from an exit code or an old acceptance boolean. I would trust the project more if the first example exposed a real evidence bundle, then guided me through a precise preflight for my own generation.

### Steps 0C-0D: benchmark, first useful moment and timing

Three official-source searches covered local image-generation onboarding, ComfyUI onboarding and Python API/CLI getting started. No comparable measured end-to-end onboarding time was found; the table therefore reports unknown times rather than invented competitor timings. These are adjacent onboarding patterns, not a claim of equal product scope.

| Tool | Verified timing | Useful onboarding choice | Primary source |
|---|---|---|---|
| ComfyUI | Not measured here / no comparable published value established | Desktop/local generation and workflow/agent integration entry points. | [Official docs](https://docs.comfy.org/) |
| Hugging Face Diffusers | Not measured here; model download and hardware matter | Small pipeline API, installation guide, explicit loading and memory optimization paths. | [Quickstart](https://huggingface.co/docs/diffusers/quicktour) |
| FastAPI | Not measured here | Minimal API example, explicit entrypoint and automatically discoverable docs. | [First steps](https://fastapi.tiangolo.com/tutorial/first-steps/) |
| FrameGenX today | Fresh-machine onboarding not timed | Native launcher, detailed TESTING guide, real artifacts and supplied-contract CLI. | README / TESTING / control guide |

The first useful moment is inspecting a real contract/image/evidence bundle and seeing why the result is unverified, then reproducing it when the GPU/cache are ready. Choose a copy-paste evidence inspection command over building another hosted playground. Package a compact real example with hashes, provenance and the existing experimental label; it must not be a fabricated successful output. The inspection path must not import torch, download models or require provider keys. The command/package design is proposed and will be finalized in the eight passes; it does not exist merely because it is named here.

| Timing clock | Current evidence | Proposed target / honest boundary |
|---|---|---|
| First evidence inspection | Not measured on a clean machine | <=2 minutes after repository and supported Python are present, with no ML install or network; separately report clone/Python setup time. |
| Warm-cache supplied-contract generation | Saved example ~122.5s overall; one run, not a percentile | <=5 minutes for the documented example after preflight, with a <=600s job budget. Measure the actual distribution. |
| Cold GPU setup and first generation | Not timed; requires large model/dependency downloads | No under-five-minute promise. Show prerequisites, measured download/setup sizes and staged progress. |

The competitive target is meaningful feedback under five minutes for the documented prerequisites; “zero to GPU image” remains unmeasured. This split prevents a saved replay from being misrepresented as new generation.

### Steps 0F-0G: developer journey and confusion report

| Stage | Actual path / friction evidence | Planned resolution / status |
|---|---|---|
| 1. Discover | README introduces the contract loop and links TESTING. Historical competitor docs overstate quality. | Put honest scope and one next action first; C13 corrects claims. Planned. |
| 2. Evaluate | Reports distinguish uncalibrated diagnostics; no lightweight installed-package inspection entrypoint. | Real compact evidence bundle, no-key/CPU inspection and explicit limitations. Planned. |
| 3. Install | Version promises conflict; default package pulls ML dependencies; Windows/Linux commands mixed in CONTRIBUTING. | Canonical tested profiles, clean-install gates, separate shell instructions and preflight. Planned. |
| 4. Hello world | Default prompt route can require providers/retrieval; control example lives deeper in docs. | Evidence inspection first, warm-cache supplied-contract generation second. Planned. |
| 5. Integrate | Async `/jobs`, legacy `/generate` and CLI have different input support and result semantics. | Versioned shared contract/job schema, complete request/poll/result examples and compatibility policy. Planned. |
| 6. Debug | Several useful messages exist, but artifact absence/corruption and some raw errors lack stable codes/remedies. | Problem/cause/fix/docs envelope; redacted diagnostics keyed by run/job. Planned. |
| 7. Upgrade | No CHANGELOG found; manual SQLite widening and historical evidence schemas. | Additive migration, backup/restore rehearsal, versioned readers and deprecation guide. Planned. |
| 8. Scale / deploy | Loopback Compose, hardcoded cache mount and process-local GPU exclusion. | Explicit local versus hosted runbook and one worker/device; no throughput promise from Redis. Planned. |
| 9. Migrate / extend | Stage/verifier extension guide exists; model/dataset/project licensing and portable export need clarification. | Stable extension contract, owner-approved license/notice policy and versioned evidence export. Planned. |

```text
FIRST-TIME DEVELOPER CONFUSION REPORT (simulated from source; times are illustrative)
T+0:00 README -> TESTING: clear project purpose, experimental limitation visible.
T+0:30 README Run it vs CONTRIBUTING: conflicting Python/CUDA expectations.
T+1:00 pip install path: ML install required before an evidence-only evaluation.
T+2:00 generate example: unclear credential/network prerequisite for my exact path.
T+3:00 control guide found: supplied contract works in principle, but preflight and
       machine-readable qualification semantics need one consolidated route.
Resolution: address all five source-grounded friction points; these are not measured
failure timings or a claim that commands were executed in this review.
```

Prior DX log: July 17 review recorded 4 -> 7/10 and a 3-minute target, using DX TRIAGE. That is an older, narrower review and not proof of today's onboarding time or the expanded plan. A prior project learning about cosmetic benchmark variants was checked against current `benchmark_dataset.json`; variants remain, reinforcing contract-family splitting rather than treating every prompt suffix as independent.

### DX outside voices and consensus

The fresh-context Codex subagent and subsequent document-only Codex CLI each returned six concrete concerns. Both identified the same missing contracts for lightweight inspection, canonical installation, automation semantics, errors, documentation and upgrades. Neither ran installation or measured timing. Engineering has not yet run; the DX CLI prompt explicitly records that ordering rather than supplying a fabricated engineering consensus.

| Dimension | Subagent | CLI | Consensus before amendments |
|---|---|---|---|
| Getting started <5 minutes | CONCERN | CONCERN | CONFIRMED missing entrypoint/package contract. |
| API/CLI naming/defaults | CONCERN | CONCERN | CONFIRMED missing public interaction/exit contract. |
| Actionable errors | CONCERN | CONCERN | CONFIRMED missing stable envelope/remedies. |
| Findable complete docs | CONCERN | CONCERN | CONFIRMED authority/route gap. |
| Safe upgrades | CONCERN | CONCERN | CONFIRMED compatibility/deprecation gap. |
| Dev environment friction | CONCERN | CONCERN | CONFIRMED dependency/shell/profile gap. |

Consensus is 6/6 matching concerns, not six passing journeys. The eight passes below amend them; successful CLI output is `reports/planning/dx_codex.md` and subagent findings are `dx_subagent.md`.

### Pass 1: Getting started (5 -> 9/10)

Define the proposed checkout command `python scripts/inspect_example.py --example green-bottles --verify-hashes`. This script is standard-library-only and ships with a compact real example under `examples/evidence/green-bottles/`: PNG, confirmed contract, verification/decision, provenance and versioned SHA256 manifest. Do not repackage it as a fake qualified success. It prints `source=saved_example`, `bundle_integrity=ok`, recorded execution status, current evidence scope, qualification `unverified`, and paths for the image/evidence; checksum integrity is not visual correctness. Unknown manifest versions and corrupt/missing artifacts fail with a useful error and no network/GPU work.

The proposed first-use sequence is clone/unpack, change directory, then run that command with the documented Python. The <=2-minute target begins only after repository and Python availability; separately record those prerequisites and their setup/download costs. Supply exact example output from the shipped bundle in docs and run the command in a fresh Python subprocess with outbound access denied and torch imports prohibited during CI. This is a future acceptance test, not an executed inspector today.

Current generation remains available through the already implemented supplied-contract CLI; the following is an existing command shape, not the proposed inspector. It requires the documented GPU environment, cached models and no competing GPU job:

```powershell
$env:HF_HOME = 'E:\hf_cache' # choose your actual cache directory
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
venv\Scripts\python.exe -m src.cli generate 'two green bottles on a white surface' --config configs/constraint_control.yaml --contract configs/examples/green_bottles_contract.json --seed 43
```

Those Hugging Face flags control model loading; they do not make every application stage offline. The explicit current control profile bypasses intent with the supplied contract and disables its retrieval/external vision paths. The proposed general `--offline` policy must enforce all outbound stage restrictions and be tested separately. Generation preflight checks model revisions/files, disk space, driver/CUDA, device availability and the profile's verifier dependencies before spending GPU time.

### Pass 2: API/CLI ergonomics and automation (5 -> 9/10)

Keep existing `generate --config --contract --seed` usable. Move heavy imports behind argument parsing so `--help` and the lightweight inspection route do not initialize the model stack; do not require a new SDK for basic API use. New versioned operations use the same shared service/schema and stable public profile IDs. Local CLI may retain an explicit file-based `--config` escape hatch; remote API/MCP never accept arbitrary filesystem configuration paths.

**The v1 operations and flags below are proposed, not currently implemented.** Defaults: public v1 generation requires `profile_id`; network is offline unless a permitted provider set is supplied; a supplied contract bypasses intent. An omitted seed is selected once after request deduplication and persisted before queueing, then returned in every response. A retry with the same key must not select a new seed. The frontend can show a recommended installed profile, but always shows the actual profile and quality scope before confirmation.

| Operation | Proposed API / CLI | Contract |
|---|---|---|
| Capabilities | `GET /api/v1/capabilities`; `framegenx profiles list --json` | Installed availability, generation limits, verifier coverage, quality status and profile digest. |
| Validate contract | `POST /api/v1/contracts/validate`; `framegenx contract validate <file> --profile <id>` | Pure preview validation, canonical contract hash and eligibility/coverage; not an evaluation submission denominator. |
| Prepare intent (optional) | `POST /api/v1/contracts/prepare`; `framegenx contract prepare <text> --provider <permitted-id>` | Explicit permitted extraction returns an editable draft and provenance; no generation, confirmation or quality denominator. User reviews the exact canonical contract afterward. |
| Submit | `POST /api/v1/jobs`; `framegenx generate --contract <confirmed-file> --profile <id> --request-key <uuid>` | Exact confirmed canonical contract, hashes, profile, seed policy, network policy, request key. Returns a submission with optional job; never extracts or changes intent after confirmation. |
| Reconcile uncertain POST | `GET /api/v1/submissions?request_key=<key>`; `framegenx submissions resolve --request-key <key>` | Owner/singleton-scoped durable pending/rejected/admitted outcome with optional job ID; do not create another submission after response loss. |
| Status / wait | `GET /api/v1/jobs/{id}`; `framegenx jobs wait <id> --json` | Separate execution/qualification, actual stage, attempt, safe recovery action and error fields. |
| Cancel | `POST /api/v1/jobs/{id}/cancel`; `framegenx jobs cancel <id>` | Repeated cancel is harmless; stopping/terminal acknowledgement is explicit. |
| Result / evidence | `GET /api/v1/jobs/{id}/result`; `GET /api/v1/runs/{id}/evidence`; `framegenx evidence show <run-id> --json` | Immutable contract/profile/model/verifier identities, full requirement coverage, manifest and authorized image references. |

An idempotency key is created and saved by the client before its POST. Scope uniqueness to owner plus key in the submission table and compare the canonical original request digest; a reused key with changed content returns 409. Reserve that key before nondeterministic work, commit seed selection once, and persist rejection/confirmation-conflict/capacity outcomes as well as admitted jobs. Same-key retry resolves the original outcome; changing the contract, accepting a new profile revision or deliberately retrying a terminal outcome requires a new key and parent submission link. A crash after reservation leaves a recoverable pending admission, not a fabricated missing-job error. In local mode records/keys are not automatically purged by default. Any later hosted retention policy must publish its deduplication window; clients never deliberately reuse an old key for a new intent, and cannot assume protection after the documented window.

Proposed shared result example (labels illustrate the schema, not a new measured run):

```json
{
  "schema_version": "framegenx-result-v1",
  "job_id": "recorded-job-id",
  "run_id": "e2f31522-0b8f-48cc-a173-b5af48557ce8",
  "execution_status": "completed",
  "qualification_status": "unverified",
  "evidence_scope": "confirmed_contract_only",
  "contract_hash": "sha256:<recorded>",
  "profile_id": "instance-control-experimental",
  "profile_digest": "sha256:<recorded>",
  "seed": 43,
  "unverified_requirements": ["surface/background", "uncalibrated attribute evidence"],
  "safe_recovery_action": "inspect_evidence"
}
```

CLI JSON is one versioned object on stdout; logs/progress go to stderr and contain no credentials. Preserve legacy generate's documented operational exit semantics until deprecation. Add `--require-qualified`: 0 means qualified, 2 means constraints failed, 3 unverified, 4 execution/infrastructure failed, 64 invalid usage/contract and 130 user interruption. Submission/status/read commands return 0 for their own successful operation, not for visual qualification; their JSON status is authoritative. A pending/uncertain submission returns a documented nonterminal outcome and request key rather than a false completed result. Pin exact meanings in CLI/API compatibility tests.

Proposed complete local HTTP flow after v1 implementation; it cannot run against the current API. Prerequisites are the local API running with the same owner identity configuration and an owner-only local bootstrap. The proposed `auth local-bootstrap` writes a protected client credential file outside the repository, with Windows owner ACLs (or owner-only POSIX permissions in the equivalent guide), and prints only a non-secret file/status reference. It cannot be called remotely to mint credentials. The example captures the credential into request headers without printing it. This uses explicit bearer transport without ambient cookies; browser cookie/session transport must additionally include the configured CSRF token and exact Origin validation. Hosted examples replace this local identity with scoped hosted credentials supplied securely by the caller:

```powershell
$base = 'http://127.0.0.1:8000/api/v1'
$sessionFile = Join-Path $env:LOCALAPPDATA 'FrameGenX\local-client.json'
framegenx auth local-bootstrap --client-file $sessionFile # proposed; owner-only, no secret output
if ($LASTEXITCODE -ne 0) { throw 'Local authentication bootstrap failed.' }
$localSession = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json
$headers = @{ Authorization=('Bearer ' + $localSession.access_token) }
$contract = Get-Content -Raw 'configs/examples/green_bottles_contract.json' | ConvertFrom-Json
$previewBody = @{ contract=$contract; profile_id='instance-control-experimental'; network_policy='offline' } | ConvertTo-Json -Depth 30
$preview = Invoke-RestMethod "$base/contracts/validate" -Method Post -Headers $headers -ContentType 'application/json' -Body $previewBody
if (-not $preview.generation_eligible) { throw 'Profile is incompatible; inspect the validation result.' }
$key = [guid]::NewGuid().ToString() # persist before POST in a real client journal
$body = @{ contract=$preview.canonical_contract; profile_id=$preview.profile_id; expected_contract_hash=$preview.contract_hash; expected_profile_digest=$preview.profile_digest; acknowledged_coverage_digest=$preview.coverage_digest; seed=43; network_policy='offline' } | ConvertTo-Json -Depth 30
$submitHeaders = @{} + $headers
$submitHeaders['Idempotency-Key'] = $key
try {
  $submission = Invoke-RestMethod "$base/jobs" -Method Post -Headers $submitHeaders -ContentType 'application/json' -Body $body
} catch {
  $submission = Invoke-RestMethod "$base/submissions?request_key=$key" -Headers $headers # includes ledger-only rejection
  if (-not $submission.submission_id) { throw } # keep journal/key; never blindly create a new one
}
$deadline = (Get-Date).AddSeconds(600)
while ($submission.admission_status -eq 'pending' -and (Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 1
  $submission = Invoke-RestMethod "$base/submissions?request_key=$key" -Headers $headers
}
if ($submission.admission_status -eq 'rejected') { throw "Submission rejected: $($submission.error.code). Retain $($submission.submission_id)." }
if (-not $submission.job_id) { throw "Admission is unresolved; retain request key $key and resolve it later." }
$jobId = $submission.job_id
do {
  $state = Invoke-RestMethod "$base/jobs/$jobId" -Headers $headers
  if ($state.execution_status -in @('completed','failed','cancelled')) { break }
  Start-Sleep -Seconds 1
} while ((Get-Date) -lt $deadline)
if ($state.execution_status -notin @('completed','failed','cancelled')) { throw "Wait deadline reached; retain job $jobId and request key $key." }
$result = Invoke-RestMethod "$base/jobs/$jobId/result" -Headers $headers
$result | ConvertTo-Json -Depth 30 # inspect qualification_status separately
```

The production client journal, machine-readable uncertainty outcome and typed HTTP error handling are tested service/client behavior, not implied by this educational script. Contract and profile hashes are recomputed on submission; a stale confirmation returns an actionable conflict without changing the contract.

### Pass 3: Errors and debugging (6 -> 9/10)

Define `framegenx-error-v1` shared by API, CLI and tool adapters: stable `code`, readable `message`, `cause`, `field_path`, `fix`, `docs_ref`, request/job/run IDs when known, `retryable`, and an enumerated `recovery_action`. The backend owns recovery safety, such as `retry_read`, `resolve_request`, `new_linked_job`, `edit_contract` or `operator_required`; clients do not infer safe retry from a generic 500. Documentation references are versioned project-relative paths resolvable from the packaged docs or release source. Do not serialize raw SDK exceptions, full environment/config dumps or secret-bearing request bodies.

| Specific current path | Current user experience | Proposed code / HTTP / remedy |
|---|---|---|
| CLI missing YAML / invalid contract JSON | “Configuration file does not exist” without path, or uncategorized read/parse error | `FGX_CONFIG_MISSING` / usage 64; show sanitized local path and how to pass `--config`. `FGX_CONTRACT_INVALID` / 422, field path, original text retained, edit and validate again. |
| Required cached model/verifier missing or wrong hash | Loader exception or downstream stage error | `FGX_PROFILE_UNAVAILABLE` / 503, no GPU scheduling; name missing model revision, required bytes/disk and documented explicit cache preparation command. Never download without request. |
| Lost job / artifact JSON corruption | Unknown in-memory job or absent panel | `FGX_JOB_NOT_FOUND` / 404 with request-key reconciliation if available; `FGX_EVIDENCE_CORRUPT` / 409 or recorded failed result with manifest diagnostics, operator recovery and no qualified cache reuse. |

Additional named examples cover invalid relation IDs (422/edit contract), unavailable provider/auth permission (503/configuration action, no secret echo), changed content under one request key (409/new deliberate key), GPU OOM (recorded execution failure, operator/device cleanup), and quarantined worker (503/operator required, no repeated launch). Bounded retries apply only to specified transient codes and one total deadline; every permanent error has problem, cause, fix and a help reference. Verbose mode adds redacted timings/identities, not private provider payloads.

### Pass 4: Documentation and learning (5 -> 9/10)

Make `docs/COMPATIBILITY.md` the proposed canonical tested environment/profile table, linked by README, TESTING, CONTRIBUTING and Docker notes. README presents purpose, limitations and a linear Inspect evidence -> Prepare local generation -> Run supplied contract path. TESTING covers verification commands and what each proves; CONTRIBUTING covers clean environment, tests, stage/verifier extensions and PR evidence. Fix the AGENTS link to the repository root and replace owner-specific `E:\BrosKi\framegenx` requirements with “from your checkout root”; an example cache path is configurable, not universal.

Add a docs index with six task routes: first inspection, local generation, CLI/API integration, debug/recover, contribution and upgrade. Include separate PowerShell-compatible and Linux shell blocks, expected output generated from tests, prerequisite/network labels and explicit existing-versus-proposed/released commands. Avoid copying the compatibility table into four independent guides. Schema-generated API docs supplement a complete asynchronous workflow example; they do not replace it.

Use the existing stage/verifier extension guide as the base for one runnable minimal verifier example returning passed/failed/unverified plus method version. Its tests prove missing/unknown evidence cannot qualify a contract. Keep a no-key example and one real failed/unverified bundle easy to find. A developer should reach the appropriate guide within two minutes in an observed task; no such timing has been measured yet.

### Pass 5: Upgrade and migration (6 -> 9/10)

Publish compatibility across application version, API version, DB migration version, ContractV2 and evidence/manifest versions. Keep current `/jobs`, `/generate`, `/api/runs` and archived readers as documented compatibility adapters during introduction of `/api/v1`. Deprecation removal needs at least one stable replacement release and 90 days of explicit notice, a migration guide and an owner-approved release decision; warning headers/help text name the replacement. This is a proposed policy, not a claim that a stable v1 is already published.

Upgrade sequence: stop admission, drain or safely stop the device worker, establish a declared cutoff and back up a consistent database snapshot plus all referenced immutable artifact payloads and manifests. Payloads include final/attempt images, masks, contracts, observations, decisions and required provenance. Verify a backup inventory against its hashes, then restore into an empty isolated directory with the original artifact storage unavailable; read representative archived/new bundles and reconcile in-flight records at the cutoff. Only then run additive migration, run archived/read-only fixtures, start a compatible worker/API and execute a synthetic smoke. Do not permit mixed worker versions to share one device; negotiate API/worker schema capability and refuse incompatible work. Older applications may read only declared supported evidence versions; future unknown schemas fail with `FGX_SCHEMA_UNSUPPORTED`, offer read-only raw export and never rewrite them.

Rollback preserves additive columns and existing bundles when compatible. A destructive DB downgrade or restoring an older backup requires an explicit maintenance plan covering newer jobs/artifacts; no automatic rollback discards them. Add CHANGELOG and an upgrade guide, test migration reruns/interruptions and retain representative old-run fixtures. Codemods/data transforms are provided only for an actual breaking change, with preview and backups; do not invent a migration framework for unneeded rewrites.

### Pass 6: Developer environment and tooling (5 -> 8/10)

Three profiles are explicit: inspection uses supported Python and standard library only; CPU contributor checks install pinned development/CPU dependencies with no CUDA or provider calls; native GPU generation uses the tested CUDA/torch/model profile. UI development additionally needs supported Node/npm, TypeScript and the existing Vite build. Training, hosted DB/Redis and tracing extras are separately installable/activatable and tested in isolated profiles, so one optional package does not silently change the generator's environment.

Live version probes in this planning turn returned Python 3.14.6, Node 26.7.0, npm 11.19.0 and PowerShell 7.6.6; this proves executable versions only, not clean-install support. The installed Vite declares Node `^20.19.0 || >=22.12.0`; Docker currently selects Node 22. Initial compatibility candidates are the existing Python/CUDA pins, host Node 26.7 and container Node 22, narrowed to exact tested versions/digests before release. Do not repeat the package's broad >=3.10 promise without a passing matrix. Use PowerShell statements without `&&` unless the guide explicitly requires PowerShell 7.

CI validates Windows paths with spaces, Linux CPU installs, dependency resolution, no-network tests, frontend type/build/lint and optional-feature imports. GPU validation is a separately labeled NVIDIA lane, not a presumed standard GitHub runner. Editor hints come from generated TypeScript types and Python schemas; fixtures/mock providers let contributors debug without spending tokens. Update cache-volume configuration for non-E drives and document proxy/offline cache preparation. This review did not execute a fresh install or CI run; platform support remains gated by those results.

### Pass 7: Community, contribution and extension (4 -> 8/10)

No current adoption/community or support SLA is inferred. Use the repository's existing contribution guide and an explicitly enabled issue/discussion route, with a redacted bug-report template asking for version, profile, run/incident IDs and a minimal synthetic reproducer. Do not ask users to attach `.env` or private prompts. The owner must choose a project license and contribution terms before redistribution promises; model/dataset/third-party notices are tracked separately.

Keep stage/verifier registration documented and versioned; a small checked example teaches extension without requiring CrewAI/ADK. For MCP, document tool schemas, read/write scopes, cancellation and per-call budgets; it is a protocol integration rather than a fabricated Codex skill ecosystem. Optional paid services identify which task sends data externally and require user/provider configuration; the local inspection and supplied-contract route need no OpenAI/Claude/Gemini account. No new hosted free tier, community team or pricing is promised by this draft.

Proposed follow-up TODO: add license/notices and issue-template/reproducibility guidance before package/container redistribution. Value is clearer reuse and safer bug reports; cost is owner policy and maintaining a small notice inventory. It depends on P0 packaging and the owner's licensing decision, with no assumption about legal terms.

### Pass 8: DX measurement and feedback (6 -> 9/10)

Record observed task start/end, prerequisite state, shell/OS, cache warmth, command outcome and friction category for evidence inspection, CPU setup, warm generation and cold setup separately. Collect a short local task diary plus reproducible failure reports; recording is opt-in and does not export prompts, images, audio or environment values. Proposed acceptance is first inspection <=2 minutes after declared prerequisites and a warm example <=5 minutes, with achieved distribution, incomplete attempts and participant counts shown. Do not convert simulated timestamps into telemetry.

Before each release, have a fresh contributor or owner in a clean isolated environment follow the published commands and compare observed outcomes with the docs. Track successful task completion, time, error-recovery success and perceived clarity rather than raw command counts alone. Review recurring friction after each release and once per quarter; update the compatibility/docs examples in the same change. These feedback loops establish an empirical DX baseline after implementation; current competitor timings and new-product onboarding remain unknown.

### DX scorecard and implementation checklist

| Dimension | Initial primary | Final plan | Historical prior / interpretation |
|---|---:|---:|---|
| Getting started | 5 | 9 | No per-dimension historical score recorded. |
| API/CLI/SDK | 5 | 9 | Prior overall 7 is not a dimension baseline. |
| Errors/debugging | 6 | 9 | Stable envelope/remedies now specified. |
| Documentation | 5 | 9 | Canonical authority and complete task routes. |
| Upgrade/migration | 6 | 9 | Compatibility/deprecation/downgrade rules. |
| Environment/tooling | 5 | 8 | Actual clean-install/platform execution still required. |
| Community/ecosystem | 4 | 8 | Owner license/support decisions remain explicit gates. |
| Measurement/feedback | 6 | 9 | Prerequisite-aware measured tasks planned. |
| Overall planning completeness | 5.25 | 8.75 | Not an implemented-product score or measured trend. |

TTHW: current fresh-machine value unknown -> proposed <=2-minute evidence inspection after prerequisites; warm generation <=5 minutes; cold setup separately measured. Competitive tier is a target for those explicit prerequisites, not an attained ranking. First useful moment is designed through a real evidence command, not a new hosted service. Persona and DX POLISH mode remain as declared in Step 0.

- [ ] Ship/test the standard-library inspector, real versioned example and expected output.
- [ ] Measure the declared timing clocks in clean environments; report setup separately.
- [ ] Verify no-key/no-network inspection and explicit generation preflight.
- [ ] Freeze v1 operation/default/result/exit/uncertainty semantics and test legacy adapters.
- [ ] Ensure every public error has problem, cause, fix, docs and safe recovery action.
- [ ] Generate types/docs from the authoritative schemas and test full asynchronous examples.
- [ ] Publish tested compatibility, clean shell-specific setup and reproducible model-cache instructions.
- [ ] Test upgrade/restore/downgrade boundaries with archived evidence fixtures.
- [ ] Add changelog and owner-approved license/notices before redistribution.
- [ ] Provide a safe issue/reproducer path and minimal verifier extension example.
- [ ] Keep optional paid integrations off until configured; do not promise a hosted free tier.
- [ ] Measure/find/fix onboarding friction without exporting private data.

DX principle coverage: first-use friction, learning by doing, actionable uncertainty, explicit defaults/escape hatches, contextual examples and a meaningful first artifact are all specified. New language SDKs, a hosted playground, marketplace and skill distribution are NOT in scope without demonstrated demand. Existing CLI, TESTING, stage/verifier guide, FastAPI docs, real artifacts and pinned requirements are reused rather than replaced. All new commands above remain proposed until implementation.

### DX implementation tasks

- [ ] **X1 (P1, human: 1-2d / CC: 0.5-1d)** - onboarding - Add the dependency-free evidence inspector and compact real example with integrity/error handling. Surfaced by DX Pass 1 and both voices: first-use deliverable was unspecified. Files: proposed `scripts/inspect_example.py`, `examples/evidence/green-bottles/`, README and inspector tests. Verify: fresh Python/no torch/no network, intact/missing/corrupt/future-version bundles, exact documented output and observed timing.
- [ ] **X2 (P1, human: 3-4d / CC: 1-2d)** - interface contract - Implement shared v1 operations, result/error schemas and explicit automation exit/reconcile behavior. Surfaced by Passes 2/3: execution success can be mistaken for qualification and uncertain POST can duplicate work. Files: API/service schemas, CLI/client journal, MCP adapter and compatibility tests. Verify: full HTTP/CLI sequence, lost response, reused key conflict, strict exit codes, offline policy and no secret output.
- [ ] **X3 (P1, human: 2-3d / CC: 1-2d)** - developer docs - Consolidate compatibility/setup/error/upgrade guides and executable examples. Surfaced by Passes 4-6: conflicting prerequisites and duplicated instructions. Files: README, TESTING, CONTRIBUTING, pyproject/requirements, proposed compatibility/errors/upgrade docs and CI. Verify: clean Windows/Linux tasks, paths with spaces, working docs links and generated examples.
- [ ] **X4 (P2 before redistribution, human: 1-2d / CC: 0.5-1d)** - contribution - Add owner-approved licensing/notices, changelog and safe issue/extension guidance. Surfaced by Passes 5/7: distribution/support assumptions are unestablished. Files: owner-selected LICENSE/NOTICE, CHANGELOG, contribution and issue templates. Verify: owner policy recorded, dependency/model/dataset notices, no private data required for reports.
- [ ] **X5 (P2, human: 1-2d / CC: 0.5-1d)** - DX measurement - Run prerequisite-aware contributor task observations and retain local redacted friction reports. Surfaced by Pass 8: roleplay and targets are not measured onboarding. Files: task protocol/report, compatibility docs and smoke scripts. Verify: actual completion/recovery/timing observations, no invented competitor comparison and no telemetry without opt-in.

Phase 2.5 complete: all eight passes, nine-stage journey, 200-word-class perspective, benchmark, confusion report, checklist and tasks written. Both voices found six concerns and matched 6/6 judgments; amendments are specifications, not executed fixes. Engineering reviews this final amended plan next.

## 10. Engineering review

### Step 0: actual-code scope challenge and built-in search

The expanded roadmap touches more than eight files and more than two service boundaries. Per autoplan, preserve complete scope and implement it as reviewed vertical releases, with conditional integrations still pending UC2; do not turn it into one large refactor. The minimum first change set addresses evidence qualification/coverage and reproducible tests, then typed contract interaction and durable execution. The existing-code reuse map in 1.2 remains authoritative: custom stages, provider router, PyMilvus retrieval, verification ledger and training/MLflow modules are reused.

Source-confirmed findings, with confidence scoped to what was read (no runtime reproduction in this planning turn):

| ID | Severity / confidence | Exact motivating source | Required treatment |
|---|---|---|---|
| E-A1 | P1, 9/10 | `src/api.py:80` `JOBS: Dict[str, Dict[str, Any]] = {}`; `:81` `GPU_BUSY = threading.Lock()`; `:213` `threading.Thread(target=_run_job, args=(job,), daemon=True).start()` | Durable jobs, process boundary and device ownership; current primitives are per-process, not restart recovery. |
| E-A2 | P1, 9/10 | `src/api.py:97` `except Exception:` followed by `return None`; `src/core/context.py:82` `with open(file_path, "w") as f:` | Distinguish absent/corrupt/pending, use atomic artifact finalization and explicit reconciliation. |
| E-Q1 | P1, 9/10 | `src/core/db.py:17` `self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)` | Do not carry one connection into the new API/worker architecture; use per-transaction connections and tested locking. This line alone is not proof of an observed data-corruption incident. |
| E-Q2 | P1, 9/10 | `tests/conftest.py:9` `load_dotenv(Path(__file__).resolve().parent.parent / ".env")` | Hermetic CPU CI must not depend on developer credentials; inject fixtures and deny unexpected outbound calls. No credential leak is claimed to have occurred. |
| E-Q3 | P2, 9/10 | `src/cli.py:5` `from src.pipeline import FrameGenXPipeline`; `pyproject.toml` base dependency list includes torch/diffusers | Decouple help/inspection from ML initialization; define separate supported installation profiles. |
| E-A3 | P1, 9/10 | `benchmark.py:136` `pipeline = FrameGenXPipeline(config_overrides=overrides)`; `src/cli.py:44` `pipeline = FrameGenXPipeline(config_path=args.config, config_overrides=overrides)` | GPU ownership must include CLI/benchmark/training/standalone verification, not merely asynchronous API workers. The later architecture pass must make this explicit. |

Search-before-building results: [FastAPI BackgroundTasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) is for in-process work and its own docs discuss separate tools for heavy computation; swapping a daemon thread for BackgroundTasks does not supply durable GPU scheduling [Layer 1: reuse framework where it fits]. SQL transactions/uniqueness and a small supervised worker are enough for initial local capacity; compare existing built-ins before choosing a queue framework [Layer 1]. [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) can persist graph state, while an in-memory checkpointer cannot survive restart; this does not replace artifact transactions or physical GPU ownership [Layer 1/2]. [PostgreSQL row locking](https://www.postgresql.org/docs/current/sql-select.html) may support a hosted queue claimant; queue-specific lock skipping is not a general consistent-read strategy [Layer 1]. No new Celery/Temporal/agent rewrite is required by this draft.

Distribution is in scope for the inspector/example, typed schema/client artifacts, Python package and Docker image. Source-checkout use comes first; package/container publication follows actual clean builds, notices and owner approval. Existing Docker/Compose are reused and made portable, not declared verified from their presence. No trained weights are automatically published. The previous single initial commit and old review logs do not establish the huge current working tree's CI provenance.

TODOS cross-check: the historical regional-conditioning TODO predates the implemented experimental controller and gains an evidence-scoped status note rather than being blindly marked fully complete. Deferred candidates from CEO/Design/DX are recorded as contextual proposals without rewriting completed historical notes. Independent engineering voices ran sequentially, then Architecture -> Code quality -> Tests -> Performance on the amended scope.

### Independent engineering voices and consensus

#### INDEPENDENT SUBAGENT (eng - source review)

The fresh-context Codex source reviewer returned six findings: five high (common GPU boundary, complete cancellation/finalizer recovery, submission-first identity, promotion policy and per-instance evidence) and one medium (backup payloads). Dimension verdicts: Architecture CONCERN; Tests CONCERN; Performance CONCERN; Security OK for proposed scope; Errors CONCERN; Deployment CONCERN. Full scoped record: [engineering subagent report](../../reports/planning/eng_subagent.md). This is the available Codex substitute for the skill's requested Claude reviewer; no Claude tool or different-model review is claimed.

#### CODEX SAYS (eng - architecture challenge)

The subsequent document-only CLI review returned six findings: five high (GPU ownership/deadlines, recovery/durability, submission identity, semantic evidence binding and local/export security) and one medium (release provenance/distribution). It independently judged all six dimensions CONCERN. [Full verbatim output](../../reports/planning/eng_codex.md) is retained with its frozen input and exit-0 log; it did not read current source or run tests. This supplements a source-reading reviewer without supplying cross-model diversity.

| Dimension | Source subagent | Document-only CLI | Initial consensus / treatment |
|---|---|---|---|
| Architecture sound | CONCERN | CONCERN | CONFIRMED concern; A54-A56 define common ownership and durable identities. |
| Tests sufficient | CONCERN | CONCERN | CONFIRMED concern; A66-A67 map all planned flows and required evals. |
| Performance risks addressed | CONCERN | CONCERN | CONFIRMED concern; A68-A70 bound deadlines, peaks and indexed work. |
| Security threats covered | OK for proposed scope | CONCERN | DISAGREE; missing local/export detail addressed, T9 surfaces the local identity mechanism. |
| Error paths handled | CONCERN | CONCERN | CONFIRMED concern; complete state/recovery rules, typed errors and fault schedules. |
| Deployment manageable | CONCERN | CONCERN | CONFIRMED concern; A57/A65/A71 add complete restore, source/build identity and durability tier. |

Consensus is **5/6 matching judgments, one disagreement**, not five passing implementations. Every substantive issue is treated in the plan; T9 remains a preference choice. Both review instances are Codex. Missing model diversity is explicit, and static agreement is not empirical verification.

### Pass 1: Architecture review

Four principal issues are resolved in the specification by A54-A57: device exclusion, submission identity/order, cancellation/finalization recovery and complete restoration. These are plan fixes; current runtime primitives still need implementation and tests. The source findings in Step 0 and the independent review support high confidence in the structural gaps, not a claim that a production corruption incident was reproduced.

```text
EXISTING React / CLI / future MCP / optional voice-agent inputs
  | generated types + runtime schemas + explicit confirmation
  v
lightweight API / shared command service
  | owner/payload/network/budget checks (hosted identity is a separate gate)
  v
P0 submission repository: UNIQUE(owner, key), original request digest, chosen seed
  | pending admission -> capability/capacity rejection (persisted, no GPU)
  `-> one transaction: admitted outcome + linked queued job
       | SQLite local / PostgreSQL hosted; optional Redis wakeup -> DB scan fallback
       v
P2 supervisor -> common physical-device gate -> one authorized child
  |                ^ API, CLI, benchmark, train, verifier, CUDA preflight all join
  v
EXISTING stages -> generate -> independent verify -> bounded repair -> policy
  | provider router / retrieval are permitted, deadline-bound proposal inputs
  v
temporary immutable attempt -> hash + atomic rename -> guarded terminal DB commit
  | finalizer loss / uncertain commit -> exclusive reconciler -> attach or fail
  v
versioned result + manifest -> UI/CLI/evidence export + redacted trace IDs

backup cutoff -> DB + manifests + EVERY referenced payload -> isolated restore
training/evaluation -> versioned promotion evaluator -> qualified or experimental
```

**Boundaries and coupling.** One application service owns submission, state transitions and recovery policy; HTTP, CLI and tools are adapters. The deterministic stage executor remains the generation owner. Repository adapters implement the same conformance tests for SQLite and later PostgreSQL; Redis is a notification optimization with no job truth. No all-framework dependency is introduced into the base profile. Put inline diagrams beside the proposed job transition service, artifact finalizer/reconciler and device gate, and update existing stage-flow comments when their adapters change.

**Scaling and bottlenecks.** One physical GPU is intentionally serial. API replication is allowed only after removing import-time ML construction and proving a common job repository/device gate. Admission is bounded; API CPU concurrency cannot raise GPU capacity. The local DB, artifact volume and supervisor remain explicit single points of failure with recoverable state, not a fabricated high-availability claim. Hosted multi-worker capacity requires separate physical devices and an approved capacity budget.

**Security.** The existing local singleton profile does not establish hosted tenant isolation. New v1 operations share request limits, allowlisted profiles, owner-scoped lookup/artifacts and permitted provider sets. Validate auth before retaining unauthorized payloads. Retain the current path/UUID artifact protections and test symlink/encoded traversal as well as authorization. Contracts, retrieved documents, tool descriptions and remote messages cannot grant execution privileges or alter acceptance. Read-only MCP comes first; outbound callbacks/voice/agents remain conditional, with exact recipient/region/data permissions. No new security gap was found beyond these already specified boundaries; implementation proof is still required.

**Distribution.** Build CPU inspection/package and frontend assets in clean Windows/Linux CI, record dependency locks, SBOM/notices and source identity; publish only owner-approved artifacts. A separately qualified NVIDIA native/container lane records driver, CUDA, GPU device and base-image digests. Pin supported target architecture initially to the tested Windows/Linux x86_64 profiles; other architectures remain unadvertised until their clean tests pass. Source checkout remains the initial delivery path. A package index/container registry/release destination is an owner choice before publication, not a missing promise hidden by a Dockerfile. No auto-publication, deployment or model-weight upload is authorized by this plan review.

### Pass 2: Code quality review

Five primary issues are specified by A58-A62. They reuse existing models/services rather than add a framework wrapper. Exact source evidence and bounded confidence:

| Issue | Source quote / confidence | Proposed treatment |
|---|---|---|
| CQ1 per-instance evidence loss | `src/verification/contract_evidence.py:30` `key = (kind, str(target).lower(), str(expected).lower())`; `:91` `add("attribute", class_name, attribute, "failed" if contradicted else "unverified",` (9/10 static structure) | Carry entity ID plus stable requirement ID; class-name diagnostics cannot establish individual binding. |
| CQ2 volatile confirmation/cache identity | `src/core/stage.py:118` `provided_contract.setdefault("created_at", datetime.now(timezone.utc).isoformat())`; `src/intent/models.py:19` `id: str = Field(default_factory=lambda: str(uuid.uuid4()))` (9/10) | Return one canonical confirmed revision; do not regenerate defaults after confirmation. Current supplied-contract hashing is not automatically a v1 semantic identity implementation. |
| CQ3 weaker promotion policy | `src/training/registry.py:40` `or report.get("independent_image_groups", 0) < 50 or report.get("seeds", 0) < 3`; `:47` `return {"quality_status": "qualified" if qualified else "unverified",` (9/10) | Preserve provenance checks and upgrade the policy evaluator to all P0 gates; do not rely on old thresholds. |
| CQ4 responsibility/error seams | `src/api.py:73` `pipeline = FrameGenXPipeline()`; `:97-98` `except Exception:` / `return None` (9/10) | Lazy ML service construction, one typed error/result boundary, explicit absent/corrupt/pending readers and thin HTTP/CLI adapters. |
| CQ5 non-hermetic test setup | `tests/conftest.py:9` `load_dotenv(Path(__file__).resolve().parent.parent / ".env")` (9/10) | CPU tests use injected non-secret fixtures, temp storage and outbound denial; opt-in live/GPU tests use separate markers/config. No actual secret leak is asserted. |

**Canonical contract and evidence invariants.** Retain existing Entity IDs when present. A new versioned canonical revision envelope carries a complete predicate list with stable requirement IDs and explicit entity references; legacy ContractV2 is read through an adapter with its narrower evidence scope. First normalization assigns deterministic source-path IDs to missing instances/requirements, rejects duplicate/unknown IDs, and returns the exact canonical object for review. Once assigned, IDs survive editing/reordering and are never regenerated from class name or current list index. Instance attributes remain separate even when two bottles share a color; aggregate counts are explicit derived predicates, not a substitute for the two bindings.

One documented normalization version defines UTF-8 serialization, object-key ordering, numeric forms, absent/null semantics and any semantically unordered sets. Preserve original intent and meaningful requirement values; reject nonfinite numbers and invalid geometry. Exclude audit timestamps from the semantic digest; retain their literal bytes in the artifact hash. Distinguish (1) semantic contract revision hash, (2) original request digest including omitted-versus-explicit seed and permission set, (3) exact artifact-byte hashes and (4) profile/policy/calibration digests. Preview is deterministic and does not generate new UUIDs or wall-clock defaults that change the confirmation hash. Repeating the same canonical preview yields the same digest; changing an obligation produces a new revision. The client sends the returned canonical object, as the amended HTTP example does.

Qualification consumes an exact predicate-ID coverage map bound to the confirmed contract hash, final pixel hash, run/attempt ID, model/profile digest and verifier/policy/calibration versions. Missing, duplicate, extra or mismatched evidence cannot qualify. One-to-one instance assignments are based on recorded observations and the declared matching method, never selected to maximize agreement with expected counts/colors. Ambiguous identity remains unverified; spatial references use entity IDs rather than the first same-class box. Repair clears/recomputes attempt evidence; no earlier image's passing observation may qualify the final image. Legacy `passed`/`is_successful` retain their operational meaning under versioned readers and never retroactively become calibrated full-contract qualification.

**Promotion authority.** Introduce a typed, versioned study report and one authoritative evaluator shared by registry activation, API capabilities, CLI and model-cache reads. Require frozen protocol/calibration/split/source/model/profile identities, independent reviewer/adjudication provenance, per-category tests, accepted-case precision counts/bounds, yield/abstention/deadline measurements, preregistered effect/power decision and artifact-regression outcomes. Human-labeled evidence evaluates the actual selection policy, including repair-selected outputs; calibrating isolated first attempts does not establish the final pipeline's acceptance precision. A positive overall interval with a category regression or missing gate remains experimental. Hash/schema validity establish integrity and compatibility, not the truth of human labels; reports come from the trusted owner/evaluation pipeline, never arbitrary client uploads. Qualification is scoped to its exact policy/model/profile revision; an old report cannot qualify a changed policy. Local experimental opt-in stays explicit and cannot populate a qualified-result cache.

**Maintainability.** Reuse stage registration, repository interfaces, LLMRouter and training validators. Factor canonicalization, coverage/qualification and transition rules into small pure functions with typed outcomes, with side effects in narrow services. Avoid a generic base class for LangChain/CrewAI/ADK and avoid parallel Python/TS schema definitions. One generated schema/type pipeline feeds UI, API docs, CLI JSON and MCP. Exceptions at public boundaries map to the shared redacted error envelope; unexpected failures retain a private incident ID without a false success or raw provider exception. Keep best-effort progress/trace callbacks best-effort, while mandatory evidence writes fail explicitly. Split frontend components by contract editor, job status, requirement evidence and run actions; update flow comments alongside changed behavior.

**Local security disagreement and export specification (A63-A65).** The source subagent marked the already gated security scope OK; the document-only CLI marked it CONCERN. The missing hostile-browser and export boundaries are valid gaps in this draft. T9 recommends the smallest local session/capability mechanism behind the same auth abstraction as hosted mode, preserving offline singleton use; using the full hosted identity module locally is the alternative if it meets offline/resource requirements. Both choices still require exact allowed Origin/Host checks and authorization for expensive/mutating and private evidence routes, including legacy adapters. Do not equate CORS with authentication. Use maintained framework middleware/session/token primitives rather than custom cryptography; [Starlette middleware](https://www.starlette.io/middleware/) and [OWASP CSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html) establish relevant controls, while their fit here is a planning inference. Bind to loopback, reject rebinding-style Host values, validate browser origin, issue a local capability/session through an owner-only bootstrap, keep tokens out of URLs/logs and apply CSRF defenses appropriate to the chosen cookie/header transport. CLI/MCP must present the local capability too. No cloud account becomes mandatory for offline inspection or local generation. A documented security migration may add authentication to legacy routes; legacy compatibility is not a bypass. A compromised OS account remains outside this local protection claim.

Export serializers use an allowlist of versioned IDs, digests, policy outcomes and selected non-sensitive scalar fields. Raw SDK bodies, arbitrary config/environment dumps, keys and unrestricted command arguments are never included. Full local evidence is private by default and exported only through an explicit owner action using safe config references. A redacted derivative omits raw prompts, free-text requirements, audio and pixels by default; users can deliberately include content after reviewing what will leave the machine, but the system cannot infer that every secret embedded in an image or arbitrary prose has been removed. Strip image metadata when deriving an included preview, render untrusted text safely and never mutate the original evidence. Derivatives have their own manifest/hash and an omission list referencing the original immutable bundle; omitted reproduction inputs prevent a claim of complete reproducibility. Leak fixtures verify the declared allowlist, not universal secret detection.

Freeze a reviewable release input manifest containing every execution-relevant tracked and untracked source/config/schema/dependency file selected for delivery, without staging, deleting or rewriting the current tree. Record patch/source hashes, clean build inputs and immutable wheel/sdist/container digests; bind qualification reports to the actual source/build/model/profile identities. Explicit package inclusion rules ship required schemas/docs/example and exclude `.env`, private runs, caches, datasets and credentials. Inspect the built archive, not merely `.gitignore`. A clean isolated install must pass offline inspection and archived readers independently of the developer's caches. No release provenance is claimed from 38eb0f0 alone.

### Pass 3: Test review and complete flow map

**Framework detection and actual reads.** No CLAUDE.md testing authority, Playwright configuration or tracked CI workflow was found. Python uses pytest/FastAPI TestClient; frontend pure-state tests use Node's `node:test` and strict assertions, with no package test script yet. Read the affected API, DB, stage, contract/evidence, provider/retrieval and training source, and actual API, contract-evidence, controlled-integration, training and frontend state tests. Their cases cover useful existing behavior; they do not test a durable v1 service that has not been built. The saved 288-passed/20-warning log is historical bounded evidence. No test suite, clean install, browser journey or GPU experiment ran during this planning review.

Coverage legend: `EXISTING PARTIAL` means relevant current behavior assertions were read; `GAP` means the proposed flow has no complete implemented test. `->E2E` requires an integration/journey across real boundaries; `->EVAL` requires a quality study. These 32 groups are a planning inventory, not a denominator for measured source-line/branch coverage. Every new branch discovered during implementation must be added alongside its behavior; no fabricated 100% percentage is reported.

```text
ENTRY / USER ACTION                  TRANSFORM / STATE / SIDE EFFECT                  COVERAGE
G01 inspect saved example ----------> version/hash/read -> intact | missing | corrupt   GAP, unit+installed smoke
G02 help/install/preflight ---------> profile imports -> CPU | GPU | unavailable         EXISTING PARTIAL -> GAP
G03 choose profile ----------------> caps -> eligible | incompatible | verifier gap     EXISTING PARTIAL -> GAP
G04 text (optional) ----------------> permitted prepare -> draft | refusal | drift       GAP ->EVAL, provider integration
G05 validate supplied draft --------> canonical IDs/hash -> valid | field/bounds error   EXISTING PARTIAL -> GAP
G06 edit/review/confirm ------------> immutable revision+coverage -> save | stale        GAP ->E2E, keyboard/mobile
G07 submit confirmed canonical -----> reserve owner/key -> seed -> durable submission    GAP ->E2E, two DB connections
G08 admission ----------------------> capability/capacity -> reject | pending | job      GAP ->E2E, denominator invariant
G09 lost POST / double-click -------> journal/key lookup -> pending | rejected | job    GAP ->E2E, process crash
G10 any GPU entrypoint -------------> host device gate -> one child | busy | quarantine GAP ->E2E, real two-process race
G11 execute stage pipeline ---------> local/cache -> model | missing | OOM | timeout     EXISTING PARTIAL -> GAP ->E2E
G12 provider call ------------------> permitted set/deadline -> valid | empty | error    EXISTING PARTIAL -> GAP ->EVAL
G13 retrieval ----------------------> local/rerank -> none | selected | permitted web    EXISTING PARTIAL -> GAP ->EVAL
G14 verify/repair loop -------------> same contract, new seed -> bounded retry | stop    EXISTING PARTIAL -> GAP ->EVAL
G15 observation assignment ---------> predicate IDs+final pixel identity -> known | ?    EXISTING PARTIAL -> GAP ->EVAL
G16 acceptance policy --------------> complete calibrated evidence -> qualify | fail | ? EXISTING PARTIAL -> GAP ->EVAL
G17 finalize image/evidence --------> flush/hash/rename/DB -> committed | reconcile      GAP ->E2E, filesystem fault injection
G18 cancel queued/leased/finalizing -> CAS cutoff -> stopped | too late | uncertain      GAP ->E2E, scheduled interleavings
G19 restart/lease/finalizer loss ----> recover/reconcile -> terminal | requeue | quarantine GAP ->E2E, supervisor kill
G20 UI poll/reconnect/navigation ----> ordered status -> render | offline | expired      EXISTING PARTIAL -> GAP ->E2E
G21 history/cache/artifact read -----> owner/version/integrity -> found | absent | broken EXISTING PARTIAL -> GAP ->E2E
G22 export/reproduction ------------> allowlist/redact -> full private | derivative     GAP ->E2E, leak fixtures
G23 local browser / hosted identity -> Origin/Host/token/owner -> allow | deny           EXISTING PARTIAL -> GAP ->E2E
G24 tracing/logs --------------------> redact/buffer -> delivered | dropped/collectordown GAP, integration
G25 data/train/resume ---------------> lineage/splits/device -> checkpoint | reject       EXISTING PARTIAL -> GAP ->EVAL
G26 evaluate/promote/load -----------> typed frozen report -> qualified | experimental   EXISTING PARTIAL -> GAP ->EVAL
G27 package/upgrade/restore ---------> frozen inputs + payloads -> install | safe refusal GAP ->E2E, isolated build/restore
G28 optional voice -----------------> consent/audio -> transcript -> confirm | abort    GAP ->E2E, conditional ->EVAL
G29 optional MCP/n8n/A2A ------------> scoped tool/event -> dedup | deny | bounded retry GAP ->E2E, conditional
G30 optional framework/store spike -> same task/budget -> retain | adopt | reject        GAP ->EVAL, conditional
G31 load/backpressure/deadline ------> indexed bounded queue -> admit | capacity/timeout GAP, DB+process+resource integration
G32 benchmark UI / metrics ---------> scoped cohorts/unknowns -> honest tables/download EXISTING PARTIAL -> GAP ->E2E
```

The following named test contracts define both expected behavior and failure branches. File names prefixed `proposed` are design targets, not existing tests. Extend existing suites when the behavior already has an owner rather than duplicate assertions in every adapter.

| Group | Test location and type | Required meaningful assertions / edge cases |
|---|---|---|
| G01 | proposed `tests/test_evidence_inspector.py`, installed CLI smoke | With no torch/key/network, real example reports saved/unverified and exact identities; missing/changed PNG/JSON, empty manifest, traversal/symlink escape and unknown version fail usefully. No integrity-to-quality upgrade. |
| G02 | proposed `tests/test_install_profiles.py`, Windows/Linux clean CI | Help/inspection do not import torch/router or download; CPU/GPU/training extras resolve separately; paths with spaces, missing config/cache/driver and unsupported versions produce named remedies. Record cold setup separately from inspection time. |
| G03 | extend `tests/test_controlled_integration.py`; proposed capabilities API tests | Incompatible generation blocks before load; missing installed weights block; strict required-verifier outage blocks; allowlisted experimental verifier gap requires acknowledgement and stays unverified. Arbitrary profile paths and stale digests rejected. |
| G04 | extend intent tests plus frozen prompt audit | Nil/blank/Unicode/maximum input, count/color swaps, negation, unsupported predicates, schema-valid omission/invention and refusal; preserve original text, require exact contract review, do not extract again at submit. Compare provider/model revisions on 120 held-out intent pairs. |
| G05 | proposed `tests/test_contract_revision.py`, pure unit/property fixtures | Duplicate/missing IDs, dangling spatial references, reversed/zero/out-of-canvas boxes, negative/huge counts, nonfinite numbers, empty entities, absent/null distinctions; deterministic canonical hash and ID stability after reorder; audit timestamps do not alter semantic hash. Meaningful edits always alter revision identity. |
| G06 | proposed `frontend/tests/contract-workspace.spec.ts`, browser journey | Add/remove individual entity/attribute/relation; retain unsupported requirements; field error focus; stale confirmation conflict; running job keeps old revision. Keyboard-only operation, focus return, announcements, 320px/768px/desktop layouts and WCAG 2.2 AA automated/manual checks. Back/forward and unsaved-draft recovery. |
| G07 | proposed `tests/test_submissions.py`, repository/API integration | Concurrent same owner/key/body gives one submission/seed and at most one job; changed payload conflicts; same key under distinct hosted owners isolated. Crash after key reservation/seed commit/before job link is recoverable. Omitted seed differs from explicit seed request digest and is selected once. |
| G08 | same submission suite, integration | Lost capability/confirmation/capacity rejection response is resolvable without a job; same-key retry preserves original outcome; malformed/unauthorized requests reported separately; every valid unique submission including unsupported contracts remains in the specified denominator. Preview does not count. |
| G09 | proposed client-journal/CLI tests plus API process integration | Persist key/body digest before POST; drop response before/after commit; reopen client and resolve pending/rejected/admitted outcome. Lookup timeout/404 is uncertainty, never license to choose a new key. Two tabs/double clicks reconcile rather than launch twice. |
| G10 | proposed `tests/test_device_ownership.py`, real CPU sentinel processes and NVIDIA lane | Race API/CLI/benchmark/train/verifier across checkouts and DB paths; duplicate supervisor, parent crash with child alive, PID reuse/device remapping, unsupported mixed native/container runtime and failed termination. No second model load; quarantine when stop is unconfirmed. GPU lane confirms actual residency. |
| G11 | extend stage/control tests; one complete local integration | Supplied contract bypasses intent/retrieval/provider; absent optional dependencies cannot trigger imports/downloads. Missing/corrupt weights, OOM, stage failure and total deadline yield execution error, stable ledger and retained evidence. No stale accepted image returned after a later failed attempt. |
| G12 | extend `tests/test_quality_regressions.py` and router tests; proposed provider contract suite | Mock 401/403/404/429/5xx, timeout, refusal, null content, invalid JSON, schema-valid semantic drift, empty vision and cost exhaustion. Retry only allowed transient cases within one deadline; no unpermitted fallback/secret output. Optional live provider eval is isolated and explicitly budgeted. |
| G13 | extend `tests/test_retrieval.py`, retrieval/quality eval | Empty/malformed/stale documents, mismatched embedding version, unavailable store/reranker/grader, injected document instructions and web disabled. Retrieval unavailable is recorded; no secret-bearing exception, contract mutation or implicit outbound request. RAGAS/text metrics stay distinct from image CSR. |
| G14 | extend controlled/stage/repair tests and frozen image eval | At most two repairs under the chosen budget; deterministic distinct recorded seeds; no contract weakening; failed/unknown required observations cannot accept. Verifier/provider outage and insufficient remaining time stop with explicit state; compare first attempt and selected final policy separately. |
| G15 | extend `tests/test_contract_evidence.py`, unit plus labeled fixtures | Repeated same-class/same-color requirements stay separate; swapped attributes, missing/reordered observations, ambiguous assignments and first-box bias remain failed/unverified. Duplicate/extra/missing predicate IDs, wrong contract/attempt/pixel/method hashes cannot qualify. |
| G16 | proposed `tests/test_qualification_policy.py`, unit plus calibrated eval | Empty/partial evidence, uncalibrated method, unknown category, stale calibration/policy, manual download/selection and legacy operational pass never qualify. Valid full exact coverage qualifies only under the qualified policy. Study includes repair-selected outputs and human disagreement/abstention. |
| G17 | proposed `tests/test_artifact_commit.py`, fault/process integration | Inject partial write, permission/disk-full, flush/rename/DB failures and response loss at each boundary. Concurrent reconciliation is idempotent; mismatched lineage cannot attach; missing/corrupt committed image is an integrity error/cache miss. Test platform-specific durability before claiming power-loss resilience. |
| G18 | proposed `tests/test_job_transitions.py`, fake-clock/model schedules + process test | Cancel from every nonterminal state; both cancel-before-commit and commit-before-cancel; repeated/late cancel, DB timeout and worker death while cancelling. No cancelled-to-queued edge or cancelled qualified image. Retain too-late terminal outcome. |
| G19 | proposed `tests/test_job_recovery.py`, real two-process integration | Kill API/worker/supervisor/finalizer/reconciler; expire lease while old child lives; simulate uncertain DB commit and restarted cancellation. Only current recovery authority may adopt exact predecessor bundle; no new GPU work until old process confirmed dead. Operator-required state is visible, not endless silent retry. |
| G20 | extend `runState.test.js`; proposed browser journey | Queued/running/finalizing/reconciling/cancelling/terminal render separately from qualification; old poll responses cannot overwrite newer/terminal state. Offline/reconnect, tab close/reopen, refresh, backward navigation, missing result, expired hosted session and new draft preserve the authoritative job and show a safe action. |
| G21 | extend API tests and DB conformance suite | Zero/one/10,000 records, equal timestamps/concurrent insert, stable pagination, invalid UUID/encoded path/symlink, missing/corrupt/unknown-version bundles and per-owner access. Cache key includes immutable semantic/profile/model/verifier/policy identities; experimental or changed evidence cannot become a qualified hit. Legacy reads remain narrower. |
| G22 | proposed `tests/test_evidence_export.py`, API/download integration | Exact selected image/evidence/manifest agree; full local bundle private by default; redacted derivative has a new manifest and omission list. Plant synthetic secrets in prompt/config/URL/query/exception/EXIF/command args/nested metadata; no leak to exports/logs/UI. Render malicious markup as text and refuse arbitrary paths. |
| G23 | proposed `tests/test_local_access.py`; hosted ownership suite conditional | Allowed local origin/host/session succeeds; hostile origin, rebinding-style Host, missing/invalid local token and unauthenticated expensive/mutating requests fail before GPU admission. CORS tests alone are insufficient. Hosted two-user object/evidence/cancel access, expired/revoked sessions and quota races deny without metadata leakage. |
| G24 | proposed `tests/test_trace_policy.py`, collector fixture | Trace IDs join a failed run without private payloads; nested synthetic secrets removed; slow/down collector and full buffer drop diagnostics within limits and do not alter job qualification. Explicit payload export permission is checked and revocable. No collector needed for core tests. |
| G25 | extend `tests/test_training.py`, preparation/resume fixtures and bounded GPU experiment | Licensed lineage, same-scene family leakage, duplicates/corruption, mislabeled synthetic swaps, tensor/source hashes and no test-label selection. Resume sampler/RNG equals uninterrupted run under declared deterministic profile; device gate excludes generation; NaN/OOM/interruption cannot report qualified/finished inaccurately. |
| G26 | extend registry/adapter-evaluation tests, typed policy fixtures and locked study | Positive overall CI with missing precision/calibration/splits, category regression, underpowered cohort, insufficient accepted cases, malformed bounds or wrong model/source/profile/policy hash cannot promote. Correlated seeds/variants remain in one family. Zero/empty/partial denominators and unresolved labels yield intervals/unavailable; report all frozen cells including failures. |
| G27 | proposed packaging/migration/restore tests, clean isolated integration | Freeze complete source allowlist including relevant untracked changes without altering them; install wheel/sdist/container in empty environments; include schemas/example, exclude private runs/env/cache/credentials. Additive migration reruns/interruption, incompatible API/worker refusal and full DB+payload restore with original volume unavailable. Verify hashes and archived-reader behavior. |
| G28 | conditional proposed voice journey/eval | Permission denied, capture cancelled, unsupported/oversized audio, empty/inaccurate transcript, network loss and microphone state. Editable text fallback and exact contract confirmation before job; no automatic cloning, publishing or notification. Speech error metrics and final image quality reported separately. |
| G29 | conditional MCP/tool/workflow integration | Read scopes first; write tool uses same confirmed submission/idempotency/device policy. Malicious tool/document text, arbitrary paths/shell, unauthorized recipient, invalid/replayed callback signatures, duplicated/out-of-order delivery, remote-agent budget exhaustion and timeout all deny or recover visibly. No real mass notification. |
| G30 | conditional isolated Streamlit/framework/store experiments | Keep baseline LangGraph/Milvus and direct pipeline; compare CrewAI/ADK/LlamaIndex/Chroma/Graph RAG only on named frozen tasks, same time/call budgets and labels. Backend/prototype status semantics agree; unavailable optional package cannot break core imports. Reject additions with no measured value/resource fit. |
| G31 | proposed repository/load/deadline tests, synthetic jobs and NVIDIA resource lane | One active/two waiting, 10x/100x requests, repeated status polling, concurrent reads/writes, index usage and bounded memory/query counts. Total deadline covers queue/provider/retries/repair/finalization/stop; no second launch after watchdog quarantine. Record peak GPU/process/system memory with optional services. |
| G32 | extend stats/state tests; proposed benchmark browser journey | Separate operational count/spatial metric, full CSR, precision, abstention, coverage, intent retention and study version. Null/unknown never renders as zero or perfect; denominator includes rejection/incomplete cells as declared; seed families not independent rows. Downloaded report matches visible cohort and immutable source. |

**Existing behavior retained.** API tests already assert bounded pagination, configured CORS, artifact path constraints, busy-GPU handling, contract-not-silently-ignored, final-image consistency, execution-error distinction and truthful hard-check metric naming. Contract-evidence tests cover color/negation unknowns, unsupported depth, skipped/ambiguous spatial checks, empty extraction and VQA partials. Controlled tests cover invalid sampler before load, immutable supplied contract, local contradiction, missing weights, offline repair and cache separation. Training tests cover split/provenance/tensor hashes, sampler resume, nonfinite metrics, experimental adapter rejection and malformed intervals. Four frontend state tests distinguish rejected/missing/accepted attempts and unavailable VQA. These are behavior tests with useful edges; none substitutes for new cross-process/recovery or human visual evaluation.

**Required eval suites (A67).** Run extraction preservation on held-out original prompts; provider capability/fault/schema/semantic suites with local fixtures plus explicitly permitted live cases; retrieval relevance/injection/offline suites and named-task ablations; exact contract identity through bounded repair; verifier calibration against blinded human labels; equal-budget first-output/final-policy comparisons with GenEval/T2ICompBench-relevant count, binding and relation cases plus local background/reflection hard negatives; training leakage/resume and model-promotion study. Include each claimed predicate and record unsupported ones. Compare against the frozen current baseline, not a changed sampler/budget or a favorite image. RAGAS, trace-tool evaluators, CLIP/reward, denoising loss and tests passing cannot replace full-contract image labels. All providers/models/datasets/licenses must be recorded. Insufficient power, accepted-case count or reliable labels means inconclusive/experimental.

**2am/hostile/chaos paths.** The critical rehearsal starts from a confirmed supplied contract, drops its POST response, resolves the same submission, kills the API, cancels while the worker finalizes, restarts the supervisor with the old child still alive and confirms no second load. Then kill the finalizer after rename and before DB commit, reconcile once under concurrent readers and verify the terminal image/predicate map/manifest agree. In a synthetic directory, fill the artifact volume, corrupt one image, expire a lease and deny termination; recovery must surface an operator-required/quarantined device and never qualify incomplete output. Hosted rehearsal separately attempts a second owner's evidence/cancel URL. These are future controlled tests, not actions taken against user data.

**QA artifact.** The user-facing routes, interactions, edge cases and critical paths are written to `C:/Users/vaish/.gstack/projects/vaishak-v-nair-FrameGenX/vaish-main-eng-review-test-plan-20261015-214500.md`. This is the primary future `/qa` input. Live browser automation remains unavailable under the earlier approval rejection; no Playwright/CDP workaround is authorized. Implementation must obtain an approved browser environment before claiming responsive/a11y/journey validation. Conditional features carry the above tests when their entry gate admits them; mandatory core tests ship with their implementation, not as later debt.

### Pass 4: Performance and resource review

Three performance issues are resolved in the specification by A68-A70: total deadline enforcement, peak resource measurement and bounded indexed access/cache identity. A71 adds a reliability requirement for artifact durability, not a fourth throughput claim. Actual latency/resource targets remain proposed and must be frozen and measured.

1. **One end-to-end clock.** The proposed 600-second job budget begins at durable submission and includes queue wait, provider fallback/retries, model load, verification/repair and finalization. Preparation before confirmation has its own explicit call deadline. Before each stage, check the remaining budget and reserve measured shutdown/finalization time; do not start a repair that cannot fit. Choose the exact stage/stop allowances after measurement and freeze them in the profile. A supervisor watchdog requests cooperative stop, escalates to child termination within the reserved stop allowance, confirms exit or quarantines at the deadline. A non-preemptible CUDA/process failure may exceed the stop allowance: report timeout/operator required, block further device work and never claim termination or a qualified result merely because the clock expired. Queue/cancellation and execution deadlines are separate recorded timestamps within this one public total; the local client wait timeout does not create a fresh server budget. Do not multiply a 30-second timeout by every provider and repair without an outer bound.

2. **Measure worst observed peaks as well as p95.** Keep the proposed 5.0 GB FrameGenX device budget and 1.0 GB physical headroom, <=10 GB process-tree private working set and <=14 GB system committed memory on the 6 GB/16 GB laptop; now require the representative maximum/peak as well as percentiles to fit the frozen budget. Measure the whole device/host process tree, browser and enabled services, including driver/non-torch allocations and training/reranker residency. Use one active GPU workload and at most two waiting initially. Sequential unload/CPU offload must be demonstrated for generator, CLIP, DINO, SAM and optional models; `empty_cache` alone is not proof that another process stopped. Pin retrieval rerankers to CPU unless deliberately admitted through the device owner. No simultaneous training/generation. A headroom preflight/watchdog reduces risk but cannot prevent every transient allocation; fail/OOM evidence remains an explicit disqualification for the tested resource claim. Optional Postgres/Redis/trace/vector-store containers must pass the combined host budget or remain off. Collect the 20-run representative distribution and resource peaks before publishing percentile promises.

3. **DB, cache and I/O bounds.** Current `RunTrackerDB` queries use one shared connection and primary-key schema; current history fetch is bounded and performs a per-row artifact-directory existence check. This is a source-observed access pattern, not a measured production N+1 incident. Use per-transaction repository connections, SQLite WAL/busy-timeout settings tested on the supported local filesystem, and no DB transaction held during GPU/provider/file-copy work. Add unique `(owner_id, request_key)` and submission-to-job indexes, claim indexes over state/queued time, owner/history `(created_at, id)` ordering and cache lookup over immutable revision/profile/model/verifier/policy identities. Use deterministic cursor ordering for v1; preserve bounded legacy pagination. Explain plans and query-count tests cover 10,000 synthetic rows, concurrent inserts and polling. Avoid opening every PNG/JSON for a history page; persist artifact integrity/availability summaries and verify full hashes at the read/cache-use boundary. Keep payloads by reference, event/poll response sizes bounded, queue backpressure explicit, exports streamed with size limits and trace buffers finite. A cache hit is allowed only for the exact declared quality scope and intact bundle; neither a legacy operational pass nor an experimental adapter populates a qualified cache.

**Artifact durability and backups (A71).** For each supported filesystem/runtime, specify payload+manifest write, flush/file sync, same-filesystem rename and required parent metadata durability before terminal commit. Confirm the database's configured transaction durability too. POSIX and Windows flush/rename guarantees differ; a generic `os.replace` is not a power-loss guarantee. Implement/test the platform helper and publish the exact failure tier supported: process crash, OS crash or power loss. If the required durability primitive/profile cannot be established, report the limitation and do not advertise that stronger resilience or qualify it for that delivery profile. Test interruption before/after each durability boundary and full restore of DB plus every referenced payload with original storage absent. The reconciler uses state/token/cancel guards from Pass 1; an intact orphan directory cannot override a completed/cancelled record.

**No premature optimization.** Reuse SQLite/local logs/stages and existing MLflow first. SQL indexes, bounded messages and one GPU owner have identifiable failure/performance reasons; adding Redis, multiple agent frameworks or a second vector store without measured bottlenecks does not. Separate cold cache/download/model initialization, queue wait and warm generation in reports. Quantization/rank/attention/conditioning changes run equal-budget quality/resource ablations before adoption; a faster image that violates more requirements is not an improvement under FrameGenX's metric.

### Engineering failure modes registry

Each row maps to the identically numbered test group in Pass 3. `Gap` means new coverage is required; `Partial` names relevant existing behavior only. All handlers below are proposed unless explicitly described as existing. There are no silent untreated branches intentionally accepted in the new specification; this does not mean the current runtime implements them.

| Group | Realistic production failure | Existing coverage | Planned handler and user-visible outcome |
|---|---|---|---|
| G01 | Packaged example PNG missing or checksum changed | Gap | Inspector exits with exact missing/integrity code and path; no fabricated qualified example. |
| G02 | Help imports unavailable CUDA package | Gap | Lightweight imports and profile-specific remedy; inspect/help still available. |
| G03 | Installed model/verifier differs from confirmed capability | Partial control tests | Revalidate digests; block incompatible/strict unavailable profile or explicitly label allowed experimental gap. |
| G04 | Extraction omits a negation despite valid JSON | Partial intent fixtures | Preserve original intent, review/edit exact draft; held-out intent score separate from image CSR. |
| G05 | Default UUID/time changes confirmation hash | Partial supplied-contract tests | Canonical returned object and versioned semantic identity; stale/invalid revision requires reconfirmation. |
| G06 | User edits draft while an old job runs | Gap | Separate immutable job revision and recoverable draft; field/focus feedback. |
| G07 | Process dies between request reservation and job link | Gap | Pending durable submission scanner; one seed/outcome and at most one job. |
| G08 | Rejection response lost with no job | Gap | Resolve durable submission/key and display persisted rejection without new compute. |
| G09 | Two tabs resubmit after a timeout | Gap | Persist client key/journal; resolve original outcome; 409 on changed payload. |
| G10 | Supervisor dies while CUDA child survives | Gap | Confirm child exit or quarantine physical GPU; no second load and visible operator action. |
| G11 | Generation OOM or missing local weights | Partial stage/control tests | Record execution failure and unverified retained output; release/confirm process resources. |
| G12 | Refusal/empty provider output triggers an unsafe fallback | Partial router regressions | Separate refusal/empty/error codes, permitted providers only, bounded total deadline. |
| G13 | Grader/store unavailable or retrieved text injects instructions | Partial retrieval tests | Record unavailable data, keep context unprivileged, no implicit web/contract mutation. |
| G14 | Repairs exhaust budget or retain earlier pass evidence | Partial stage/repair tests | Stop with failed/unverified result; recompute exact-attempt evidence, preserve contract. |
| G15 | Same-class attributes collapse or bind to wrong object | Partial evidence tests | Stable predicate/entity IDs, declared assignment, unknown for ambiguous instance. |
| G16 | Missing calibration or mismatched pixels appear accepted | Partial unknown/legacy tests | One exact-coverage policy returns unverified/failed; no manual or proxy-score upgrade. |
| G17 | Disk fills after rename or final DB commit response is lost | Gap; current readers conflate some errors | Durability helper and guarded reconciler; explicit integrity/recovery status, no corrupt qualified cache. |
| G18 | Cancel arrives while final commit wins/loses | Gap | Serialized cutoff, stopped/too-late distinction and immutable terminal outcome. |
| G19 | Stale finalizer/reconciler races a newer authority | Gap | Current recovery token/predecessor lineage/cancel guard; quarantine on uncertain live process. |
| G20 | Late poll overwrites a newer terminal UI state | Partial run-state tests | Versioned ordered updates, stable reconnect link and explicit offline/stopping/recovery action. |
| G21 | Old evidence schema/corrupt JSON or cache entry is read as a pass | Partial API/path/legacy tests | Version-aware read, typed corruption, exact quality-scope cache check and bounded history. |
| G22 | Secret embedded in metadata leaves in a bundle | Gap | Allowlisted redacted derivative, explicit content export, new manifest/omission list and safe text rendering. |
| G23 | Hostile webpage triggers costly loopback mutation | Partial CORS/path tests only | Local Host/Origin/session-capability boundary; hosted owner checks separately; deny before admission. |
| G24 | Collector down or secret nested in span attributes | Gap | Bounded non-blocking redaction/buffer/drop counter; local IDs and job still work. |
| G25 | Leakage or resume mismatch yields misleading training success | Partial provenance/resume tests | Reject lineage/split/hash issues; bounded device-owned training, failure status and retained checkpoint. |
| G26 | Positive overall metric hides category regression | Partial adapter registry tests | Versioned all-gates evaluator; report experimental/inconclusive and exact missing gate. |
| G27 | Restore has manifests but no original pixels | Gap | Consistent DB+payload backup and original-volume-absent restore; refuse incompatible release/worker. |
| G28 | Audio transcript changes intended color/count | Conditional gap | Editable transcript/text equivalent and exact confirmation; permission/error states visible. |
| G29 | Remote task/event replays or requests unauthorized action | Conditional gap | Scoped schema/signature/dedup/deadline policy; no arbitrary shell/path/recipient or duplicate side effect. |
| G30 | Optional framework/store adds latency without quality value | Conditional gap | Isolated equal-budget task study; retain baseline/reject module and keep core imports working. |
| G31 | Burst admission/polling consumes RAM or deadline never ends | Gap | Bounded indexed queue/responses, capacity error, total watchdog/stop/quarantine and peak resource gate. |
| G32 | Null CSR becomes perfect or a filtered cohort hides failures | Partial stats/state tests | Explicit scope/unknown/denominator and frozen cohort/download consistency. |

**Critical-gap assessment.** Current structural risks remain open: transient jobs can disappear after restart; artifact corruption can be treated as absence; independent GPU entrypoints lack a common enforceable owner. These are three critical risk classes for the planned reliability guarantees, not three reproduced production incidents. All have explicit test and error/recovery requirements above; no critical branch is intentionally left without a planned treatment. Per-instance qualification and promotion-gate changes also block their respective new qualification release; existing diagnostic unknowns are not evidence that false qualification was observed. Browser, hosted and hardware evidence remain empirical gates.

### Engineering NOT in scope and existing-code reuse

The nine first-release exclusions in Section 4 and technology matrix remain intact, subject to UC2. Additional delivery boundaries: no automatic publication/deployment or trained-weight upload, no untested ARM/macOS support promise, no browser-automation bypass and no shared native/container GPU operation until a central ownership design is tested. These are explicit scope/authorization limits, not dropped entries from the original proposal. Streamlit, CrewAI/ADK, LlamaIndex/Chroma/Graph RAG, voice, n8n/A2A and a dedicated trace backend retain named entry gates and contextual backlog items.

Reuse the current React navigation/design tokens, Pydantic entities, deterministic stage registry, regional controller, router, local evidence diagnostics, repository/archive readers, training provenance/resume/export and MLflow records. Extend their schemas and ownership policies where the current implementation is insufficient; do not reuse class-name deduplication or old promotion thresholds unchanged. Docker/Compose/package files are starting inputs, not proof of portable delivery. Existing tests become regression anchors with original semantics preserved under explicit legacy adapters.

### Worktree parallelization strategy (future implementation)

| Step | Modules touched | Depends on |
|---|---|---|
| A1 canonical contract/evidence/promotion policy | `src/intent`, `src/verification`, `src/training`, schema export, root `tests` | Owner plan decisions and frozen P0 protocol |
| A2 submission/job/device/artifact service | `src/core`, `src/api.py`, proposed `src/jobs`, root `tests` | A1 shared identities; P0 submission foundation before P1 clients |
| B1 typed workspace and frontend journeys | `frontend/src`, `frontend/tests`, generated frontend schemas | A1 exported interfaces; live submit waits for A2 recovery gate |
| C1 study rubric/corpus/protocol | `data/benchmarks`, `reports/benchmarks`, `docs/evaluation` | Owner task/thresholds/licenses; independent of backend implementation after schema agreement |
| D1 inspector/package/CI/docs/restore | `scripts`, `examples`, root packaging/deployment, `.github`, setup docs, root `tests` | A1/A2 and B1 release inputs; combines shared root tests sequentially |
| E1 GPU calibration/comparison/training | evaluation runners, immutable data/reports and one device | Qualified execution/profile identity, C1 labels and a frozen compute/annotation budget |

Lane A: A1 -> A2 sequential, shared core policy and root tests. Lane B: B1 after schema freeze, frontend-only work. Lane C: C1 independent study assets after protocol agreement. Lane D: D1 after the core interfaces settle, with one owner for root tests/setup/packaging. Lane E: all real GPU workloads sequential on one physical owner, including evaluation and training. A+C can start in separate worktrees; B joins after A1, then merge/review before D. CPU work may overlap, GPU runs may not. Root `tests`, schema exports and shared docs are conflict zones: changes there are coordinated/merged sequentially rather than claimed independent. These are five proposed lanes, up to three disjoint CPU lanes, one delivery integration lane and one serial GPU lane; no implementation worktrees were started by this planning review.

### Engineering implementation tasks

- [ ] **E1 (P1, human: 3-5d / CC: 1-3d)** - device ownership - Enforce the common physical-device gate before CUDA across all entrypoints and contain supervised children. Surfaced by both voices/G10: API-only locking cannot enforce the hardware contract. Files: proposed device service, CLI, benchmark, training/preflight, verifier, retrieval device selection and process tests. Verify: cross-checkout cross-entrypoint races, duplicate supervisor, surviving child, stop failure and NVIDIA residency.
- [ ] **E2 (P1, human: 3-4d / CC: 1-2d)** - submission identity - Implement the P0 durable submission/key/digest/seed foundation and rejection-aware lookup before v1 clients. Surfaced by both voices/G07-G09: a rejected submission may have no job. Files: repository/migrations, shared admission service, API/CLI client journal and tests. Verify: response loss, rejected duplicates, reservation/seed/link crashes, changed-body conflict and denominator integrity.
- [ ] **E3 (P1, human: 4-6d / CC: 2-3d)** - recovery - Implement the complete cancel/finalizer/reconciler transition table and guarded durability helpers. Surfaced by both voices/G17-G19: leased/finalizing recovery and cancel cutoff were incomplete. Files: job/supervisor/artifact services, platform durability adapter and fault tests. Verify: both commit/cancel orders, process kill at each boundary, concurrent reconcile, stale lineage, disk full and quarantined device.
- [ ] **E4 (P1, human: 3-5d / CC: 1-3d)** - contract evidence - Implement canonical stable instance/predicate identities and final-pixel/attempt-bound exact coverage. Surfaced by source review and CLI/G04-G06/G15-G16: class-level deduplication and schema-only confirmation lose meaning. Files: intent models/canonicalizer, evidence policy, schema export, UI adapters and regression/eval fixtures. Verify: repeated/swapped/reordered objects, missing/extra evidence, deterministic hashes and no post-confirm extraction.
- [ ] **E5 (P1 for qualification, human: 3-4d / CC: 1-2d)** - promotion policy - Replace the old partial registry gate with one versioned typed all-gates evaluator. Surfaced by source review/G26: positive overall interval does not enforce precision/calibration/category/usefulness gates. Files: training registry/evaluation, shared policy schemas, capability/cache readers and adversarial tests. Verify: each missing gate blocks, exact study/source/model binding, independent final-policy labels and incomplete studies remain experimental.
- [ ] **E6 (P1, human: 3-5d / CC: 1-3d)** - local and export security - Add local session/capability, Host/Origin and legacy-route protection plus allowlisted private/derivative exports. Surfaced by CLI/G22-G23: loopback/CORS and raw bundle promises leave browser/export gaps. Files: middleware/auth abstraction, API/CLI/MCP adapters, export serializers, UI actions and security fixtures. Verify: hostile origins/hosts/tokens, no admission on denial, synthetic nested/pixel metadata secrets and explicit omission manifests.
- [ ] **E7 (P1, human: 3-5d / CC: 1-3d)** - reproducible delivery - Freeze full source/build identities and restore database plus referenced payloads in clean isolated package/container profiles. Surfaced by both voices/G27: manifests alone cannot restore pixels, and HEAD omits working-tree source. Files: packaging/CI/Docker, source inventory, migrations/backup/restore, example assets and installed tests. Verify: wheel/sdist/container contents, original-volume-absent restore, archived readers, incompatible-worker refusal and no private files shipped.
- [ ] **E8 (P1, human: 3-5d / CC: 1-3d)** - test and resource gates - Make CPU tests hermetic and enforce bounded deadline/queue/index/cache policies with recorded peak resource evidence. Surfaced by G02/G12/G24/G31 and Pass 4: .env coupling and p95-only/uncapped retries cannot support the proposed guarantees. Files: test fixtures, repository/query migrations, supervisor deadlines, profile preflight, resource/load harness and CI. Verify: no live secrets/network in CPU tests, query bounds, one active/two waiting, total deadline/stop/quarantine and 20 representative resource observations.

Effort is a planning range, not a promised AI speedup or elapsed benchmark cost. C/D/X tasks overlap these work packages and must be reconciled before implementation; the aggregator retains non-identical tasks rather than silently deleting scope.

### Engineering completion summary

| Required field | Result |
|---|---|
| Step 0 scope challenge | Expanded scope retained as conditional vertical releases per autoplan/UC2; actual source reviewed, not a universal refactor. |
| Architecture | Four principal gaps plus two cross-cutting local-security/release-provenance specifications; dependency diagram and concrete boundaries written. |
| Code quality | Five issues with quoted source and bounded confidence; IDs, hashes, policy authority, service/error seams and hermetic setup specified. |
| Tests | Full diagram, 32 proposed flow groups/gaps with existing partial coverage distinguished; named assertions/evals and external QA artifact written. |
| Performance | Three issues: total budget, peak whole-system resources and bounded indexed data/cache access; durability failure tier also specified. |
| NOT in scope / exists | Written; all original technologies retained in the matrix/conditional backlog; existing modules reused with necessary hardening. |
| TODOS | Six contextual unapproved proposals collected; historical regional-controller status clarified; mandatory P1 accessibility is not deferred. |
| Failure modes | 32 rows mapped to tests/handlers. Three current structural critical risk classes remain until implementation; none intentionally untreated in the new spec. |
| Outside voices | Source Codex subagent then document-only Codex CLI, six findings each; 5/6 matching concern judgments, security disagreement -> T9. No Claude/cross-model claim. |
| Parallelization | Five proposed lanes; up to three disjoint CPU lanes, delivery integration and one serial GPU lane; shared root tests/schema/docs coordinated. |
| Lake Score | 19/19 engineering completeness treatments A54-A62/A64-A71/A73-A74 choose complete handling in the plan; excludes taste A63 and backlog-location A72. This is a planning-choice count, not a quality score. |
| Retrospective | Only initial HEAD 38eb0f0 is committed; saved reports and the large dirty tree are not a prior review-driven delivery history. Revalidate from frozen source/build inputs. |
| Still unverified | All new implementation, clean CI/install, actual browser/a11y, peak resources, recovery and calibrated hallucination reduction. No new tests/GPU/remote deployment ran. |

The 2026-10-16 source-review amendment check marked all six original findings treated but found two consistency gaps in the HTTP authentication example and generic omitted-contract behavior. A73/A74 correct them. The final bounded check returned **PASS, specification only**, with no remaining contradiction in those clauses. This is not a clean runtime or product-quality verdict, and the CLI did not independently reread the final amendments.

Phase 3 complete: source reviewer six initial issues plus two amendment inconsistencies, CLI six concerns; 5/6 initial matching judgments and one disagreement surfaced as T9. Findings have specification treatments, not runtime fixes. Task aggregation and artifact verification follow; owner approval is still pending.

## 11. Cross-phase themes and final review package

These themes appeared in multiple fresh-context source reviews. CLI reviews were sequential and included prior-phase summaries; their agreement is not a fully blind or cross-model replication.

| Theme | Independent phase signals | Plan consequence |
|---|---|---|
| Qualification must describe actual evidence | CEO precision/intent/denominators; Design status authority; DX exit/result meaning; Eng coverage/promotion bindings | Separate operational execution, requirement observations and calibrated qualification everywhere; unknowns never become passes. |
| Contract identity must survive the whole journey | Design editable instances/revisions; DX consistent interface identities; Eng class-level dedup and volatile defaults | One canonical revision, stable predicate/entity IDs and exact final-pixel/attempt evidence. |
| Recovery is a user workflow as well as backend state | CEO interrupted/terminal paths; Design visible actions; DX actionable uncertainty/upgrades; Eng lost submissions/finalizers | Durable submission before job, one device owner, complete cancel/reconcile rules and authenticated examples. |
| Dependencies must fit a reproducible laptop task | CEO hardware/scope; DX clean install/profile friction; Eng GPU entrypoints/peaks/deadlines | Keep existing modules, isolate optional tools, measure combined resources and require a named task before expansion. |
| Delivery needs actual provenance | CEO replayable manifests; DX compatibility/upgrade/docs; Eng release identity and full-payload restore | Freeze source/build inputs, ship only allowed artifacts, restore original-volume-absent and preserve legacy scope. |

### Aggregated implementation tasks

[Implementation task list](../../reports/planning/IMPLEMENTATION_TASKS.md) contains the full aggregated markdown, source findings, estimated effort and intended files; [structured records](../../reports/planning/implementation_tasks.json) retain machine-readable tasks. Four phase JSONLs were written under the external gstack project directory and filtered to run `20261015-framegenx-stack`, branch main and the recent commit window before aggregation. Older July review records on the same initial commit were excluded. Exact deduplication retains **30 tasks: CEO 13, Design 4, DX 5, Engineering 8; P1 23, P2 6, P3 1**. P1 applies to each task's declared release scope; hosted/qualified/optional gates do not become mandatory for a local experimental release.

Non-identical overlapping tasks remain marked for reconciliation, not added together as 30 independent estimates. Work packages include shared qualification/identity, typed workspace, submission/device/recovery/artifacts, clean delivery/docs/restore, provider/tool/tracing experiments and measured training. The proposed QA artifact covers 32 flow groups. Six contextual follow-ups F1-F6 are in `TODOS.md`; mandatory P1 accessibility/tokens/docs measurement stay in the release plan.

### Owner choices presented at the final gate (accepted by A, 2026-10-16)

There are **74 audit entries: 63 mechanical decisions, nine taste choices and two user challenges**. A25 is retained as a superseded initial location choice; A72 explicitly records its reversal. These are planning decisions, not 74 product changes. The plan had unusually high ambiguity (nine taste choices). The user selected A and accepted every recommendation below on 2026-10-16; the table preserves the alternatives for audit. Conditional audience/budget/rights/resource decisions are still required at their activation gates.

| Item | Recommendation | Viable alternative / downstream effect |
|---|---|---|
| UC1, CEO | Replace “everything works perfectly” with a scoped experimental-quality baseline. | You may mean execution/UI stability only; preserving that narrower statement must not imply calibrated image accuracy. If we are wrong, extra calibration delays work you already consider adequate. |
| UC2, CEO | Adopt in stages, with one runtime owner per concern and optional tools admitted by named-task evidence. | Keep the complete stack as a learning/research laboratory; this adds setup/memory/maintenance and slower quality attribution. If broad framework exploration is the goal, staged product gates may be too restrictive. |
| T1, CEO | Durable local refresh/recovery before live v1 submission. | Minimal legacy-only patch; v1 stays inactive with narrower documented recovery guarantees. |
| T2, CEO | Add private/full and explicitly redacted derivative evidence export after schema/version gates. | Defer portable export; keep local inspection until collaboration demand exists. |
| T3, CEO | Voice and remote workflow entry gates after the core journey. | Prioritize a named accessibility/integration slice sooner, with the same privacy/resource/confirmation gates. |
| T4, CEO | MLflow/local logs first; compare one dedicated trace candidate, with Phoenix a local candidate. | Select LangSmith or Opik based on the measured task/privacy/operations comparison; no parallel trace truth stores. |
| T5, CEO | Preserve integrated Generate -> Verify -> Repair -> Accept; test verifier-only alternative D. | Focus on verification over external generators; changes the product boundary and onboarding, not acceptance integrity. |
| T6, CEO | Freeze usefulness/latency/effect thresholds from observed owner/user tasks before study. | Keep a research-only goal without commercial-demand gating; still preregister technical thresholds and quality evidence. |
| T7, Design | Inline contract review within Studio. | Guided stepper for complex contracts; preserve revision, recovery and keyboard behavior. |
| T8, DX | Explicit profile and offline-by-default v1 automation, chosen seed recorded. | Prompt-first permitted-cloud preparation; requires provider permission, budget and exact contract confirmation before submit. |
| T9, Engineering | Small offline local session/capability boundary using the shared auth abstraction. | Reuse the full hosted identity module locally if it passes offline/resource requirements; more setup and identity scope. Both retain Host/Origin/CSRF and private-export controls. |

UC1 is an evidence/claim challenge, and UC2 includes feasibility risk under the 6 GB/16 GB hardware contract. Both review instances support staged, measurable treatment, but they may be missing your distinction between a stable local tool and a qualified model, or a goal of learning many frameworks. Neither challenge silently replaces your direction. Budget, licenses, data residency, intended audience, final task thresholds and publication destinations remain explicit conditional activation decisions rather than invented current requirements.

### Pre-gate verification

All four phase outputs are present: CEO's 11 review sections/strategic alternatives/error and failure registries/dream/reuse/completion, Design's seven scored passes/litmus/actions/a11y specification, DX's eight scored passes/persona/empathy/journey/TTHW/checklist, and Engineering's actual-source scope/four passes/diagrams/32-flow test artifact/failures/reuse/non-scope/completion. Independent source and document review records and consensus tables exist for each phase, with same-family and scope limits disclosed. Cross-phase themes, audit decisions, task JSONLs/aggregation and contextual backlog are written. Required artifact/format checks are recorded in `reports/planning/pre_gate_verification.json`; these check the planning package, not the application.

The user selected A on 2026-10-16; phase logs now record that roadmap approval with runtime/empirical gaps retained. The existing 2026-07-17 clean entries are stale and do not clear this dirty tree for delivery. No commit/push, source cleanup, application dependency installation, GPU workload or deployment occurred during the original planning review. The separately requested gstack tooling repair and its bounded side effects are recorded in `reports/autoplan_tooling_setup_2026-10-15.log`. Subsequent execution is tracked independently; this plan is not a runtime test report.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|---|---|---|---|---|---|
| CEO | `/autoplan` Phase 1 | Scope, premises and strategy | 1 phase; 2 source/spec rounds, 1 successful CLI | REVIEWED, approval pending | Final spec recheck 9/10; initial voices 4/6 matching judgments, two strategic disagreements included. |
| Design | Phase 2 | Typed contract/evidence workspace | 1 phase; source + CLI | REVIEWED, browser proof pending | Primary planning score 8.7/10; 7/7 matching litmus judgments, not seven visual passes. |
| DX | Phase 2.5 | Setup, asynchronous API/CLI and recovery | 1 phase; source + CLI | REVIEWED, execution pending | Primary planning score 8.75/10; initial 6/6 matching concerns; new TTHW not measured. |
| Engineering | Phase 3 | Architecture, code quality, tests and performance | 1 phase; source initial + 2 bounded checks, CLI | REVIEWED WITH CONCERNS, approval pending | Six findings per initial voice, 5/6 matching judgments, T9 disagreement; two consistency fixes rechecked PASS for specification only. |
| Independent CLI | All four phases | Adversarial document challenges | 4 successful frozen-document runs | COMPLETE, document-only | Same Codex family; no independent current-source/runtime verification by CLI. |

**CODEX:** Findings absorbed as reviewable plan amendments; raw outputs/frozen inputs retained. Fresh-context source reviewers supplied independent source reads; there was no Claude or cross-model review. CEO's initial source-reading CLI attempt stalled and was stopped before successful document-only review.

**VERDICT:** APPROVED ROADMAP, DONE_WITH_CONCERNS — user A accepts UC1/UC2 and T1-T9. No product-runtime, visual-quality, calibrated hallucination-reduction, hosted-security, clean-install or CI qualification is granted. Implementation and its tests proceed through the separate execution ledger; this is not ship clearance.

All eleven gate choices are resolved by the user's A. Future conditional budget/license/hosting/provider decisions are not silently approved by this acceptance. New research execution has its own evidence and compatibility gates.

NO UNRESOLVED DECISIONS
