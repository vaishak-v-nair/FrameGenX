Document-only review. Source descriptions remain unverified; all fixes and tests below are proposed.

| Dimension | Judgment | Reason |
|---|---|---|
| Architecture sound | CONCERN | Physical device ownership and submission identity remain incomplete across entry points. |
| Test coverage sufficient | CONCERN | Recovery tests lack several required crash schedules and semantic qualification invariants. |
| Performance risks addressed | CONCERN | Resource budgets lack enforceable deadlines and coverage of competing launch paths. |
| Security threats covered | CONCERN | Local browser access and portable evidence need explicit security boundaries. |
| Error paths handled | CONCERN | Reconciliation, cancellation and rejected-submission recovery have unresolved outcomes. |
| Deployment risk manageable | CONCERN | Release identity and distribution contents are insufficiently specified for the dirty source baseline. |

1. **HIGH — GPU exclusivity is a stated invariant without a complete enforcement boundary.**  
   **Plan:** §10 E-A3 says ownership must include “CLI/benchmark/training/standalone verification”; P2 specifies “exactly one supervisor/worker to each physical GPU.” The plan does not settle whether those other launch paths submit to that supervisor or acquire some separate lock. Two supervisors using different databases, containers or checkouts can each believe they own the same device. Process-level p95 memory also cannot protect against a brief allocation spike or another launcher.

   **Fix:** Specify one host-level ownership mechanism keyed by physical GPU identity, shared by every supported GPU entry point. Define supervisor crash recovery, child-process containment, container/native coordination and fail-closed behavior when ownership cannot be established. Make the total job deadline enforceable through cancellation and process termination; distinguish queue, execution and shutdown deadlines.

   **Tests:** Race API, CLI, benchmark and training launches across two checkouts and native/container profiles. Kill the supervisor while its child survives; simulate failed termination and GPU identity remapping. Verify no second launch, bounded shutdown and quarantine. Measure peak device/system memory as well as percentiles.

2. **HIGH — Crash recovery and cancellation do not form a complete state machine.**  
   **Plan:** §4.4 permits `finalizing -> reconciliation_required`, but supplies no transitions out of reconciliation. Worker-loss recovery covers `leased/running`; cancellation is specified only from `running`. “Cancel near completion” promises both schedules without defining whether finalizing can still be cancelled.

   **Fix:** Define an explicit cancellation cutoff and recovery transitions for every nonterminal state, including `cancel_requested`, `finalizing` and reconciliation. Specify who resumes reconciliation, how its ownership is fenced, and how process loss during cancellation reaches a terminal outcome. Define required file/directory durability operations per supported platform; atomic rename alone does not guarantee survival after power loss.

   **Tests:** Crash before/after file flush, rename, DB commit and response; crash during cancellation and reconciliation. Repeat reconciliation concurrently. Assert one authoritative terminal outcome, no stuck jobs, no cancelled result becoming qualified and no completed reference to undurable evidence.

3. **HIGH — Idempotency cannot currently reconcile submissions that never create jobs.**  
   **Plan:** P0 records valid capability rejections in a ledger “with optional linked job ID”; DX reconciles uncertain requests through `GET /jobs?request_key=...`. P2 places uniqueness on owner/key, but does not locate that reservation transaction relative to ledger insertion, eligibility checks, seed assignment and job creation.

   **Fix:** Make the durable submission the idempotency authority. Reserve owner/key and canonical request digest atomically before nondeterministic work; expose resolution returning pending, rejected or job-linked submissions. Define crash recovery between each stage and consistent key behavior across stale confirmations, capacity rejection and eligibility rejection. Keep validation previews outside the denominator.

   **Tests:** Concurrent duplicates, lost rejection responses, crash after key reservation, crash after ledger insertion and omitted-seed retries. Assert one submission, at most one job, stable rejection/seed, changed-content conflict and correct denominator accounting.

4. **HIGH — Semantic qualification needs a stronger evidence-binding contract.**  
   **Plan:** P3 says “reject semantic contract drift even when JSON is syntactically valid”; P0 requires full requirement coverage. Neither specifies the invariant proving that evidence evaluates every confirmed predicate against the final image. The v1 submit operation also allows “permitted intent” with confirmed hashes without defining the extraction-to-confirmation protocol.

   **Fix:** Require qualification to consume a complete predicate-ID mapping bound to contract hash, final pixel hash, attempt, verifier configuration and policy/calibration version. Reject duplicate, missing, extra or mismatched evidence. Define separate intent preparation and exact-contract confirmation so extraction cannot silently occur after approval. Calibration must cover the actual decision policy, including bounded repair and candidate selection.

   **Tests:** Schema-valid omitted predicates, swapped entity bindings, duplicate evidence, evidence from an earlier attempt, modified final pixels, empty observations and stale calibration identities. Verify abstention or typed failure, never qualification. Exercise repair-selected outputs in the independent acceptance study.

5. **HIGH — Local security and bundle redaction remain underspecified.**  
   **Plan:** §4.3 acknowledges “loopback access … is not user authentication,” but most access protections activate only for hosting. Evidence export promises redacted metadata and checksums without specifying treatment of credentials embedded in prompts, provider bodies, URLs, image metadata or reproducibility commands.

   **Fix:** Define the local browser threat model, including hostile websites, Origin/Host validation, DNS rebinding and protection for expensive or mutating routes. Specify an allowlisted export schema, secret-reference handling and safe rendering of untrusted text. Give redacted derivatives their own manifests and explicitly identify omitted reproducibility inputs.

   **Tests:** Cross-origin localhost requests, rebinding-style Host headers and unauthorized local mutations. Plant secret fixtures in nested metadata, URLs, exceptions, image metadata and command arguments; inspect every exported/logged surface. Verify redaction preserves readable evidence without claiming complete reproduction.

6. **MEDIUM — Upgrade compatibility does not establish release provenance or safe distribution.**  
   **Plan:** Snapshot is “38eb0f0 plus existing working-tree changes”; §10 acknowledges extensive untracked provenance risk. The plan mentions release provenance and notices but does not define the exact distributable source, artifact allowlist or binding between tested binaries and qualification evidence.

   **Fix:** Require a reviewable source snapshot or patch manifest covering all execution-relevant changes, clean isolated package/container builds, explicit inclusion rules and immutable release digests. Bind qualification reports to those digests and model/profile identities. Publish only after the stated license/notices gate; preserve upgrade fixtures independently of developer caches.

   **Tests:** Build and install wheel, source archive and container from the frozen release input. Verify required examples/schemas are included, private runs/configuration/secrets are excluded, and installed artifacts pass offline inspection and archived-reader checks. Confirm unsupported API/worker combinations refuse work before claiming the GPU.