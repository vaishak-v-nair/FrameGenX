# Engineering independent source review

Fresh-context Codex subagent `/root/eng_review`; read-only source and plan review. No commands, tests, GPU workloads or browser interaction executed. Same model family as the CLI voice; not a Claude or cross-model review.

## Dimension verdicts before amendment

Architecture CONCERN; tests CONCERN; performance CONCERN; security OK for proposed scope; error paths CONCERN; deployment CONCERN.

## Findings

1. **High, confidence 9/10: GPU exclusion across entrypoints.** One API supervisor does not protect direct CLI (`src/cli.py:44`), benchmark (`benchmark.py:136`), training (`src/training/train.py:216-217`) and standalone verifier (`src/verification/verify_stage.py:145`). Every entrypoint must acquire common ownership before loading a GPU model or initializing CUDA. Test two-process cross-entrypoint contention and duplicate supervisors.
2. **High, 10/10: finalization/cancellation recovery.** The plan's transitions omit cancellation during leased/finalizing, worker death while finalizing and recovery from reconciliation_required. Define a cancellation linearization point, finalizer lease renewal and reconciler ownership/current token/cancel checks. Test both cancel/commit orders and kill after finalizing.
3. **High, 9/10: ledger-only rejection reconciliation.** Valid capability rejections have no job, so job-only request-key lookup cannot recover their lost responses. Submission must be the first durable owner/key/request identity, with persisted outcome and optional job. Commit omitted seed once, reconcile pending/rejected/linked states and implement the foundation before dependent clients.
4. **High, 9/10: promotion registry weaker than new gates.** `src/training/registry.py:39-47` can qualify using at least 50 independent groups, three seeds and a positive interval; the plan adds calibration, accepted-case precision, protocol/split identities and category non-inferiority. Use one versioned typed policy evaluator; missing evidence stays experimental. Adversarial reports with positive overall intervals must fail when those new gates are absent or regressed.
5. **High, 9/10: per-instance evidence identity.** `src/verification/contract_evidence.py:30` deduplicates `(kind, target, expected)` and attributes target class names at lines 74/91. Repeated same-class instances and swapped attributes cannot be faithfully bound. Carry stable entity/requirement IDs through canonical contract, observations, manifest and UI; preserve each obligation and return unverified for ambiguous assignment. Test reordered/swapped/missing observations.
6. **Medium, 8/10: incomplete backup recipe.** Database plus manifests cannot restore generated pixels, masks and observations after volume loss. Back up every referenced immutable payload at a declared cutoff; restore into an empty directory with original storage unavailable and verify hashes/readability.

All six are addressed as engineering specifications in the amended plan. Runtime implementation and verification remain open gates.

## Amendment check, 2026-10-16

The same source reviewer confirmed complete specification treatments for all six original findings: common pre-CUDA gate, complete cancel/finalizer transitions, submission-first identity, all-gates promotion policy, per-instance evidence and DB-plus-payload restoration. The new predicate/pixel/policy binding, deadline/durability and release-source manifest clauses were otherwise consistent.

Two remaining contradictions were found: the local HTTP example omitted authentication, and a generic omitted-contract paragraph conflicted with exact-contract-only v1 submission. A73 adds owner-only local bootstrap and bearer headers on each call with browser CSRF transport distinguished; A74 scopes parsing to preparation/legacy and makes absent v1 contract a 422. The final bounded source check returned PASS for these clauses, with no remaining contradiction. No implementation or runtime certification is implied.
