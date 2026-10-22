DX review: **DONE_WITH_CONCERNS**. This assesses the supplied plan only. Repository descriptions, source findings and timings remain unverified by me. Scores measure planning completeness, not implemented behavior. Engineering remains pending and must review the final amended plan last.

| DX dimension | Judgment | Reason |
|---|---|---|
| Getting started <5 minutes, honest prerequisites | CONCERN | Evidence inspection has a bounded target, but its packaging and entrypoint are unspecified. Cold GPU setup correctly has no five-minute promise. |
| Guessable API/CLI naming and defaults | CONCERN | Shared identities are planned; public command names, request fields, defaults and automation behavior remain incomplete. |
| Actionable errors | CONCERN | Failure categories exist, but a shared public error envelope, exit codes and recovery examples are missing. |
| Findable, complete documentation | CONCERN | Conflicting prerequisites are identified without a defined canonical documentation route or complete first integration example. |
| Safe upgrades | CONCERN | Additive migration and historical semantics are specified; executable compatibility, preflight and downgrade rules are not. |
| Development-environment friction | CONCERN | Optional services are separated conceptually, but lightweight dependency boundaries and Windows contributor setup need concrete contracts. |

| Planning dimension | Score /10 |
|---|---:|
| Getting started | 7 |
| API/CLI/SDK | 6 |
| Errors/debugging | 7 |
| Documentation | 6 |
| Upgrade/migration | 7 |
| Environment/tooling | 6 |
| Community/ecosystem | 5 |
| Measurement/feedback | 8 |
| **Mean** | **6.5** |

Six gaps should become explicit DX acceptance criteria before the Engineering review.

1. **HIGH: The first useful moment lacks a deliverable contract.**  
   “Inspect evidence without importing torch” does not establish how a fresh contributor reaches that experience. The real example also needs defined completeness, provenance and expected inspection output.

   **Fix:** Specify a bundled, versioned real example containing the image, confirmed contract, evidence, manifest and checksums. Define one proposed inspection entrypoint, its supported Python version, dependency boundary and expected output. It must distinguish recorded execution from current qualification and distinguish bundle integrity from image correctness. Missing files, checksum failure and unsupported bundle versions need examples. Preserve the experimental result rather than constructing a successful demonstration. Measure first inspection separately from cloning and installing Python; make no fresh-generation claim from replay.

2. **HIGH: CLI/API parity lacks a public interaction contract.**  
   Matching contract/profile/seed hashes does not define what an automation developer sends, receives or does after an uncertain submission. The plan also leaves default network access, waiting behavior and command success ambiguous.

   **Fix:** Freeze a small public operation table for validation, capabilities, submission, status, result and evidence inspection. Specify required fields, default profile selection, seed behavior, provider permissions, confirmation binding and idempotency-key lifecycle. Include one complete proposed CLI and HTTP sequence covering submit → lost response → reconcile → poll → inspect evidence. Define machine-readable stdout, diagnostic stderr and exit-code semantics, separating execution success from qualification. No additional SDK is required.

3. **HIGH: Error taxonomy is not yet an actionable error interface.**  
   Exception classes and sanitized incident IDs help implementation, but users still lack consistent remedies. “Retry only when safe” needs a server-owned representation.

   **Fix:** Define one versioned error envelope shared by API and CLI, with stable `code`, readable `message`, field path where relevant, correlation IDs, retryability and a documented recovery action. Specify how CLI exit codes map to it. Add complete examples for invalid relation references, unavailable profiles, idempotency conflicts, GPU OOM, corrupt evidence and quarantined workers. Distinguish retrying a read, reconciling an uncertain submission and explicitly creating a new linked job. Document which actions require a local operator.

4. **MEDIUM: Documentation has no defined authority or contributor route.**  
   Correcting individual version statements will not prevent README, TESTING, CONTRIBUTING and architecture guides from drifting again. API documentation generated from schemas also cannot explain the complete asynchronous workflow.

   **Fix:** Designate a canonical tested compatibility table and require other setup pages to reference it. Define a short documentation map: inspect evidence; prepare local generation; integrate through CLI/API; debug/recover; contribute; upgrade. Include separate PowerShell and Linux examples, expected outputs and prerequisite/network labels. Add one minimal verifier-extension example covering registration, unknown evidence and qualification tests. Assign owner decisions for project licensing and contribution terms without inferring a license.

5. **HIGH: Upgrade safety stops before the user-facing compatibility contract.**  
   Additive columns do not establish whether an older application can read newer records, whether workers can overlap versions, or what happens when an archived schema is unsupported.

   **Fix:** Publish a compatibility matrix spanning application, API, database, contract and evidence versions. Define migration preflight, backup location, supported upgrade paths, mixed API/worker-version policy and rollback limits. Require a representative old-run fixture and migration walkthrough showing preserved IDs, hashes and archived verdict semantics. Specify actionable refusal for unsupported future schemas. Establish a deprecation policy for legacy `/generate` before changing its behavior.

6. **MEDIUM: Environment setup and DX measurement remain disconnected.**  
   A conceptual optional-dependency matrix does not prove evidence inspection avoids ML imports or that contributors can run useful checks without CUDA. Simulated confusion timestamps cannot establish onboarding performance.

   **Fix:** Specify minimal evidence-inspection, CPU-contributor and GPU-generation dependency profiles, including tested Python/Node versions, lock strategy and explicit model-download/cache behavior. Require clean Windows validation, including paths with spaces, and checks for disk, driver and cache readiness before generation. Measure observed onboarding separately for evidence inspection, CPU contribution and warm/cold generation. Record prerequisite state, task completion, errors and time without exporting private prompts. Compare future revisions against that baseline; competitor onboarding time remains unknown.