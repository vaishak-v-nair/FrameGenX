# Independent DX review

Source: fresh-context Codex subagent `/root/dx_review`; static document and selected README/TESTING/CONTRIBUTING/CLI/package source. No execution or new timing measurement.

Six consensus dimensions all initially CONCERN: first use <5 minutes, API/CLI semantics, actionable errors, documentation findability, upgrade safety and environment friction.

Planning scores (/10): getting started 7; API/CLI 6; debugging 7; docs 6; migration 7; environment 5; community 4; measurement 8.

Findings:

1. HIGH: define a standard-library evidence inspection entrypoint and packaged bundle; current CLI imports the pipeline before parsing, and default package dependencies include ML.
2. HIGH: resolve Python/torch/CUDA/Node/shell support and clean-install instructions; README, CONTRIBUTING and pyproject conflict, and PowerShell `&&` assumes a shell version.
3. HIGH: define shared machine-readable execution/qualification output and compatible strict-exit behavior; current exit zero can accompany unverified evidence.
4. MEDIUM: publish stable error envelopes with code/problem/cause/fix/docs, retryability and incident IDs; missing config/contract/cache handling needs context.
5. MEDIUM: consolidate the README/TESTING setup loop, owner-specific paths and broken AGENTS link. Keep redistribution licensing as an explicit owner gate.
6. MEDIUM: define supported API/evidence versions, deprecation notice, migration/downgrade behavior and archived fixtures.

These identify planning omissions. Existing source gaps are not represented as fixed by writing the plan.
