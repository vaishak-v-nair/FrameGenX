# Multiplayer AI scope review

Owner wording: "Option: A. multiplayer AI". Two prior A choices were pending:
Human review inside Benchmarks and generated-file Git cleanup. A clarification
has been sent before changing either. The word multiplayer also needs a product
boundary: AI agent collaboration, shared human workspaces, or both.

## Proposed first slice if the owner means AI agents collaborating

Retain the contract and deterministic pipeline as the authority. Separate roles
can propose work, but agreement between language models is not visual evidence.

| Role | Input | Allowed output | Boundary |
| --- | --- | --- | --- |
| Intent compiler | User request | Typed draft contract and ambiguities | User reviews the contract; requirements cannot disappear |
| Scene planner | Frozen contract | Instance layout and supported generation plan | No change to counts, attributes or constraints |
| Generator | Plan and chosen profile | Image and exact generation identity | One device lease, bounded attempts |
| Independent verifier | Final image and contract | Predicate observations and unknowns | Different visual model families; no self-certification |
| Repair critic | Failed/unknown predicates and evidence | Supported repair proposal | Cannot relax the frozen contract to manufacture a pass |
| Evidence checker | Attempt identities and observations | Complete/incomplete/corrupt evidence state | Acceptance requires the declared coverage and calibration gates |

The existing pipeline already separates several of these responsibilities.
First extend its typed interfaces and bounded recovery. Introduce a graph
orchestrator only for an observed branching/recovery need, with one runtime owner;
installing several competing agent frameworks does not improve image accuracy.

On the 6 GB laptop, model roles share one physical GPU sequentially. Parallel
read-only CPU work can be considered separately within the 16 GB host budget.
No new cloud model calls, paid judges or third-party messages are authorized by
this wording. Provider permission, redaction, quotas and the contract-review
approval remain explicit gates.

The UI should show the requirement being checked, the observation and who or
what supplied it. Agent names or avatars must not imply independent calibration.

## If the owner means multiple people

Shared workspaces need separate user identity, owner/project authorization,
versioned edits, reviewer attribution and conflicts, durable jobs, quotas,
PostgreSQL migrations and tested recovery. The local in-memory job API is not a
multi-user service. Do not expose it publicly or add invitation/send actions
before the approved hosted/authentication gates are implemented and tested.

## If the owner means both

First ship the local evidence/review and durable execution foundations, then
define which humans can approve each agent's proposed action. One person's
approval must not grant another account access to private prompts or exports.
The current download and optimization feasibility study continues independently;
it provides no multi-user security or multi-agent quality certification.
