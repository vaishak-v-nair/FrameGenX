---
name: framegenx-guidance
description: Core architectural and behavioral rules for all FrameGenX agents.
---
# FrameGenX Skill

You are working on FrameGenX.

Before writing code:

1. Read AGENTS.md
2. Read architecture.md
3. Read research.md

Always optimize for:

- reproducibility
- benchmarkability
- modularity
- verification

Never optimize for:

- aesthetics only
- marketing claims
- unsupported assumptions

When proposing features:

Ask:

Can it be measured?

Can it be benchmarked?

Can it be reproduced?

Can it run on 6GB VRAM?

If not, reject the proposal.

All code must:

- be modular
- include tests
- support local execution
- save intermediate artifacts

Treat contracts as source-of-truth.

Treat prompts as generated artifacts.

FrameGenX is a verification-first system.
