# System Architecture

Prompt
↓
Contract
↓
Scene Graph
↓
Generation
↓
Verification
↓
Repair
↓
Acceptance

## Modules

### Intent Engine
Translates ambiguous natural language into a strict, versioned JSON contract.

### Scene Planner
Calculates 2D spatial layouts, bounding boxes, and geometric relationships from the contract.

### Generator
The "Latent Drafter." A dumb rendering engine (SDXL/Flux) conditioned strictly on the output of the Scene Planner.

### Verifier
Independently verifies the generated image against the initial contract using specialized Vision-Language Models (Florence-2) and Object Detectors (GroundingDINO).

### Reporter
Generates itemized discrepancy reports (JSON) outlining exactly which constraints failed.

### Benchmark Engine
Automated test suite that measures the Constraint Satisfaction Rate across a standardized dataset of complex prompts.

### Repair Engine
Autonomous agent that reads the verification report, adjusts the Scene Graph or Contract, and triggers a re-generation loop without human intervention.
