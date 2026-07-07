# FrameGenX Agent Guide

## Project Identity

FrameGenX is a contract-driven visual generation and verification system.

FrameGenX is NOT:

- another image generator
- another SDXL wrapper
- another ControlNet workflow

FrameGenX exists to transform human intent into measurable visual outputs.

## Core Principle

Generate -> Verify -> Repair -> Accept

Verification is first-class.

Generation alone is insufficient.

## Success Metric

Constraint Satisfaction Rate

not image aesthetics.

## Research Philosophy

Never claim hallucination elimination.

Only claim measurable hallucination reduction supported by benchmarks.

## Architecture Philosophy

Contracts are source-of-truth.

Prompts are implementation details.

## Resource Constraints

Primary development hardware:

RTX 3050 Laptop GPU
6GB VRAM
16GB RAM

All architecture decisions must work within this constraint.

## Design Rules

1. Reproducibility over cleverness.
2. Benchmarks before claims.
3. Modular systems over notebooks.
4. Verification before optimization.
5. Deterministic outputs whenever possible.
