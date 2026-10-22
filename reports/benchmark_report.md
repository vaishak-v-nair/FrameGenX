# FrameGenX — Detailed Benchmark Report

Generated from `runs/framegenx.db`

## 1. Ablation Summary

| Config | N | CSR | CLIP_norm | VQA | Reward | Avg Attempts | Avg Latency (s) |
|--------|---|-----|-----------|-----|--------|--------------|-----------------|
| Baseline | 15 | 53% | 0.522 | 0.860 | 0.691 | 2.40 | 197.5 |
| + RAG | 15 | 40% | 0.538 | 0.913 | 0.725 | 3.20 | 277.2 |
| + RAG + LoRA | 15 | 73% | 0.554 | 0.923 | 0.739 | 2.60 | 239.3 |
| Full (RAG+LoRA+Reflect) | 15 | 33% | 0.601 | 0.917 | 0.759 | 2.53 | 229.0 |

## 2. Per-Category CSR Breakdown

| Category | Baseline | + RAG | + RAG + LoRA | Full (RAG+LoRA+Reflect) |
|----------|------|------|------|------|
| color_constraints | 80% (5) | 80% (5) | 100% (5) | 40% (5) |
| object_count | 40% (10) | 20% (10) | 60% (10) | 30% (10) |

## 3. Reward Distribution

### Baseline
- N: 15
- Mean: 0.6912
- Std: 0.1300
- Min: 0.5189
- Max: 0.9315
- Distribution:
  0.5 | ███ (3)
  0.6 | ███ (3)
  0.7 | ██ (2)
  0.8 | ██████ (6)
  0.9 | █ (1)

### + RAG
- N: 15
- Mean: 0.7255
- Std: 0.1243
- Min: 0.4133
- Max: 0.8528
- Distribution:
  0.4 | █ (1)
  0.5 | █ (1)
  0.6 | █ (1)
  0.7 | ███ (3)
  0.8 | ███████ (7)
  0.9 | ██ (2)

### + RAG + LoRA
- N: 15
- Mean: 0.7387
- Std: 0.1063
- Min: 0.5208
- Max: 0.8739
- Distribution:
  0.5 | █ (1)
  0.6 | ██ (2)
  0.7 | ███ (3)
  0.8 | ██████ (6)
  0.9 | ███ (3)

### Full (RAG+LoRA+Reflect)
- N: 12
- Mean: 0.7591
- Std: 0.1127
- Min: 0.5228
- Max: 0.8684
- Distribution:
  0.5 | █ (1)
  0.6 | █ (1)
  0.7 | ██ (2)
  0.8 | █████ (5)
  0.9 | ███ (3)

## 4. Attempt Distribution

| Config | 1 attempt | 2 attempts | 3 attempts | 4 attempts |
|--------|-----------|------------|------------|------------|
| Baseline | 8 | 0 | 0 | 7 |
| + RAG | 2 | 3 | 0 | 10 |
| + RAG + LoRA | 6 | 1 | 1 | 7 |
| Full (RAG+LoRA+Reflect) | 6 | 2 | 0 | 7 |

## 5. Failed Prompts (Full Config)

**10 of 15 prompts failed:**

- `two cats sitting on a wall` (R=0.745, category=object_count)
- `three red apples on a table` (R=0.840, category=object_count)
- `five books stacked on a desk` (R=0.000, category=object_count)
- `two cars parked in a driveway` (R=0.000, category=object_count)
- `two dogs playing in a park` (R=0.000, category=object_count)
- `four candles on a birthday cake` (R=0.868, category=object_count)
- `three cups on a kitchen counter` (R=0.684, category=object_count)
- `a red car parked next to a blue bicycle` (R=0.568, category=color_constraints)
- `a person wearing a yellow jacket and green pants` (R=0.788, category=color_constraints)
- `a silver laptop on a wooden desk` (R=0.523, category=color_constraints)
