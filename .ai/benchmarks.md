# Benchmark Framework

## Categories
- **Object Count**: Prompts requesting exact numbers of objects.
- **Spatial Relations**: Prompts requiring strict positional logic (e.g., inside, under, left of).
- **Color Constraints**: Testing attribute isolation and prompt bleed.
- **Text Rendering**: Verifying spelling and legibility of embedded text.
- **Complex Scenes**: High entity count with overlapping relationships.

## Metrics
- **Constraint Satisfaction Rate**: % of images passing all Hard Constraints.
- **Repair Success Rate**: % of failed images successfully fixed by the Autonomous Repair Loop.
- **Generation Time**: End-to-end latency per verified image.
- **VRAM Usage**: Peak VRAM consumed (must stay under 6GB).
