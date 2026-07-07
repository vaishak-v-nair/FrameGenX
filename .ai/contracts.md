# Contract Specification

Contract > Prompt

## Contract v2 Structure

- **Subject**: The overarching theme or subject of the scene.
- **Objects**: Explicit list of all distinct entities required.
- **Counts**: Exact integer counts for specific object classes.
- **Colors**: Hex codes or explicit string attributes tightly bound to specific objects.
- **Spatial Relations**: Strict 2D geometric constraints (bounding boxes, z-index, occlusion rules).
- **Text Requirements**: Exact strings that must be rendered via OCR verification.
- **Style Constraints**: Photography style, lighting, lens types.
- **Hard Constraints**: Rules that must not be broken (e.g., "Must contain exactly 3 apples"). Failure triggers rejection.
- **Soft Constraints**: Preferences that do not trigger rejection if missed (e.g., "Golden hour lighting preferred").
