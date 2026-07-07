# Verification Specification

## 1. Verification Architecture
The verification layer acts as the compiler's test suite for visual outputs. It takes two inputs: the `generated_image.png` and the `contract_v2.json`. Its sole purpose is to independently prove or disprove that the image satisfies the contract.

To prevent bias, the verification layer **must not** share the same foundational architecture as the generation layer.

## 2. Detection Mechanisms

### A. Spatial and Numeracy Verification
- **Primary Tool**: GroundingDINO (Open-Vocabulary Object Detection).
- **Process**: The system queries GroundingDINO using the `class_name` of every entity in the contract.
- **Validation**: 
  - Count validation: Does the number of detected bounding boxes match the number of declared entities?
  - Spatial validation: Do the relative positions (IoU, centroids) of the detected bounding boxes match the spatial relationships defined in the contract?

### B. Semantic and Attribute Verification
- **Primary Tool**: Florence-2 (Vision-Language Foundation Model) or CLIP.
- **Process**: The system crops the generated image using the bounding boxes identified by GroundingDINO. It then evaluates the cropped region against the entity's specific `attributes` (e.g., "red", "metallic").
- **Validation**: Prevents attribute leakage (prompt bleed) by ensuring the red color is isolated to the car and not the house.

### C. Text Verification
- **Primary Tool**: EasyOCR or Florence-2 OCR.
- **Process**: Scans the image for explicit text regions and computes the Levenshtein distance against the `explicit_text` defined in the contract.

## 3. Scoring Methodology
The verifier outputs a binary Pass/Fail based on calculated confidence thresholds, alongside a detailed JSON discrepancy report.

- **Constraint Satisfaction Rate (CSR)**: A binary score (1 or 0) for the overall image.
- **Component Scores**:
  - `object_presence_score`: Are all entities present?
  - `attribute_fidelity_score`: Are all attributes applied to the correct entities?
  - `spatial_accuracy_score`: Do the relationships hold true?

## 4. Confidence Thresholds
Because vision models (like GroundingDINO) are not perfect, strict thresholds must be tuned.
- **Object Detection Confidence Threshold**: Set to 0.45. If an object is detected with < 0.45 confidence, it is considered missing.
- **Attribute Matching Threshold (CLIP)**: Cosine similarity > 28.0 is required to confirm an attribute is present.
- **Failure Condition**: If *any* critical constraint falls below the threshold, the entire image is rejected and routed to the autonomous repair loop.
