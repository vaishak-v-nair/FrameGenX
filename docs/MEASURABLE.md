# Measurable Metrics

To ensure FrameGenX remains a rigorous engineering project, every goal must be strictly quantifiable. If it cannot be measured, it is not a project goal.

## 1. Constraint Satisfaction Rate (CSR)
The primary metric of success. The percentage of generated images that successfully pass the independent verification layer for all constraints defined in the JSON contract, without human review.

## 2. Specific Accuracy Metrics
- **Object Count Accuracy**: The percentage of generations where the number of requested objects exactly matches the number of objects detected by the Verifier (e.g., GroundingDINO). This is the core metric for the MVP.
- **Spatial Accuracy**: The percentage of generations where the relative bounding boxes (IoU, centroid relationships) of detected objects match the strict geometric constraints defined in the contract.
- **Color/Attribute Accuracy**: The percentage of generations where specific attributes (verified via CLIP or Florence-2) are correctly isolated to their target entities without leaking to adjacent entities (prompt bleed).
- **OCR Accuracy**: The Levenshtein distance between requested explicit text and the text detected in the final image via EasyOCR.

## 3. System Performance Metrics
- **Repair Success Rate**: The percentage of initially failed generations that are successfully corrected after autonomous repair loops.
- **Average Repair Loops**: The mean number of iterations required to achieve a passing CSR.
- **Generation Cost**: The compute time (Time To Completion) required to produce a verified, successful image.
- **Peak VRAM Usage**: Maximum GPU memory consumed during the sequential pipeline execution (crucial for maintaining the 6GB VRAM constraint).
