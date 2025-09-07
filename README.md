# Visual Data Trustworthiness Scoring with Ethical Annotations

## Problem Statement
This project aims to develop an ethical, privacy-preserving video trustworthiness scoring system capable of detecting both deepfake content and other forms of video manipulation such as compression artifacts, frame inconsistencies, and misleading framing. The system will provide both visual (e.g., attention heatmaps) and textual explanations to help users understand why certain frames or segments may be untrustworthy. The design prioritizes fairness, transparency, and human oversight, ensuring that the system avoids harmful false positives and biases while remaining computationally feasible on modest hardware setups such as RTX 3050 GPUs. The project will follow responsible AI guidelines, including privacy protection, GDPR compliance, and ethical annotation protocols, to create a trustworthy and socially impactful solution for video verification.


## What this repo provides (v0.1)
- Lightweight baseline pipeline (OpenCV + NumPy) that analyzes videos and outputs:
  - Frame/segment trustworthiness scores.
  - Visual "attention" overlays (edge-based saliency as a placeholder).
  - Textual explanations describing why segments may be flagged.
- Ethical, privacy, and governance scaffolding: model card, data card, GDPR notes, annotation guide, risk register.
- Fairness hooks: subgroup-aware error/fairness metrics (requires user-provided metadata).
- Computation-conscious: runs on CPU; can be extended to GPU (RTX 3050) easily.
- CI/lint/test scaffolding; pre-commit config; Dockerfile; Makefile.



## Responsible AI, Privacy & GDPR
- Face blurring utility included for consent-unknown footage (`src/vdt_scoring/privacy/face_blur.py`).
- Logging avoids PII by default (`src/vdt_scoring/governance/logging.py`).
- See `PRIVACY.md`, `ETHICS.md`, `ANNOTATION_GUIDE.md`, `MODEL_CARD.md`, `DATA_CARD.md` and `RISK_REGISTER.md`.


