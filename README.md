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

> ⚠️ **Note**: The included detector is a *heuristic baseline* for demonstration only. Replace `src/vdt_scoring/models/*` with trained models (e.g., PyTorch) for production.

## Quickstart
```bash
# 1) Create & activate a virtual env (Python 3.10+ recommended)
python -m venv .venv
source ./.venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Generate a synthetic test video
python examples/make_synthetic_video.py --out examples/synthetic.mp4

# 4) Run the CLI on a video
python cli.py --video examples/synthetic.mp4 --out runs/example_run

# 5) Inspect outputs
# - runs/example_run/results.json          (scores & explanations)
# - runs/example_run/heatmaps/frame_XXX.png (visual overlays)
```

## Git Setup (step-by-step)
```bash
git init
git add .
git commit -m "feat: initial scaffold for ethical video trustworthiness scoring"
# If you use GitHub CLI:
# gh repo create vdt-ethical --public --source=. --remote=origin --push
# Otherwise create a repo on GitHub manually, then:
# git remote add origin https://github.com/<you>/vdt-ethical.git
# git push -u origin main
```

## Architecture (baseline)
```
video -> frame sampler -> heuristics (edge + motion + blur/blocks proxy)
     -> per-frame/segment scores -> calibration (placeholder)
     -> explanations: visual overlays + textual reasons
     -> JSON report + saved artifacts
```
To use learned models, replace heuristics with CNN/Transformer backbones and plug in explainers (e.g., Grad-CAM via Captum).

## Responsible AI, Privacy & GDPR
- Face blurring utility included for consent-unknown footage (`src/vdt_scoring/privacy/face_blur.py`).
- Logging avoids PII by default (`src/vdt_scoring/governance/logging.py`).
- See `PRIVACY.md`, `ETHICS.md`, `ANNOTATION_GUIDE.md`, `MODEL_CARD.md`, `DATA_CARD.md` and `RISK_REGISTER.md`.

## Roadmap
- [ ] Replace heuristics with trained deepfake/manipulation models.
- [ ] Add Grad-CAM style visual explanations for trained models.
- [ ] Expand fairness metrics & calibration.
- [ ] Add streaming inference.
- [ ] Ship GPU Docker image with CUDA (optional).
