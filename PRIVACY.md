# Privacy & GDPR Notes

**Scope:** This repository provides tooling and guidelines for privacy-preserving video analysis.

- **Data Minimization:** Process only what's necessary. Prefer on-device inference.
- **Consent:** Obtain explicit consent for identifiable footage. Otherwise, use `privacy/face_blur.py` or avoid storage.
- **Retention:** Define short retention periods; default to deleting derived frames when not needed.
- **Anonymization:** Remove/obscure faces and identifying metadata where feasible.
- **Data Subject Rights:** Enable deletion and access upon request (track via `governance/logs/` if you persist outputs).
- **DPIA:** For high-risk deployments, perform a Data Protection Impact Assessment.
- **Cross-border Transfers:** Follow local regulations and standard contractual clauses.
- **Children’s Data:** Avoid unless strictly necessary and legally compliant.
