# Ethical Annotation Protocol

- **Annotator Safety:** Avoid graphic/traumatic content where possible; provide support resources.
- **Consent & Privacy:** Use licensed or consented data; blur faces if unsure.
- **Label Taxonomy:** 
  - `deepfake` (face/voice swap, synthesis), 
  - `compression_artifact`, 
  - `frame_inconsistency` (duplication, jitter, temporal cuts), 
  - `misleading_framing/context`.
- **Quality Control:** Double annotation + adjudication for disagreements.
- **Bias Controls:** Balance sources across demographics; audit label rates & errors per subgroup.
- **Documentation:** Keep dataset cards up to date (`DATA_CARD.md`). Track provenance for every sample.
