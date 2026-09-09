# Blockers and decision boundaries

- Three category CSVs are available in `data/`. Verified collection metadata is still required: source provenance, category/query definitions, weekly interval boundaries, timezone, availability, and scale compatibility. The configured Sunday-start UTC calendar is provisional.
- The normal approved build requires a recorded review of the fixed 2023 coastal-island diagnostic. The existing user-requested demand-v2-provisional export did not grant that approval. Unverified cross-market comparability blocks approval; stable leadership requires an explicit review decision. Do not manufacture ranking changes.
- Optional economic records are not supplied. Their absence does not block the tourism-only core once diagnostic prerequisites pass.
- The source application code referenced by the project context is not present in this workspace. Verify its live request and availability behavior before integration.
- PyTorch, Google Colab GPU, Weights & Biases tracking, and Google Drive artifacts are selected. The Colab notebook and runner are implemented; real-data training still requires the documented reviews. Experiment package versions are pinned in training/requirements.txt. Runtime setup, external Python/CUDA/hardware verification, the exact Drive folder/owner/access/retention, persistence verification, and production deployment remain unresolved. See docs/training.md.
- A data owner must confirm source permissions, Google Trends collection consistency, privacy constraints, retention, and dataset versioning before training begins.
- Actual weekly or monthly arrivals or booking data is not available as a confirmed target. Google Trends remains a proxy until a better target is approved.
- Heavy training must run outside the local IDE after the external environment and credentials are configured.
- Production deployment, access control, retention policy, and human review requirements remain product-owner decisions.
- Never commit secrets, large datasets, model weights, checkpoints, tracker exports, or generated artifacts. Google Drive is the selected destination, but the exact external artifact location is not configured. Access to artifacts/demand-v2-provisional/ failed during documentation review; verify its saved vocabulary and schema before execution.
