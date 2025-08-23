# TMRT — Token-Manifold Red Teaming: Competition Submission Writeup (Draft)

This repository provides the TMRT framework submission for the OpenAI GPT-OSS-20b Red‑Teaming Challenge.

Contents in this dataset:
- `kaggle_submission.json` — primary submission manifest used for the competition
- `issues/` — per-issue JSON files (one finding per file), matching the competition `findings.schema`

Summary of the submission
- 11 findings (tmrt_001 .. tmrt_011)
- Techniques: Unicode exploits, scaffolding, embedding-optimization, steganography, metamorphic prompts
- Validation: reproducibility scripts + Docker environment included in the repository

How to reproduce
1. Clone the repository and activate the provided venv.
2. Run `tools/prepare_issues.py` to regenerate per-issue JSONs.
3. See `FINAL_SUBMISSION_PACKAGE/submission_files/kaggle_submission.json` for the combined submission manifest.

Notes
- This is a draft writeup prepared for the competition's Writeups section. Please refine the executive summary and attach demonstration notebooks before publishing to Kaggle Writeups.

License: CC0-1.0
