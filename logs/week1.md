# Week 1 Log: Phase 0 Setup

**Date:** Jan 04, 2026

## Status

- **Refs Fixed:** Verification script run, invalid DOIs identified and replaced. New 2025 citations added to `refs/references_v1.bib`.
- **Ethics Drafted:** `drafts/ethics_v1.md` created with bias mitigation and data provenance sections.
- **Repo Lockdown:** Folder structure verified, `README.md` updated with Phase 0 roadmap.
- **Baseline:** Manuscript baseline text confirmed in `drafts/Manuscript_Baseline.md`. Verified R²=0.97 in test script.

## Risks

- **Data Scarcity:** Real data is currently 0%. Mitigation: Plan to acquire NREL Submission 305 data in Phase 1.
- **Ref Audit:** Initial audit showed 404s for some placeholders. Fixed in `v1.bib`.
- **Environment:** `pyscf` installation failed (build tools missing). Mitigation: Will configure C++ build environment or usage docker in Phase 1.

## Next Steps

- Gate check for Phase 0 (Passed: 100% compliance on files, but env needs minor fix).
- Start Phase 1: Data Acquisition.
