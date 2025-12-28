# KPIs and Oversight

## Metrics

Track these quantitatively in a dedicated `/logs/kpis.csv` file within the GitHub repo, updated weekly via automated script. Use a 0-10 scale for subjective metrics like novelty.

*   **Novelty:**
    *   *Pre:* 4/10 (derivative overlaps with 2025 DT-ML papers).
    *   *Post:* 9.5/10 (first unified PEM DT with quantum sims and >20% LCOH validated experimentally).
*   **Real Data Percentage:**
    *   *Pre:* 0% (pure synthetic).
    *   *Post:* 85% (blend mandates >80% from NREL/IRENA/experiments).
*   **Acceptance Odds:**
    *   *Pre:* 10% (Nature Energy).
    *   *Post:* 35% (elevated via real validation and co-authors).
*   **Model Performance:**
    *   *Pre:* R2=0.97 on synth.
    *   *Post:* R2>=0.96 on 80% real test set with 95% CI <±0.02 and F1>=0.98.
*   **Citation Diversity:**
    *   *Pre:* 40% from IF>20.
    *   *Post:* 70% (add 20+ 2025 cites).
*   **Readability/Compliance Score:**
    *   *Pre:* Flesch 45.
    *   *Post:* Flesch>=65, 100% compliant.

## Gates

Binary pass/fail checkpoints at phase ends.

*   **Pass if:** Real data >80%.
*   **Pass if:** Novelty delta >25% vs. priors.
*   **Pass if:** Compliance 100%.
*   **Pass if:** Model Robustness >95% (CV R2).
*   **Pass if:** Impact Projections Validated (LCOH drop >22%).
*   **Pass if:** Ethics Coverage Complete.

## Risks

Top risks ranked by severity (1=high).

1.  **Data Scarcity (Prob: 45%):** NREL access denied. *Mitigate:* Fallback to IRENA/ACS open datasets.
2.  **Novelty Doubts (Prob: 50%):** Reviewers flag as incremental. *Mitigate:* Add unique quantum sim via pyscf.
3.  **Ethics Flags (Prob: 30%):** Overclaims or bias undetected. *Mitigate:* Add explicit caveats.
4.  **Overfit/Invalid Metrics (Prob: 35%):** High R2 on synth doesn't hold. *Mitigate:* Re-run with SHAP and bootstrap CI.
5.  **Reference Errors (Prob: 20%):** DOIs break post-update. *Mitigate:* Weekly Crossref check.
6.  **Structure Non-Compliance (Prob: 25%):** Mismatch with journal. *Mitigate:* Auto-parse draft against guidelines.
7.  **Collaboration Delays (Prob: 40%):** No co-authors respond. *Mitigate:* Target 5 experts via semantics search.
8.  **Rejection on Scope (Prob: 15%):** Too India-focused. *Mitigate:* Generalize implications to global.
