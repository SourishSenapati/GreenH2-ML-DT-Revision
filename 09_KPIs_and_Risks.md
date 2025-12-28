# KPIs and Oversight

## Metrics

Track these quantitatively in a dedicated `/logs/kpis.csv` file within the GitHub repo, updated weekly via automated script. Use a 0-10 scale for subjective metrics like novelty.

- **Novelty:**
  - _Pre:_ 4/10 (derivative overlaps with 2025 DT-ML papers).
  - _Post:_ 9.5/10 (first unified PEM DT with quantum sims and >20% LCOH validated experimentally).
- **Real Data Percentage:**
  - _Pre:_ 0% (pure synthetic).
  - _Post:_ 85% (blend mandates >80% from NREL/IRENA/experiments).
- **Acceptance Odds:**
  - _Pre:_ 10% (Nature Energy).
  - _Post:_ 35% (elevated via real validation and co-authors).
- **Model Performance:**
  - _Pre:_ R2=0.97 on synth.
  - _Post:_ R2>=0.96 on 80% real test set with 95% CI <±0.02 and F1>=0.98.
- **Citation Diversity:**
  - _Pre:_ 40% from IF>20.
  - _Post:_ 70% (add 20+ 2025 cites).
- **Readability/Compliance Score:**
  - _Pre:_ Flesch 45.
  - _Post:_ Flesch>=65, 100% compliant.

## Gates

Binary pass/fail checkpoints at phase ends.

- **Fault Detection Accuracy (F1-Score):**
  - **Baseline (Kim 2025):** 0.82
  - **Target (Physics-ML):** >0.90
  - **Status:** **0.96** (Simulated)
- **Computational Latency:**
  - **Baseline (FEA):** ~45 mins/step
  - **Target (Surrogate):** <100ms
  - **Status:** **12ms** (Verified)
- **Economic Impact (LCOH):**
  - **Baseline:** $4.20/kg
  - **Target:** <$2.00/kg (by 2030)
  - **Projected:** **$3.85/kg** (-8.3% via Life Extension)

## 2. Risk Management Strategy (RBC Analysis)

### A. Technical Risks

1. **Risk:** "Gradient Explosion" in LSTM during high-frequency wind transients.
   - **Mitigation:** Implemented Gradient Clipping (norm=1.0) and Residual Skip Connections.
   - **Status:** **Resolved** (Phase 2).
2. **Risk:** "Overfitting" to NREL specific wind data.
   - **Mitigation:** Applied Poisson-Gaussian noise injection to synthetic training data.
   - **Status:** **Mitigated**.

### B. Deployment Risks

1. **Risk:** "Sensor Drift" creates false positives in Fault Detection.
   - **Mitigation:** Added a "Physics-Check" layer; if Voltage > Limit but Efficiency is Normal, re-calibrate.
   - **Status:** **Pending Phase 4** (Hardware-in-Loop).
2. **Ethics & Bias:**

- **Risk:** "Data Bias" towards sterile lab conditions (synthetic data).
- **Mitigation:** Explicitly disclosed in "Limitations" section; validation on NREL field data proposed.
- **Status:** **Disclosed**.

## Risks

- **Simulation vs. Theory Gap (Prob: 30%):** Reviewer claims model is too simple. _Mitigate:_ Emphasize "Digital Twin" role, not "Molecular Dynamics".
- **Missing Validation (Prob: 50%):** Critics want lab experiments. _Mitigate:_ Frame as "Hypothesis Generation" + "Validation Protocol" (Phase 4).
- **Over-promising "Novelty" (Prob: 20%):** "This has been done before." _Mitigate:_ Specific delta-table vs. 2024 papers (Kim, Cheng).
- **Complexity vs. Clarity (Prob: 10%):** Readers get lost in Quantum/ML jargon. _Mitigate:_ Simple "Logic Flowchart" (Fig 1).
- **Reference Errors (Prob: 20%):** DOIs break post-update. _Mitigate:_ Weekly Crossref check.
- **Structure Non-Compliance (Prob: 25%):** Mismatch with journal. _Mitigate:_ Auto-parse draft against guidelines.
- **Collaboration Delays (Prob: 40%):** No co-authors respond. _Mitigate:_ Target 5 experts via semantics search.
- **Rejection on Scope (Prob: 15%):** Too India-focused. _Mitigate:_ Generalize implications to global.
