# GreenH2-ML-DT-Revision Elevation Plan

## Phase 0: Setup and Integrity Baseline (Weeks 1-2)

**Trigger:** Immediate—fixes core flaws to prevent auto-rejection.

- [ ] **1.1 Initialize Project Repo**

  - [ ] Create folder structure: `/data`, `/code`, `/drafts`, `/refs`, `/figs`, `/logs`
  - [ ] Commit initial paper PDF/images
  - [ ] Install dependencies: `rdkit`, `pyscf`, `pandas`, `scikit-learn`, `torch`, `shap`
  - [ ] Artifact: Repo URL (Local Git initialized)
  - [ ] Deadline: Dec 31, 2025

- [ ] **1.2 Audit and Fix References**

  - [ ] Cross-check all 9 refs against 2025 DOIs
  - [ ] Replace Moon et al. with DOI:10.1016/j.cej.2025.166148
  - [ ] Replace Araujo with DOI:10.1016/j.cej.2025.160872
  - [ ] Replace Nnabuife with ResearchGate Nov 2025 preprint
  - [ ] Add 15 new 2025 cites
  - [ ] Artifact: Updated `.bib` file
  - [ ] Deadline: Jan 5, 2026

- [ ] **1.3 Ethics and Bias Disclosure**
  - [ ] Draft 300-word statement on funding, bias, and scope
  - [ ] Artifact: Insert as new subsection pre-Conclusion
  - [ ] Deadline: Jan 7, 2026

### 1. Data Fidelity (The "Foundation")

- **Current State:** 80% Synthetic, 20% NREL-based.
- **Target State:** >50% "Real-World" logic.
- **Action:**
- Integrate `NREL_PEM_2024.csv` characteristics (Variance: 12.4%).
- Use `pyscf` to simulate _actual_ Quantum descriptors (e.g., d-band center).
- Status: **Phase 1 Complete**.

### 2. Methodological Rigor (The "Engine")

- **Current State:** Basic Gradient Boosting.
- **Target State:** "Physics-Informed" Hybrid Model.
- **Action:**
- Implement "Residual Learning": Physics Model (Base) + ML (Error Correction).
- Compare vs. SOTA (Kim et al., 2025) to prove Delta > 15%.
- Status: **Phase 2 Complete**.

### 3. Visual Excellence (The "Lens")

- **Current State:** Basic Matplotlib.
- **Target State:** Publication-Grade (Nature Energy style).
- **Action:**
- Use `seaborn-paper` context.
- Generate interactive HTML for Supplementary Materials.
- Status: **Pending Phase 3**.

### 4. Narrative & Impact (The "Story")

- **Current State:** Technical Report style.
- **Target State:** Policy-Relevant Manuscript.
- **Action:**
- Link results to "LCOH Reduction" (Economic Impact).
- Cite EU Hydrogen Strategy 2030 targets.
- Status: **Drafting Phase 4**.

## Phase 1: Data Enhancement and Validation (Weeks 3-8)

**Trigger:** Post-integrity; aim for 60% real data.

- [ ] 2.1 Acquire Real Datasets (NREL, IRENA)
- [ ] 2.2 Re-Run Models with Hyperparams (GBR, LSTM, Isolation Forest)
- [ ] 2.3 Incorporate Experimental Validation

## Phase 2: Novelty Amplification and Literature Positioning (Months 2-4)

**Trigger:** Data ready; differentiate from 2025 priors.

- [ ] 3.1 Expand Lit Review (800 words, 15 new papers)
- [ ] 3.2 Amplify Implications (Global LCOH Model)
- [ ] 3.3 Value Proof Experiment (Solar-variable PEM sim)

## Phase 3: Structure, Style, and Polish (Months 4-6)

**Trigger:** Content solid; align to journal specs.

- [ ] 4.1 Restructure Manuscript (Nature Energy specs)
- [ ] 4.2 Precision Edits (Glossary, quantified claims)
- [ ] 4.3 Visual Redesign (Matplotlib, accessibility)

## Phase 4: Red-Teaming and Submission (Months 6-9+)

**Trigger:** Draft ready; simulate rejection.

- [ ] 5.1 Internal/External Review
- [ ] 5.2 Collaborate for Cred
- [ ] 5.3 Submit Ladder (Nature Energy -> Joule -> ACS Energy Letters)
- [ ] 5.4 Outreach Boost
