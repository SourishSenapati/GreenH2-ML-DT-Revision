# Ethics and Integrity Disclosure

## 1. Conflict of Interest

The authors declare no competing financial interests. This research was self-funded by the Department of Chemical Engineering, Jadavpur University. The funding sources had no role in the study design, data collection, analysis, decision to publish, or preparation of the manuscript.

## 2. Scope and Limitations (PEM vs. Alkaline)

Our Digital Twin framework is parameterized explicitly for **Proton Exchange Membrane (PEM)** electrolyzers. We caution against direct transferability to Alkaline systems without recalibration of the degradation constants. We assume that approximately 80% of the active mitigation logic (e.g., current ramping protocols) remains conceptually valid, but specific kinetic parameters are unique to PEM chemistry.

## 3. Geographic Bias and Mitigation

This study references the **Indian National Green Hydrogen Mission (NGHM)** (approx. 15% of the text) as a primary case study. To ensure global relevance and mitigate regional bias, we have drawn parallel comparisons with **European Union (EU)** green hydrogen strategies and **IRENA 2025** cost projections [12]. The techno-economic conclusions are normalized to USD to facilitate international applicability.

## 4. Machine Learning Risks

We acknowledge the risk of overfitting inherent in high-degree-of-freedom models.

- **Overfitting Risk:** We observed a 5% generalization gap between training and validation sets.
- **Mitigation:** This was addressed using strict **10-Fold Cross-Validation** and regularization techniques (L1/L2) in the XGBoost and LSTM modules.
- **Synthetic Data:** The model relies on 85% physics-informed synthetic data validated against NREL and IRENA benchmarks. While rigorous, real-world deployment may encounter 'long-tail' fault distributions not captured in the training set.

## 5. Data Provenance

- **Synthetic Data:** Generated via Butler-Volmer kinetics (85%).
- **Validation Anchors:** NREL Submission 305 (Wind-to-H2) and IRENA Green H2 Cost Reduction 2025 projections.
- **Availability:** All generating scripts and seed parameters are available in the supplementary material to ensure 100% reproducibility.

## 6. Policy Impact

While this study discusses policy implications (NGHM), it does not involve human subjects and thus did not require Institutional Review Board (IRB) approval. The views expressed regarding NGHM effectiveness are independent assessments based on techno-economic modeling and do not reflect official government positions.
