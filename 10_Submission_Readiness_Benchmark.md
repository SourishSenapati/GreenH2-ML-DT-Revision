# Submission Readiness Benchmark

**Rule:** `Post-metrics hit = Ready for submission`

This document defines the **non-negotiable** success criteria. The manuscript will NOT be submitted until every single "Post" metric below is achieved and verified.

## The Success Equation

$$
\text{Ready} = (\text{Novelty} \ge 9.5) \land (\text{Real Data} \ge 85\%) \land (\text{Model } R^2 \ge 0.96) \land (\text{Compliance} == 100\%)
$$

## Strict Metric Thresholds

| Metric | Current State (Pre) | **Target (Post) - THE BENCHMARK** | Verification Method |
| :--- | :--- | :--- | :--- |
| **Novelty Score** | 4/10 | **9.5/10** | Peer review confirms "First unified PEM DT with quantum sims + exp validation". |
| **Real Data %** | 0% | **85%** | `df['source'].value_counts()` shows >80% rows from NREL/Experiments. |
| **Model Accuracy** | 0.97 (Synth) | **R² $\ge$ 0.96 (Real)** | Tested on NREL validation set (not training set). |
| **Oversight** | Manual | **Automated** | `kpis.csv` updated weekly via script. |
| **Compliance** | 70% | **100%** | Passes all checks in `08_Originality_Strategy.md`. |

## Action Protocol

1.  **Weekly Audit:** Check `logs/kpis.csv`.
2.  **Gap Analysis:** If any metric < Target, triggering **Emergency Sprint** (see Elevation Plan).
3.  **Green Light:** Only when **ALL** targets in column 3 are met, proceed to Phase 5.3 (Submit to Nature Energy).
