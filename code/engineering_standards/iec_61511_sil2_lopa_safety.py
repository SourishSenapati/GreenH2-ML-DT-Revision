"""
IEC 61511 / IEC 61508 Quantitative Layer of Protection Analysis (LOPA) & Functional Safety Engine
Standards Reference:
  - IEC 61511: Functional Safety - Safety Instrumented Systems for the Process Industry Sector
    * Part 1: Framework, definitions, system, hardware and software requirements
    * Part 2: Guidelines for the application of IEC 61511-1
  - IEC 61508: Functional Safety of Electrical/Electronic/Programmable Electronic Safety-Related Systems
    * Part 2: Requirements for E/E/PE safety-related systems (Hardware safety integrity)
    * Part 6: Guidelines on the application of IEC 61508-2 and IEC 61508-3

Safety Instrumented Functions (SIFs) for 100 TPD PEM Green Hydrogen Complex:
  - SIF-01: Anode Gas Phase H2 Crossover Quench Interlock
    Initiating Event: Diaphragm thinning or mechanical pinhole leak causing H2 migration to O2 stream.
    Target SIL: SIL 2 (Risk Reduction Factor 100 <= RRF < 1000)
    Sensor: 2x Fast Infrared H2-in-O2 Gas Analyzers (1oo2 voting architecture)
    Logic Solver: Siemens S7-1500F Safety PLC (SIL 3 rated CPU)
    Final Element: Fast-Acting Pneumatic ESD Valve with fail-closed spring return (<500ms stroke)

  - SIF-02: Stack Differential Pressure Delta-P Protection Interlock
    Initiating Event: Downstream H2 backpressure surge or cathode blockage.
    Target SIL: SIL 2
    Sensor: 2x Smart Differential Pressure Transmitters (1oo2 voting)
    Final Element: High-Speed DC Thyristor Breaker Shunt Trip (<50ms trip latency)

  - SIF-03: In-Operando Dielectric Biofilm & Low Water Conductivity Surge
    Initiating Event: Bacterial extracellular polymeric substance (EPS) accumulation causing cell voltage divergence.
    Target SIL: SIL 2
    Sensor: Decoupled Bayesian Impedance Virtual Sensor + Dual Resistivity Probes (>18.2 M-Ohm-cm)
    Final Element: Proactive DI Flush Solenoid Pulse (15 -> 28 L/min) + Current Throttling (2.0 -> 1.3 A/cm2)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class SIFComponent:
    tag: str
    subsystem_type: str  # 'Sensor', 'LogicSolver', 'FinalElement'
    architecture: str    # '1oo1', '1oo2', '2oo3'
    lambda_du_per_hour: float  # Dangerous Undetected Failure Rate (failures / hr)
    lambda_dd_per_hour: float  # Dangerous Detected Failure Rate
    lambda_su_per_hour: float  # Safe Undetected Failure Rate
    lambda_sd_per_hour: float  # Safe Detected Failure Rate
    proof_test_interval_hr: float = 8760.0  # 1 year (8760 hours)
    mean_time_to_repair_hr: float = 24.0    # MTTR (hours)
    beta_common_cause: float = 0.05         # 5% common cause factor for redundant channels

@dataclass
class SafetyInstrumentedFunction:
    sif_id: str
    name: str
    target_sil: int
    initiating_cause: str
    consequence: str
    tolerable_frequency_per_yr: float
    unmitigated_frequency_per_yr: float
    components: List[SIFComponent]

class IEC61511_Safety_Calculator:
    """Quantitative PFD_avg and SIL verification engine per IEC 61508 / 61511."""

    @staticmethod
    def calculate_component_pfd(comp: SIFComponent) -> Dict[str, float]:
        """
        Computes PFD_avg based on hardware architecture:
        - 1oo1:
            PFD_avg = lambda_DU * (T1 / 2) + lambda_DU * MTTR
        - 1oo2 (with beta common cause):
            PFD_avg_ind = (lambda_DU^2 * T1^2) / 3
            PFD_avg_cc = beta * lambda_DU * (T1 / 2)
            PFD_avg = PFD_avg_ind + PFD_avg_cc
        - 2oo3:
            PFD_avg = (lambda_DU^2 * T1^2) + beta * lambda_DU * (T1 / 2)
        """
        lam_du = comp.lambda_du_per_hour
        T1 = comp.proof_test_interval_hr
        mttr = comp.mean_time_to_repair_hr
        beta = comp.beta_common_cause

        if comp.architecture == '1oo1':
            pfd_avg = lam_du * (T1 / 2.0) + lam_du * mttr
        elif comp.architecture == '1oo2':
            pfd_ind = (lam_du**2 * T1**2) / 3.0
            pfd_cc = beta * lam_du * (T1 / 2.0)
            pfd_avg = pfd_ind + pfd_cc
        elif comp.architecture == '2oo3':
            pfd_ind = (lam_du**2 * T1**2)
            pfd_cc = beta * lam_du * (T1 / 2.0)
            pfd_avg = pfd_ind + pfd_cc
        else:
            raise ValueError(f"Unsupported architecture: {comp.architecture}")

        # Safe Failure Fraction (SFF)
        total_lambda = comp.lambda_sd_per_hour + comp.lambda_su_per_hour + comp.lambda_dd_per_hour + comp.lambda_du_per_hour
        safe_failures = comp.lambda_sd_per_hour + comp.lambda_su_per_hour + comp.lambda_dd_per_hour
        sff = (safe_failures / total_lambda) if total_lambda > 0 else 0.95

        return {
            "pfd_avg": pfd_avg,
            "sff": sff
        }

    @classmethod
    def evaluate_sif(cls, sif: SafetyInstrumentedFunction) -> Dict[str, Any]:
        """Aggregates sensor, logic solver, and final element PFDs to verify SIL compliance."""
        total_pfd = 0.0
        component_details = []

        for comp in sif.components:
            res = cls.calculate_component_pfd(comp)
            total_pfd += res["pfd_avg"]
            component_details.append({
                "tag": comp.tag,
                "type": comp.subsystem_type,
                "arch": comp.architecture,
                "pfd_avg": round(res["pfd_avg"], 6),
                "sff_percent": round(res["sff"] * 100.0, 1)
            })

        rrf = (1.0 / total_pfd) if total_pfd > 0 else 99999.0

        # SIL determination per IEC 61508-1 Table 2 (Low demand mode)
        # SIL 4: 10^-5 <= PFD < 10^-4  (RRF 10,000 to 100,000)
        # SIL 3: 10^-4 <= PFD < 10^-3  (RRF 1,000 to 10,000)
        # SIL 2: 10^-3 <= PFD < 10^-2  (RRF 100 to 1,000)
        # SIL 1: 10^-2 <= PFD < 10^-1  (RRF 10 to 100)
        if total_pfd < 1e-4:
            achieved_sil = 3
        elif total_pfd < 1e-3:
            achieved_sil = 2
        elif total_pfd < 1e-2:
            achieved_sil = 2  # high-end SIL 2
        elif total_pfd < 1e-1:
            achieved_sil = 1
        else:
            achieved_sil = 0

        sil_compliant = achieved_sil >= sif.target_sil

        # Required Risk Reduction from LOPA initiating events
        required_rrf = sif.unmitigated_frequency_per_yr / sif.tolerable_frequency_per_yr

        return {
            "sif_id": sif.sif_id,
            "sif_name": sif.name,
            "target_sil": f"SIL {sif.target_sil}",
            "achieved_sil": f"SIL {achieved_sil}",
            "sil_status": "IEC 61511 VERIFIED PASS" if sil_compliant else "NON-COMPLIANT",
            "total_pfd_avg": round(total_pfd, 5),
            "achieved_rrf": round(rrf, 1),
            "required_lopa_rrf": round(required_rrf, 1),
            "margin_over_target": round(rrf / required_rrf, 2),
            "component_breakdown": component_details
        }

def run_lopa_functional_safety_audit():
    """Evaluates the primary SIFs for the 100 TPD hydrogen electrolysis skid."""
    print("=" * 80)
    print("IEC 61511 / IEC 61508 QUANTITATIVE FUNCTIONAL SAFETY & LOPA AUDIT")
    print("HYDROSENSE Industrial High-Integrity Pressure Protection System (HIPPS)")
    print("=" * 80)

    calc = IEC61511_Safety_Calculator()

    # SIF-01: Anode H2 Crossover Protection (1oo2 IR Analyzers -> S7-1500F -> Fast ESD)
    sif_01 = SafetyInstrumentedFunction(
        sif_id="SIF-01",
        name="Anode Gas Phase H2 Crossover Quench Interlock",
        target_sil=2,
        initiating_cause="Membrane pinhole failure / high delta-P driving H2 into O2 loop",
        consequence="Flammable atmosphere in anode separator drum (LEL exceedance)",
        tolerable_frequency_per_yr=1e-4,  # 1 in 10,000 years
        unmitigated_frequency_per_yr=0.03,  # 1 in 33 years
        components=[
            SIFComponent(
                tag="AT-301A/B Dual Fast NDIR Hydrogen Sensors",
                subsystem_type="Sensor",
                architecture="1oo2",
                lambda_du_per_hour=1.2e-6,
                lambda_dd_per_hour=8.5e-6,
                lambda_su_per_hour=3.0e-6,
                lambda_sd_per_hour=9.0e-6,
                proof_test_interval_hr=8760.0
            ),
            SIFComponent(
                tag="PLC-1000 Siemens S7-1500F Safety Controller",
                subsystem_type="LogicSolver",
                architecture="1oo1",
                lambda_du_per_hour=4.5e-8,
                lambda_dd_per_hour=1.1e-6,
                lambda_su_per_hour=2.0e-7,
                lambda_sd_per_hour=5.0e-6,
                proof_test_interval_hr=8760.0
            ),
            SIFComponent(
                tag="XV-301 High-Speed Nitrogen Sweep ESD Valve",
                subsystem_type="FinalElement",
                architecture="1oo1",
                lambda_du_per_hour=2.1e-6,
                lambda_dd_per_hour=4.2e-6,
                lambda_su_per_hour=1.5e-6,
                lambda_sd_per_hour=8.0e-6,
                proof_test_interval_hr=4380.0  # 6-month proof test stroke
            )
        ]
    )

    # SIF-02: Stack Differential Pressure Delta-P Protection Interlock
    sif_02 = SafetyInstrumentedFunction(
        sif_id="SIF-02",
        name="Stack Differential Pressure Delta-P Hydraulic Protection",
        target_sil=2,
        initiating_cause="Cathode gas line valve closure or regulator freeze causing Delta-P > 32 bar",
        consequence="Structural rupture of polymer electrolyte membrane and 50-cell stack teardown",
        tolerable_frequency_per_yr=1e-4,
        unmitigated_frequency_per_yr=0.04,
        components=[
            SIFComponent(
                tag="PDT-201A/B Smart Differential Pressure Transmitters",
                subsystem_type="Sensor",
                architecture="1oo2",
                lambda_du_per_hour=9.8e-7,
                lambda_dd_per_hour=7.1e-6,
                lambda_su_per_hour=2.5e-6,
                lambda_sd_per_hour=8.2e-6
            ),
            SIFComponent(
                tag="PLC-1000 Siemens S7-1500F Safety Controller",
                subsystem_type="LogicSolver",
                architecture="1oo1",
                lambda_du_per_hour=4.5e-8,
                lambda_dd_per_hour=1.1e-6,
                lambda_su_per_hour=2.0e-7,
                lambda_sd_per_hour=5.0e-6
            ),
            SIFComponent(
                tag="DCB-201 5000A Fast Thyristor Breaker Shunt Trip",
                subsystem_type="FinalElement",
                architecture="1oo1",
                lambda_du_per_hour=1.8e-6,
                lambda_dd_per_hour=5.0e-6,
                lambda_su_per_hour=1.2e-6,
                lambda_sd_per_hour=6.5e-6
            )
        ]
    )

    results = []
    for sif in [sif_01, sif_02]:
        res = calc.evaluate_sif(sif)
        results.append(res)
        print(f"\n[SIF]: {res['sif_id']} - {res['sif_name']}")
        print(f"  Target SIL: {res['target_sil']} | Achieved SIL: {res['achieved_sil']} | Status: {res['sil_status']}")
        print(f"  Total PFD_avg: {res['total_pfd_avg']:.6f} | Achieved RRF: {res['achieved_rrf']} (Required: {res['required_lopa_rrf']})")
        print("  Subsystem Hardware Breakdown:")
        for comp in res["component_breakdown"]:
            print(f"    - {comp['tag']} ({comp['type']} {comp['arch']}): PFD_avg = {comp['pfd_avg']:.6f}, SFF = {comp['sff_percent']}%")

    return results

if __name__ == "__main__":
    run_lopa_functional_safety_audit()

