"""
30-Bar Differential Pressure Stack Mechanics & Gas Crossover Kinetics Engine
Standards Reference:
  - ISO 22734:2019: Hydrogen generators using water electrolysis — Industrial, commercial, and residential applications
    * Clause 4.2.3: Cross-contamination and flammable gas mixtures (<50% LEL, threshold <2.0 vol% H2 in O2, target <1.0 vol%)
    * Clause 4.3.2: Differential pressure integrity under nominal 30 bar stack operations
  - NFPA 2: Hydrogen Technologies Code (Safety thresholds for pressurized electrolysis cells)

Physical Boundary Conditions:
  - Cathode Operating Pressure (P_cath): 30.0 bar (3.0 MPa) direct delivery
  - Anode Operating Pressure (P_an): 1.50 bar (0.15 MPa) water circulation
  - Differential Pressure (Delta_P): 28.5 bar (2.85 MPa) hydraulic mechanical load
  - Operating Temperature: 80.0 deg C (353.15 K)
  - Active Area per Cell: 250 cm^2 (0.025 m^2)
  - Stack Architecture: 50-cell industrial stack (Unit EL-200)
  - Current Density (j): 0.20 to 2.50 A/cm^2 (Nominal design: 2.00 A/cm^2)
  - Membrane: Chemically Stabilized Reinforced PFSA (e.g. Nafion HP / XL reinforced with PTFE mesh, 50 um nominal)
"""

import math
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StackMechanicalInputs:
    delta_p_bar: float = 28.5  # Differential pressure across membrane (bar)
    operating_temp_c: float = 80.0  # Temperature (deg C)
    membrane_thickness_um: float = 85.0  # Reinforced ePTFE/PFSA composite (e.g. Gore-Select / Nafion HP)
    ptl_mean_pore_diameter_um: float = 25.0  # Sintered Titanium PTL (Grade 1/2 micro-porous layer)
    channel_rib_width_mm: float = 1.0  # Flow field land/rib width (mm)
    channel_width_mm: float = 1.2  # Flow field flow channel width (mm)
    membrane_youngs_modulus_mpa: float = 280.0  # Reinforced PFSA at 80C
    membrane_poissons_ratio: float = 0.38
    membrane_yield_strength_mpa: float = 24.5  # Biaxially oriented ePTFE reinforced substrate
    clamping_bolt_torque_nm: float = 7.5  # Calibrated for 3.2 MPa uniform contact stress
    tie_rod_count: int = 16

@dataclass
class CrossoverKineticsInputs:
    current_density_a_cm2: float = 2.00  # Operating current density (A/cm2)
    delta_p_bar: float = 28.5  # Differential pressure (bar)
    operating_temp_c: float = 80.0
    membrane_thickness_um: float = 85.0  # 85 um reinforced membrane
    faraday_constant: float = 96485.33  # C / mol electrons
    h2_permeability_coeff_mol_m_s_pa: float = 1.95e-15  # ePTFE-reinforced composite @ 80C
    darcy_hydraulic_permeability_m2: float = 3.50e-20    # Hydraulic Darcy crossover through dense reinforced matrix
    water_dynamic_viscosity_pa_s: float = 3.55e-4       # Liquid water @ 80C
    henry_solubility_mol_m3_pa: float = 7.15e-6          # H2 solubility in water @ 80C

class Stack30BarDifferentialEngine:
    """Calculates mechanical deflection, stress, and electrochemical gas crossover under 30 bar Delta-P."""

    @staticmethod
    def calculate_membrane_structural_integrity(inputs: StackMechanicalInputs) -> Dict[str, Any]:
        """
        Mechanics of thin circular membrane clamped over PTL pore openings under 28.5 bar Delta-P:
        Pore radius r_p = d_pore / 2 (15 um)
        Maximum deflection (Timoshenko thin circular plate under uniform load q):
            w_max = (q * r_p^4) / (64 * D_rigidity)
            D_rigidity = (E * t^3) / (12 * (1 - nu^2))
        Maximum Bending Stress at clamped boundary:
            sigma_max = (3/4) * q * (r_p / t)^2
        """
        q = inputs.delta_p_bar * 1e5  # bar -> Pa (N/m^2)
        r_p = (inputs.ptl_mean_pore_diameter_um * 1e-6) / 2.0  # m
        t_m = inputs.membrane_thickness_um * 1e-6  # m
        E = inputs.membrane_youngs_modulus_mpa * 1e6  # Pa
        nu = inputs.membrane_poissons_ratio

        # Flexural rigidity
        D = (E * (t_m**3)) / (12.0 * (1.0 - nu**2))

        # Maximum deflection over PTL pore
        w_max = (q * (r_p**4)) / (64.0 * D)  # m
        w_max_um = w_max * 1e6  # microns

        # Maximum edge stress
        sigma_max = 0.75 * q * ((r_p / t_m)**2)  # Pa
        sigma_max_mpa = sigma_max * 1e-6  # MPa

        # Safety factor against yield (18.5 MPa wet PFSA)
        safety_factor = inputs.membrane_yield_strength_mpa / sigma_max_mpa

        # Clamping compression pressure on PTL lands
        # 16x M16 bolts @ 35 Nm -> Clamping Force F_clamp = (16 * T) / (K * d)
        # K = 0.20 (nut factor), d = 0.016 m
        f_clamp_per_bolt = inputs.clamping_bolt_torque_nm / (0.20 * 0.016)  # N
        total_clamp_force_kn = (inputs.tie_rod_count * f_clamp_per_bolt) * 1e-3  # kN
        active_area_m2 = 0.025  # 250 cm^2
        land_area_ratio = inputs.channel_rib_width_mm / (inputs.channel_rib_width_mm + inputs.channel_width_mm)  # 1.0 / 2.2 = 0.455
        ptl_contact_stress_mpa = (total_clamp_force_kn * 1e3) / (active_area_m2 * land_area_ratio * 1e6)  # MPa

        status = "STRUCTURAL PASS" if safety_factor >= 2.0 and 2.0 <= ptl_contact_stress_mpa <= 4.0 else "REVIEW"

        return {
            "delta_p_applied_bar": inputs.delta_p_bar,
            "membrane_thickness_um": inputs.membrane_thickness_um,
            "ptl_pore_diameter_um": inputs.ptl_mean_pore_diameter_um,
            "pore_bridging_deflection_um": round(w_max_um, 4),
            "max_bending_stress_mpa": round(sigma_max_mpa, 3),
            "membrane_yield_strength_mpa": inputs.membrane_yield_strength_mpa,
            "mechanical_safety_factor": round(safety_factor, 2),
            "total_stack_clamping_force_kn": round(total_clamp_force_kn, 1),
            "ptl_land_contact_stress_mpa": round(ptl_contact_stress_mpa, 2),
            "structural_status": status
        }

    @staticmethod
    def calculate_gas_crossover_safety(inputs: CrossoverKineticsInputs) -> Dict[str, Any]:
        """
        Calculates dissolved H2 crossover into anode oxygen stream under 30 bar Delta-P:
        1. Fickian Diffusion Flux (mol / m^2 s):
            J_diff = (P_perm / t_m) * Delta_P_H2
        2. Hydraulic Convective Flux (mol / m^2 s):
            J_conv = (k_Darcy / mu_water) * c_H2_dissolved * (Delta_P / t_m)
            where c_H2_dissolved = K_Henry * P_cath
        3. Total Crossover Flux:
            J_total = J_diff + J_conv
        4. Faradaic Oxygen Generation Rate:
            N_O2 = j / (4 * F)  (mol O2 / m^2 s)
        5. Flammability Concentration in Anode Stream:
            vol% H2 in O2 = J_total / (2 * N_O2 + J_total) * 100
        """
        t_m = inputs.membrane_thickness_um * 1e-6  # m
        delta_p_pa = inputs.delta_p_bar * 1e5      # Pa
        p_cath_pa = 30.0 * 1e5                     # Pa

        # 1. Diffusion
        j_diff = (inputs.h2_permeability_coeff_mol_m_s_pa / t_m) * delta_p_pa

        # 2. Convection
        c_h2_sat = inputs.henry_solubility_mol_m3_pa * p_cath_pa  # mol / m^3
        dp_dx = delta_p_pa / t_m  # Pa / m
        v_darcy = (inputs.darcy_hydraulic_permeability_m2 / inputs.water_dynamic_viscosity_pa_s) * dp_dx  # m / s
        j_conv = c_h2_sat * v_darcy  # mol / m^2 s

        j_total = j_diff + j_conv  # mol H2 / m^2 s

        # Faradaic O2 rate at operating current
        j_a_m2 = inputs.current_density_a_cm2 * 1e4  # A/m^2
        n_o2 = j_a_m2 / (4.0 * inputs.faraday_constant)  # mol O2 / m^2 s

        # Mole fraction / vol% H2 in O2
        # (Dry basis equivalent: 1 mol O2 produces 2 mol gas eq or direct molar ratio)
        h2_vol_pct = (j_total / (n_o2 + j_total)) * 100.0

        # ISO 22734 threshold: < 1.0 vol% (well below 4.0 vol% LEL)
        is_safe = h2_vol_pct < 1.0
        lel_margin = (4.0 / h2_vol_pct) if h2_vol_pct > 0 else 999.0

        return {
            "current_density_a_cm2": inputs.current_density_a_cm2,
            "fickian_diffusion_flux_mol_m2_s": round(j_diff, 7),
            "hydraulic_convective_flux_mol_m2_s": round(j_conv, 7),
            "total_crossover_flux_mol_m2_s": round(j_total, 7),
            "o2_generation_rate_mol_m2_s": round(n_o2, 4),
            "h2_in_o2_volume_percent": round(h2_vol_pct, 4),
            "iso_22734_limit_percent": 1.0,
            "lower_explosive_limit_lel_pct": 4.0,
            "safety_margin_to_lel": round(lel_margin, 1),
            "crossover_compliance_status": "ISO 22734 COMPLIANT (SAFE)" if is_safe else "EXCEEDS THRESHOLD"
        }

def run_30bar_stack_mechanics_audit():
    """Executes 30 bar differential stack validation."""
    print("=" * 80)
    print("30-BAR DIFFERENTIAL PRESSURE STACK STRUCTURAL & CROSSOVER AUDIT")
    print("HYDROSENSE Industrial EL-200 50-Cell Stack (250 cm^2 Active Area)")
    print("=" * 80)

    engine = Stack30BarDifferentialEngine()
    mech_res = engine.calculate_membrane_structural_integrity(StackMechanicalInputs())
    crossover_res = engine.calculate_gas_crossover_safety(CrossoverKineticsInputs(current_density_a_cm2=2.00))

    print("\n[PART A]: Membrane Pore-Bridging Structural Mechanics (28.5 bar Delta-P):")
    for k, v in mech_res.items():
        print(f"  {k:35s}: {v}")

    print("\n[PART B]: Electrochemical Gas Crossover Safety (Cathode 30 bar -> Anode 1.5 bar):")
    for k, v in crossover_res.items():
        print(f"  {k:35s}: {v}")

    return {"mechanical": mech_res, "crossover": crossover_res}

if __name__ == "__main__":
    run_30bar_stack_mechanics_audit()
