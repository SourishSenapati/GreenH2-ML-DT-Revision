"""
ASME BPVC Section VIII Division 1 & Division 3 Quantitative Sizing Engine
Standards Reference:
  - ASME BPVC Section VIII, Division 1 (Rules for Construction of Pressure Vessels)
    * Mandatory Paragraph UG-27: Thickness of Shells under Internal Pressure
    * Mandatory Paragraph UG-32: Formed Heads and Sections, Pressure on Concave Side (2:1 Semi-Ellipsoidal)
    * Mandatory Paragraph UG-99: Standard Hydrostatic Test
  - ASME BPVC Section VIII, Division 3 (Alternative Rules for High Pressure Vessels)
    * Part KD: Design Requirements for High Pressure Vessels (Thick-Walled Cylinders & Lame Equations)

Design Parameters:
  - Working Fluid: High-Purity Hydrogen Gas (H2) & Deionized Water
  - Design Pressure (P_des): 30.0 bar (3.0 MPa) for Knockout Drum (V-300)
  - Buffer Storage Design Pressure (P_buf): 350.0 bar (35.0 MPa) for Storage Vessels (ST-800)
  - Design Temperature: 80.0 deg C (353.15 K)
  - Material: SA-240 Type 316L Stainless Steel (V-300), SA-723 Class 2 Forged Steel (ST-800)
  - Maximum Allowable Stress (S):
      * SA-240 316L @ 80 deg C: 115.0 MPa (16.68 ksi) per ASME Sec. II Part D Table 1A
      * SA-723 Cl 2 @ 50 deg C: 320.0 MPa (46.4 ksi) per ASME Sec. II Part D Table 2A
  - Joint Efficiency (E): 1.0 (Full Radiographic Examination per UW-11(a) Type 1)
  - Corrosion Allowance (CA): 1.50 mm (0.0015 m) for wet hydrogen service
"""

import math
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VesselDiv1Inputs:
    tag: str
    description: str
    inner_diameter_mm: float  # D_i (mm)
    design_pressure_bar: float  # P (bar)
    design_temp_c: float  # T (C)
    material: str
    allowable_stress_mpa: float  # S (MPa)
    joint_efficiency: float  # E (1.0 for full RT, 0.85 for spot)
    corrosion_allowance_mm: float  # CA (mm)

@dataclass
class HeadDiv1Inputs:
    tag: str
    inner_diameter_mm: float  # D (mm)
    design_pressure_bar: float  # P (bar)
    allowable_stress_mpa: float  # S (MPa)
    joint_efficiency: float  # E
    corrosion_allowance_mm: float  # CA (mm)

@dataclass
class VesselDiv3Inputs:
    tag: str
    description: str
    inner_radius_mm: float  # r_i (mm)
    outer_radius_mm: float  # r_o (mm)
    internal_pressure_bar: float  # P_i (bar)
    yield_strength_mpa: float  # S_y (MPa)
    design_margin: float = 1.732  # Von Mises factor per ASME Div 3 KD-250

class ASME_BPVC_Calculator:
    """Calculates code-compliant wall thicknesses and hydrostatic test limits."""

    @staticmethod
    def calculate_ug27_cylindrical_shell(inputs: VesselDiv1Inputs) -> Dict[str, Any]:
        """
        ASME Section VIII Div. 1, UG-27(c)(1):
        Circumferential Stress (Longitudinal Joints):
            t = (P * R) / (S * E - 0.6 * P) + CA
        Applicability Condition:
            t <= 0.5 * R or P <= 0.385 * S * E
        """
        P = inputs.design_pressure_bar * 0.1  # bar -> MPa
        R = (inputs.inner_diameter_mm / 2.0)  # mm
        S = inputs.allowable_stress_mpa
        E = inputs.joint_efficiency
        CA = inputs.corrosion_allowance_mm

        # Check applicability
        limit_p = 0.385 * S * E
        if P > limit_p:
            raise ValueError(f"Pressure P={P:.2f} MPa exceeds UG-27 thin-shell limit ({limit_p:.2f} MPa). Use Div 3 or thick cylinder.")

        denominator = S * E - 0.6 * P
        if denominator <= 0:
            raise ValueError("Allowable stress denominator non-positive.")

        t_calc = (P * R) / denominator
        t_required = t_calc + CA

        # Commercial plate rounding (standard ASTM A240 stock plates: 8, 10, 12, 14, 16, 20 mm)
        plate_sizes = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0]
        selected_plate = next((p for p in plate_sizes if p >= t_required), math.ceil(t_required))

        # Maximum Allowable Working Pressure (MAWP) new and uncorroded & corroded
        # UG-27 inverted: MAWP = (S * E * t_nom_minus_ca) / (R + 0.6 * t_nom_minus_ca)
        t_net = selected_plate - CA
        mawp_mpa = (S * E * t_net) / (R + 0.6 * t_net)
        mawp_bar = mawp_mpa * 10.0

        # UG-99 Hydrostatic Test Pressure
        # P_test = 1.3 * MAWP * (S_test / S_design)
        # For SA-240 316L, S_test (at 20C) = 115 MPa (conservative)
        p_hydro_bar = 1.3 * mawp_bar

        return {
            "component": inputs.tag,
            "code_clause": "ASME BPVC Sec VIII Div 1 UG-27(c)(1)",
            "inner_diameter_mm": inputs.inner_diameter_mm,
            "design_pressure_bar": inputs.design_pressure_bar,
            "design_temp_c": inputs.design_temp_c,
            "material": inputs.material,
            "allowable_stress_S_mpa": S,
            "joint_efficiency_E": E,
            "corrosion_allowance_CA_mm": CA,
            "t_calculated_net_mm": round(t_calc, 3),
            "t_required_with_ca_mm": round(t_required, 3),
            "commercial_plate_selected_mm": selected_plate,
            "mawp_bar": round(mawp_bar, 2),
            "ug99_hydrotest_pressure_bar": round(p_hydro_bar, 2),
            "compliance_status": "ASME SEC VIII DIV 1 COMPLIANT"
        }

    @staticmethod
    def calculate_ug32_ellipsoidal_head(inputs: HeadDiv1Inputs) -> Dict[str, Any]:
        """
        ASME Section VIII Div. 1, UG-32(d):
        2:1 Semi-Ellipsoidal Formed Heads (D / 2h = 2):
            t = (P * D) / (2 * S * E - 0.2 * P) + CA
        """
        P = inputs.design_pressure_bar * 0.1  # MPa
        D = inputs.inner_diameter_mm  # mm
        S = inputs.allowable_stress_mpa
        E = inputs.joint_efficiency
        CA = inputs.corrosion_allowance_mm

        denominator = 2.0 * S * E - 0.2 * P
        t_calc = (P * D) / denominator
        t_required = t_calc + CA

        plate_sizes = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]
        selected_plate = next((p for p in plate_sizes if p >= t_required), math.ceil(t_required))

        t_net = selected_plate - CA
        mawp_mpa = (2.0 * S * E * t_net) / (D + 0.2 * t_net)
        mawp_bar = mawp_mpa * 10.0
        p_hydro_bar = 1.3 * mawp_bar

        return {
            "component": inputs.tag,
            "code_clause": "ASME BPVC Sec VIII Div 1 UG-32(d)",
            "head_type": "2:1 Semi-Ellipsoidal Formed Head",
            "inner_diameter_mm": D,
            "t_calculated_net_mm": round(t_calc, 3),
            "t_required_with_ca_mm": round(t_required, 3),
            "commercial_head_plate_mm": selected_plate,
            "mawp_bar": round(mawp_bar, 2),
            "ug99_hydrotest_pressure_bar": round(p_hydro_bar, 2),
            "compliance_status": "ASME SEC VIII DIV 1 COMPLIANT"
        }

    @staticmethod
    def calculate_div3_high_pressure_cylinder(inputs: VesselDiv3Inputs) -> Dict[str, Any]:
        """
        ASME Section VIII Div. 3, Part KD (Alternative Rules for High Pressure):
        Thick-Walled Cylinder Lame Equations:
            Y = r_o / r_i
            Hoop Stress at Inner Bore:
                sigma_theta_i = P_i * (Y^2 + 1) / (Y^2 - 1)
            Radial Stress at Inner Bore:
                sigma_r_i = -P_i
            Von Mises Equivalent Stress at Bore:
                sigma_vm = sqrt(3)/2 * (sigma_theta_i - sigma_r_i) = P_i * sqrt(3) * Y^2 / (Y^2 - 1)
            Yield Burst Criterion (KD-250):
                P_yield = (S_y / sqrt(3)) * (Y^2 - 1) / Y^2
                P_burst = (2 / sqrt(3)) * S_y * ln(Y)
        """
        r_i = inputs.inner_radius_mm
        r_o = inputs.outer_radius_mm
        Y = r_o / r_i
        P_i = inputs.internal_pressure_bar * 0.1  # MPa
        S_y = inputs.yield_strength_mpa

        if Y <= 1.0:
            raise ValueError("Outer radius must be greater than inner radius.")

        # Lame stresses at inner wall
        sigma_theta = P_i * (Y**2 + 1.0) / (Y**2 - 1.0)
        sigma_r = -P_i
        sigma_axial = P_i / (Y**2 - 1.0)

        # Von Mises stress at inside radius
        sigma_vm = math.sqrt(0.5 * ((sigma_theta - sigma_r)**2 + (sigma_r - sigma_axial)**2 + (sigma_axial - sigma_theta)**2))

        # Div 3 allowable design pressure limit (KD-250)
        p_allowable_mpa = (S_y / inputs.design_margin) * (Y**2 - 1.0) / (Y**2 * math.sqrt(3.0))
        p_allowable_bar = p_allowable_mpa * 10.0

        # Burst pressure
        p_burst_mpa = (2.0 / math.sqrt(3.0)) * S_y * math.log(Y)
        p_burst_bar = p_burst_mpa * 10.0

        safety_margin_yield = S_y / sigma_vm
        compliance = "ASME SEC VIII DIV 3 COMPLIANT" if safety_margin_yield >= inputs.design_margin else "NON-COMPLIANT"

        return {
            "component": inputs.tag,
            "code_clause": "ASME BPVC Sec VIII Div 3 Part KD (KD-250)",
            "inner_radius_mm": r_i,
            "outer_radius_mm": r_o,
            "wall_thickness_mm": round(r_o - r_i, 2),
            "diameter_ratio_Y": round(Y, 3),
            "design_internal_pressure_bar": inputs.internal_pressure_bar,
            "hoop_stress_bore_mpa": round(sigma_theta, 2),
            "von_mises_stress_bore_mpa": round(sigma_vm, 2),
            "material_yield_strength_mpa": S_y,
            "safety_margin_on_yield": round(safety_margin_yield, 3),
            "allowable_working_pressure_bar": round(p_allowable_bar, 2),
            "theoretical_burst_pressure_bar": round(p_burst_bar, 2),
            "compliance_status": compliance
        }

def run_production_plant_asme_audit():
    """Executes code verification for the complete commercial plant."""
    print("=" * 80)
    print("ASME BPVC SECTION VIII DIV. 1 & DIV. 3 QUANTITATIVE ENGINEERING AUDIT")
    print("HYDROSENSE Turnkey Green Hydrogen Production Complex (100 TPD Skid)")
    print("=" * 80)

    calc = ASME_BPVC_Calculator()

    # 1. Knockout Drum V-300 Shell (30 bar H2 delivery)
    v300_shell = VesselDiv1Inputs(
        tag="V-300 HP Hydrogen Knockout Drum (Cylindrical Shell)",
        description="High Pressure Gas-Liquid Separator after 30-bar PEM stack",
        inner_diameter_mm=800.0,
        design_pressure_bar=30.0,
        design_temp_c=80.0,
        material="SA-240 Type 316L",
        allowable_stress_mpa=115.0,
        joint_efficiency=1.0,
        corrosion_allowance_mm=1.50
    )
    res_v300_shell = calc.calculate_ug27_cylindrical_shell(v300_shell)

    # 2. Knockout Drum V-300 2:1 Ellipsoidal Head
    v300_head = HeadDiv1Inputs(
        tag="V-300 2:1 Semi-Ellipsoidal Heads (Top & Bottom)",
        inner_diameter_mm=800.0,
        design_pressure_bar=30.0,
        allowable_stress_mpa=115.0,
        joint_efficiency=1.0,
        corrosion_allowance_mm=1.50
    )
    res_v300_head = calc.calculate_ug32_ellipsoidal_head(v300_head)

    # 3. Buffer Storage Array ST-800 High Pressure Vessel (350 bar)
    st800_vessel = VesselDiv3Inputs(
        tag="ST-800 High-Pressure Hydrogen Buffer Vessel (Cylinder)",
        description="350 bar storage accumulator for compressor pulse damping and buffer",
        inner_radius_mm=250.0,   # ID = 500 mm
        outer_radius_mm=325.0,   # OD = 650 mm (t = 75 mm forged wall)
        internal_pressure_bar=350.0,
        yield_strength_mpa=725.0,  # SA-723 Grade 3 Class 2
        design_margin=1.732
    )
    res_st800 = calc.calculate_div3_high_pressure_cylinder(st800_vessel)

    for item in [res_v300_shell, res_v300_head, res_st800]:
        print(f"\n[TAG]: {item['component']}")
        print(f"  Clause: {item['code_clause']}")
        for k, v in item.items():
            if k not in ['component', 'code_clause']:
                print(f"  {k:35s}: {v}")

    return {
        "v300_shell": res_v300_shell,
        "v300_head": res_v300_head,
        "st800_storage": res_st800
    }

if __name__ == "__main__":
    run_production_plant_asme_audit()

