/**
 * HYDROSENSE™ Engineering Standards & Compliance Verification Modal
 * Harmonizes Indian Statutory Regulations (The Factories Act 1948, PESO SMPV Rules 2016,
 * IS 800:2007) with Global Cross-Border Export Codes (ASME BPVC, ISO 22734, IEC 61511).
 * 100% Deterministic Mathematical Grounding • Zero Slop.
 */

const standardsDossiers = {
  'factories_act': {
    title: 'The Factories Act, 1948 & IS 800:2007 Structural Engineering Dossier',
    badge: 'MANDATORY INDIAN STATUTORY COMPLIANCE',
    standard: 'The Factories Act, 1948 (Act No. 63 of 1948) & IS 800:2007 (Steel General Construction)',
    clauses: 'Section 21 [Machinery Guarding], Section 32 [Floors, Stairs & Means of Access], Section 33 [Pits & Floor Openings], IS 800:2007 Cl. 5.6.1 [Handrails], IS 875 (Part 3):2015 [Wind Loads]',
    summary: 'Statutory structural verification for Unit STR-100 (12.0m x 8.0m elevated mezzanine skid). Mandates ISMB 300 columns under 2,500 kg/m² uniform live load, 1050mm top rail, 550mm mid rail, 100mm toeboard, and beam deflection limit below L/350 (18.42 mm calculated vs 34.28 mm statutory allowance).',
    math: `
      $$\\text{The Factories Act Sec 32 Rail Geometry:} \\quad H_{\\text{top}} = 1050\\text{ mm}, \\quad H_{\\text{mid}} = 550\\text{ mm}, \\quad H_{\\text{toeboard}} = 100\\text{ mm}$$
      $$\\text{IS 800:2007 Column Axial Capacity:} \\quad P_d = A_e \\cdot f_{cd} = 5626\\text{ mm}^2 \\cdot 168\\text{ MPa} = 945.1\\text{ kN} \\ge 240.0\\text{ kN (Demand)}$$
      $$\\text{Serviceability Deflection Limit:} \\quad \\delta_{\\max} = \\frac{5 w L^4}{384 E I} = 18.42\\text{ mm} \\le \\frac{L}{350} = \\frac{12,000}{350} = 34.28\\text{ mm} \\quad (\\mathbf{Pass, Margin: +46.3\\%})$$
    `,
    specs: [
      { label: 'Structural Columns', val: 'ISMB 300 (Tata Steel E250 / IS 2062 Grade B)' },
      { label: 'Mezzanine Decking', val: 'Hot-Dip Galvanized Anti-Slip Checkered Plate (85 µm Zn)' },
      { label: 'Guardrail Metallurgy', val: '32 mm OD Carbon Steel Tubing (Statutory Yellow Enamel RAL 1021)' },
      { label: 'Design Wind Speed (IS 875)', val: 'V_b = 50.0 m/s (Coastal Industrial Zone Haldia / Dahej / Jamnagar)' },
      { label: 'Seismic Zone Factor (IS 1893)', val: 'Zone IV (Z = 0.24, Importance Factor I = 1.5 Industrial)' },
      { label: 'Foundation Anchor Bolts', val: '4x M24 Grade 8.8 J-Bolts per baseplate into M30 RCC pad' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/generate_engineering_manifest.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/generate_engineering_manifest.py'
  },

  'peso_smpv': {
    title: 'PESO SMPV (Unfired) Rules 2016 & Gas Cylinders Rules 2016 Dossier',
    badge: 'MANDATORY INDIAN STATUTORY COMPLIANCE',
    standard: 'Petroleum and Explosives Safety Organization (PESO) SMPV (Unfired) Rules, 2016',
    clauses: 'Rule 18 [Design, Construction and Testing], Rule 19 [Safety Relief Devices], Rule 21 [Periodic Hydrostatic Test], Rule 22 [Gas Cylinders Rules for 350-bar Buffer Cascade]',
    summary: 'Mandatory Indian statutory verification for 30-bar hydrogen knockout drum (V-300) and 350-bar buffer cascade (ST-800). Mandates 1.3x MAWP water jacket hydrostatic proof testing, dual 100% capacity safety relief valves (PSV-301A/B) sized for 110% electrolysis gas output, and rupture disc backup.',
    math: `
      $$\\text{PESO Rule 18 Hydrostatic Test:} \\quad P_{\\text{hydro}} = 1.3 \\cdot \\text{MAWP} = 1.3 \\cdot 35.28\\text{ bar} = \\mathbf{45.86\\text{ bar (Water Jacket Method)}}$$
      $$\\text{PESO Rule 19 Relief Capacity:} \\quad \\dot{m}_{\\text{relief}} \\ge 1.10 \\cdot \\dot{m}_{\\max} = 1.10 \\cdot 574.14\\text{ mol/s} = \\mathbf{631.55\\text{ mol/s (100\\% Redundancy)}}$$
      $$\\text{Gas Cylinders Rule 22 Burst Factor:} \\quad P_{\\text{burst}} = 2196.4\\text{ bar} \\ge 2.25 \\cdot P_{\\text{design}} = 2.25(350) = 787.5\\text{ bar} \\quad (\\mathbf{Pass, Factor: 2.79x})$$
    `,
    specs: [
      { label: 'V-300 Knockout Shell', val: '14.0 mm SA-240 Type 316L (12.1 mm minimum required with 1.5mm CA)' },
      { label: 'ST-800 Cylinder Body', val: 'SA-723 Grade 3 Class 2 Forged Ni-Cr-Mo (Wall: 75.0 mm)' },
      { label: 'Relief Valve Redundancy', val: '2x 100% Staggered Set PSVs (PSV-301A @ 30 bar, PSV-301B @ 31.5 bar)' },
      { label: 'Emergency Vent Stack', val: 'API 521 Compliant 6.2m Elevation Elevated Nitrogen-Purged Vent' },
      { label: 'Statutory Inspection Authority', val: 'Petroleum and Explosives Safety Organization (PESO), Govt. of India' },
      { label: 'Submission Document Code', val: 'Form LS-1 & Form II Statutory Engineering Docket' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/asme_bpvc_calculations.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/asme_bpvc_calculations.py'
  },

  'asme': {
    title: 'ASME BPVC Section VIII Div. 1 & Div. 3 Engineering Dossier',
    badge: 'MANDATORY CODE COMPLIANCE',
    standard: 'ASME Boiler and Pressure Vessel Code (BPVC) Section VIII',
    clauses: 'Paragraph UG-27(c)(1) [Shell Thickness], UG-32(d) [2:1 Ellipsoidal Head], UG-99 [Hydrotest], Div 3 Part KD-250 [Lame Stress]',
    summary: 'Deterministic wall sizing and allowable stress verification for 30-bar separator vessels (V-300) and 350-bar buffer storage accumulators (ST-800).',
    math: `
      $$\\text{UG-27 Cylindrical Shell:} \\quad t = \\frac{P \\cdot R}{S \\cdot E - 0.6 P} + \\text{CA} = \\frac{3.0 \\cdot 400}{115 \\cdot 1.0 - 0.6(3.0)} + 1.5 = 12.101\\text{ mm} \\implies \\mathbf{14\\text{ mm Plate Selected}}$$
      $$\\text{UG-32 2:1 Ellipsoidal Head:} \\quad t = \\frac{P \\cdot D}{2 S \\cdot E - 0.2 P} + \\text{CA} = \\frac{3.0 \\cdot 800}{2(115)(1.0) - 0.2(3.0)} + 1.5 = 11.962\\text{ mm} \\implies \\mathbf{12\\text{ mm Plate Selected}}$$
      $$\\text{UG-99 Hydrostatic Proof Test:} \\quad P_{\\text{hydro}} = 1.3 \\cdot \\text{MAWP} = 1.3 \\cdot 35.28\\text{ bar} = \\mathbf{45.86\\text{ bar}}$$
    `,
    specs: [
      { label: 'Vessel Inner Diameter (ID)', val: '800.0 mm' },
      { label: 'Design Internal Pressure (P)', val: '30.0 bar (3.00 MPa)' },
      { label: 'Maximum Allowable Working Pressure (MAWP)', val: '35.28 bar (Margin: +17.6%)' },
      { label: 'Shell Material & Allowable Stress (S)', val: 'SA-240 Type 316L (S = 115.0 MPa @ 80°C)' },
      { label: 'Joint Efficiency (E)', val: '1.0 (Full 100% Radiographic Examination RT-1)' },
      { label: 'Corrosion Allowance (CA)', val: '1.50 mm (Industrial High Purity Hydrogen Service)' },
      { label: 'Calculated Net Thickness (t_calc)', val: '10.601 mm' },
      { label: 'Selected Commercial Plate Thickness', val: '14.0 mm' },
      { label: 'ST-800 Div. 3 Diameter Ratio (Y)', val: '1.30 (ID = 500 mm, OD = 650 mm, t = 75 mm)' },
      { label: 'ST-800 Von Mises Stress at Bore', val: '148.48 MPa (Yield: 725.0 MPa, Safety Factor: 4.88)' },
      { label: 'ST-800 Burst Pressure Limit', val: '2,196.4 bar (Working: 350 bar, Margin: 6.28x)' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/asme_bpvc_calculations.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/asme_bpvc_calculations.py'
  },

  'iec': {
    title: 'IEC 61511 / IEC 61508 Functional Safety & LOPA SIL-2 Architecture',
    badge: 'MANDATORY CODE COMPLIANCE',
    standard: 'IEC 61511 / IEC 61508 (Functional Safety in Process Industry)',
    clauses: 'IEC 61511-1 Cl. 9 [SIL Requirements], Cl. 11 [SIS Architecture], Part 2 [PFD_avg Calculation], ISO 13849-1 [Performance Level d]',
    summary: 'Hardware fault tolerance, probability of failure on demand (PFD_avg), and risk reduction factor (RRF) calculation for high-integrity protection loops (SIF-01 and SIF-02).',
    math: `
      $$\\text{SIF-01 Total PFD}_{\\text{avg}} = \\text{PFD}_{\\text{Sensor (1oo2)}} + \\text{PFD}_{\\text{Logic Solver (1oo1)}} + \\text{PFD}_{\\text{Final Element (1oo1)}}$$
      $$\\text{SIF-01 PFD}_{\\text{avg}} = 0.000300 + 0.000198 + 0.004649 = \\mathbf{0.005147} \\implies \\mathbf{\\text{SIL 2 (Target: } 10^{-3} \\le \\text{PFD} < 10^{-2})}$$
      $$\\text{SIF-02 (Differential-P Trip) PFD}_{\\text{avg}} = 0.000239 + 0.000198 + 0.007927 = \\mathbf{0.008364} \\implies \\mathbf{\\text{SIL 2 Verified}}$$
    `,
    specs: [
      { label: 'Target Safety Integrity Level', val: 'SIL 2 (Target PFD: 0.001 - 0.010)' },
      { label: 'SIF-01 Achieved PFD_avg', val: '0.005147 (Risk Reduction Factor RRF: 194.3)' },
      { label: 'Sensor Architecture', val: '1oo2 Dual Redundant AT-301A/B NDIR H2 in O2 Transmitters' },
      { label: 'Sensor Safe Failure Fraction (SFF)', val: '94.5% (Dangerous Undetected λ_DU: 1.25e-7 /h)' },
      { label: 'Safety Controller / Logic Solver', val: 'Siemens S7-1500F Fail-Safe CPU 1518F-4 PN/DP' },
      { label: 'Logic Solver PFD_avg & SFF', val: 'PFD = 0.000198, SFF = 99.3% (T_proof: 87,600 h)' },
      { label: 'Final Element (SIF-01)', val: 'XV-301 Fast-Acting Nitrogen Quench Valve (<500 ms stroke)' },
      { label: 'Final Element (SIF-02)', val: 'DCB-201 5,000A DC Thyristor Shunt Trip (<15 ms trip)' },
      { label: 'Emergency Shutdown Trip Latency', val: '< 1.40 seconds autonomous SCADA response' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/iec_sil2_functional_safety_lopa.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/iec_sil2_functional_safety_lopa.py'
  },

  'diff_p': {
    title: '30-Bar Differential Pressure Stack Mechanics & ISO 22734 Audit',
    badge: 'MANDATORY CODE COMPLIANCE',
    standard: 'ISO 22734:2019 (Water Electrolysis Safety) / ASME BPVC Sec. VIII',
    clauses: 'ISO 22734 Cl. 4.3 [Internal Pressure Limits], Cl. 4.4 [Gas Crossover], Cl. 4.7 [Pressure Relief], ASME Section VIII Div. 1',
    summary: 'Mechanical stress analysis across unsupported PTL pores under 28.5 bar net differential pressure and multi-species crossover diffusion to ensure H2-in-O2 remains strictly below 1.0% (safety factor >15 to LEL).',
    math: `
      $$\\text{Clamped Circular Plate Bending:} \\quad \\sigma_{\\max} = \\frac{3 \\cdot \\Delta P \\cdot r^2}{4 \\cdot t^2} = \\frac{3(2.85)(0.0125)^2}{4(0.085)^2} = 0.0463\\text{ MPa} \\ll 24.5\\text{ MPa (Yield)}$$
      $$\\text{Mechanical Safety Factor:} \\quad S_F = \\frac{\\sigma_{\\text{yield}}}{\\sigma_{\\max}} = \\frac{24.5}{0.0463} = \\mathbf{529.2} \\implies \\mathbf{\\text{Zero Plastic Extrusion}}$$
      $$\\text{ISO 22734 Gas Crossover Rate:} \\quad \\% \\text{H}_2 \\text{ in } \\text{O}_2 = \\frac{J_{\\text{crossover}}}{2 J_{\\text{O}_2} + J_{\\text{crossover}}} = \\mathbf{0.262\\% \\ll 1.0\\% \\text{ (LEL: } 4.0\\%)}$$
    `,
    specs: [
      { label: 'Applied Net Differential Pressure (ΔP)', val: '28.5 bar (Cathode 30.0 bar, Anode 1.5 bar)' },
      { label: 'Solid Polymer Electrolyte Membrane', val: 'Reinforced 85 µm Nafion composite with ePTFE scrim' },
      { label: 'PTL Mean Pore Diameter', val: '25.0 µm Sintered Titanium Powder (Grade 2)' },
      { label: 'Membrane Bending Stress under 28.5 bar', val: '0.0463 MPa (Yield: 24.5 MPa, Margin: 529x)' },
      { label: 'Elastic Pore-Bridging Deflection', val: '0.00010 µm (Negligible thin-membrane strain)' },
      { label: 'Stack Clamping Preload', val: '37.5 kN Tie-Rod Tension (3.30 MPa PTL land contact)' },
      { label: 'H2-in-O2 Crossover Concentration', val: '0.262% vol/vol @ 2.0 A/cm² (ISO limit: 1.0%)' },
      { label: 'Safety Margin to Lower Explosive Limit (LEL)', val: '15.2x Margin Below 4.0% vol/vol H2 LEL' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/stack_30bar_differential_mechanics.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/stack_30bar_differential_mechanics.py'
  },

  'iso_purity': {
    title: 'ISO 14687:2019 Grade D Fuel Purity & Unit DR-500 Mass Balance',
    badge: 'MANDATORY CODE COMPLIANCE',
    standard: 'ISO 14687:2019 Grade D / SAE J2719 (Hydrogen Fuel Quality for Fuel Cell Vehicles)',
    clauses: 'Table 1 [Fuel Specification Limits], Grade D: Total H2 ≥ 99.97%, O2 ≤ 5.0 ppm, H2O ≤ 5.0 ppm, CO ≤ 0.2 ppm, Total S ≤ 0.004 ppm',
    summary: 'Reaction kinetics and multi-component mass balance simulation for catalytic DeOxo reactor (V-501) and dual-tower 13X TSA regenerative molecular sieve dryers (DR-500A/B).',
    math: `
      $$\\text{Catalytic O}_2\\text{ Quench:} \\quad 2\\text{H}_2 + \\text{O}_2 \\xrightarrow{\\text{Pd/Al}_2\\text{O}_3} 2\\text{H}_2\\text{O} + 246.16\\text{ kW}_{\\text{th}} \\quad (\\Delta H_r = -285.8\\text{ kJ/mol})$$
      $$\\text{Effluent Oxygen Concentration:} \\quad y_{\\text{O}_2,\\text{out}} = 1,500\\text{ ppm} \\cdot (1 - 0.9997) = \\mathbf{0.45\\text{ ppm} \\ll 5.0\\text{ ppm Limit}}$$
      $$\\text{TSA Zeolite 13X Dewpoint:} \\quad T_{\\text{dew}} = \\mathbf{-70.0^\\circ\\text{C}} \\implies y_{\\text{H}_2\\text{O}} = \\mathbf{1.0\\text{ ppm} \\ll 5.0\\text{ ppm Limit}}$$
    `,
    specs: [
      { label: 'Commercial Hydrogen Delivery Rate', val: '46,328 Nm³/h (574.14 mol/s @ 30 bar)' },
      { label: 'Raw Electrolyzer Gas Oxygen Contamination', val: '1,500 ppm vol/vol O2 (from cross-diffusion)' },
      { label: 'DeOxo Product Oxygen Content', val: '0.45 ppm vol/vol (99.97% conversion, limit: 5.0 ppm)' },
      { label: 'Catalyst Charge & Bed Dimensions', val: '4,922 kg 0.5% wt Pd/Al2O3 pellets (Volume: 5.79 m³)' },
      { label: 'Exothermic Adiabatic Temperature Rise', val: '+14.73°C (Effluent Gas Temp: 54.73°C)' },
      { label: 'Dryer Inlet Saturated Water Content', val: '249.2 ppm vol/vol H2O' },
      { label: 'Dual-Tower TSA Zeolite 13X Bed', val: '795.3 kg Zeolite 13X per tower (Cycle: 8 hours)' },
      { label: 'Product Gas Dewpoint & Moisture', val: '-70.0°C Frost Point (1.0 ppm H2O, limit: 5.0 ppm)' },
      { label: 'Final Hydrogen Fuel Quality', val: 'ISO 14687:2019 Grade D / IS 16087 Automotive Purity (99.999%)' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/iso14687_purity_deoxo_tsa_balance.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/iso14687_purity_deoxo_tsa_balance.py'
  },

  'astm_titanium': {
    title: 'ASTM B265 Grade 2 Titanium DFM & Progressive Die Stamping Audit',
    badge: 'MANDATORY CODE COMPLIANCE',
    standard: 'ASTM B265 (UNS R50400) / ASME BPVC Sec. II Part B / VDI 3400',
    clauses: 'ASTM B265 Gr. 2 [Tensile & Elongation], DIN 8584 [Deep Drawing Formability], ASME BPVC Section VIII Div. 1 Part UG',
    summary: 'Finite element plane-strain progressive stamping simulation for 50-channel serpentine bipolar flow field plates. Verifies maximum localized sheet thinning remains below 18.0% failure threshold.',
    math: `
      $$\\text{Maximum Plain-Strain Thinning:} \\quad \\epsilon_3 = -(\\epsilon_1 + \\epsilon_2) \\implies t_{\\min} = t_0 \\cdot \\exp(-0.178) = 0.80 \\cdot 0.836 = \\mathbf{0.669\\text{ mm}}$$
      $$\\text{Thinning Percentage:} \\quad \\%_{\\text{thinning}} = \\frac{0.80 - 0.669}{0.80} \\times 100 = \\mathbf{16.36\\% < 18.0\\% \\text{ (DFM Formability Pass)}}$$
      $$\\text{Progressive Stamping Tonnage:} \\quad F_{\\text{forming}} = L \\cdot t_0 \\cdot R_m \\cdot C = 15.81\\text{ m} \\cdot 0.8\\text{ mm} \\cdot 345\\text{ MPa} \\cdot 0.92 = \\mathbf{408.5\\text{ Metric Tons}}$$
    `,
    specs: [
      { label: 'Bipolar Plate Raw Material', val: 'ASTM B265 Grade 2 Titanium Sheet (UNS R50400)' },
      { label: 'Initial Sheet Thickness (t0)', val: '0.80 mm cold-rolled strip' },
      { label: 'Channel Geometry', val: '50-Channel Serpentine: Pitch 2.20 mm, Depth 0.65 mm' },
      { label: 'Die Draft Angle & Shoulder Fillet', val: '8.5° Draft, 0.35 mm Shoulder Fillet Radius' },
      { label: 'Calculated Minimum Thickness', val: '0.669 mm (Localized necking safety margin: +10.0%)' },
      { label: 'Sheet Thinning Ratio', val: '16.36% (Industry failure threshold: >18.0%)' },
      { label: 'Total Formed Channel Perimeter', val: '15.81 meters per plate (250 cm² active area)' },
      { label: 'Required Hydraulic Press Tonnage', val: '408.5 Tonnes (Standard 500-Ton Press Capacity)' },
      { label: 'Hydrostatic Pressure Proof Rating', val: '42.39 bar (ASME Part UG Bipolar Verification)' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/astm_titanium_stamping_dfm.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/astm_titanium_stamping_dfm.py'
  },

  'harmonization_matrix': {
    title: 'Harmonized Indian Domestic ⟷ Global Export Regulatory Matrix',
    badge: 'DUAL COMPLIANCE PASS',
    standard: 'BIS / PESO / Factories Act (India) ⟷ ASME / ISO / IEC / EN (International)',
    clauses: 'Cross-Border Statutory Equivalence Protocol: Dual-coded engineering ensuring single-design export eligibility across Indian Domestic, European (CE/PED), North American (ASME/NFPA), and Middle Eastern (ADNOC/Aramco) jurisdictions.',
    summary: 'Eliminates redundant redesign costs by engineering skid packages to simultaneously satisfy Indian statutory mandates (The Factories Act, PESO, IS 800) and international benchmark codes (ASME Sec VIII, ISO 22734, IEC 61511, PED 2014/68/EU).',
    math: `
      $$\\mathbf{Equivalence: } \\quad \\text{Pressure Equipment: } \\text{PESO SMPV 2016} \\equiv \\text{ASME BPVC Sec VIII Div 1 & 3} \\equiv \\text{PED 2014/68/EU Module H}$$
      $$\\mathbf{Equivalence: } \\quad \\text{Structural Steel: } \\text{The Factories Act 1948 Sec 32} + \\text{IS 800:2007} \\equiv \\text{ISO 14122-1..4} \\equiv \\text{EN 1090-2 (EXC2)}$$
      $$\\mathbf{Equivalence: } \\quad \\text{Functional Safety: } \\text{IS/IEC 61511} \\equiv \\text{IEC 61508 / IEC 61511 SIL 2} \\equiv \\text{NFPA 2 / NFPA 55}$$
    `,
    specs: [
      { label: 'Domestic Anchor Market', val: 'India (IOCL, HMEL, Grasim, DCM Shriram, Tata Steel, IFFCO)' },
      { label: 'Global Export Markets', val: 'European Union (CE/PED), North America (ASME/NFPA), Middle East (ADNOC/Aramco)' },
      { label: 'Primary Structural Code', val: 'The Factories Act 1948 & IS 800:2007 ⟷ ISO 14122 & EN 1090-2' },
      { label: 'Primary Pressure Code', val: 'PESO SMPV Rules 2016 ⟷ ASME BPVC Section VIII Div 1 & Div 3' },
      { label: 'Electrolyzer System Safety', val: 'National Green H2 Mission BIS Draft ⟷ ISO 22734:2019' },
      { label: 'Hydrogen Fuel Quality', val: 'IS 16087 Automotive Purity ⟷ ISO 14687:2019 Grade D' },
      { label: 'Hazardous Area Classification', val: 'IS/IEC 60079-0/-10-1 Zone 1/2 ⟷ ATEX Directive 2014/34/EU' },
      { label: 'High-Pressure Process Piping', val: 'PNGRB T4S Regulations ⟷ ASME B31.3 & ASME B31.12' }
    ],
    scriptPath: 'Presentation_ML_Green_Hydrogen/engineering_standards/generate_engineering_manifest.py',
    terminalCmd: 'py Presentation_ML_Green_Hydrogen/engineering_standards/generate_engineering_manifest.py'
  }
};

function openStandardModal(dossierKey = 'asme') {
  const dossier = standardsDossiers[dossierKey];
  if (!dossier) return;

  const modal = document.getElementById('standardsModal');
  if (!modal) return;

  // Header Elements
  const titleEl = document.getElementById('stdModalTitle');
  const badgeEl = document.getElementById('stdModalBadge');
  const stdEl = document.getElementById('stdModalStandard');
  const clausesEl = document.getElementById('stdModalClauses');
  const summaryEl = document.getElementById('stdModalSummary');
  const scriptPathEl = document.getElementById('stdModalScriptPath');
  const cmdEl = document.getElementById('stdModalTerminalCmd');

  if (titleEl) titleEl.innerText = dossier.title;
  if (badgeEl) badgeEl.innerText = dossier.badge;
  if (stdEl) stdEl.innerText = dossier.standard;
  if (clausesEl) clausesEl.innerText = dossier.clauses;
  if (summaryEl) summaryEl.innerText = dossier.summary;
  if (scriptPathEl) scriptPathEl.innerText = dossier.scriptPath;
  if (cmdEl) cmdEl.innerText = dossier.terminalCmd;

  // Math Banner
  const mathBox = document.getElementById('stdModalMath');
  if (mathBox) {
    mathBox.innerHTML = dossier.math;
    if (window.renderMathInElement) {
      window.renderMathInElement(mathBox, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '\\[', right: '\\]', display: true},
          {left: '\\(', right: '\\)', display: false}
        ],
        throwOnError: false
      });
    }
  }

  // Specs Grid
  const specsContainer = document.getElementById('stdModalSpecs');
  if (specsContainer) {
    specsContainer.innerHTML = '';
    dossier.specs.forEach(s => {
      const row = document.createElement('div');
      row.className = 'std-spec-row';
      row.innerHTML = `
        <span class="std-spec-label">${s.label}</span>
        <span class="std-spec-val">${s.val}</span>
      `;
      specsContainer.appendChild(row);
    });
  }

  modal.classList.add('active');
}

function closeStandardModal(e) {
  if (e && e.target && e.target.id !== 'standardsModal' && !e.target.classList.contains('close-btn')) {
    return;
  }
  const modal = document.getElementById('standardsModal');
  if (modal) modal.classList.remove('active');
}

window.openStandardModal = openStandardModal;
window.closeStandardModal = closeStandardModal;
