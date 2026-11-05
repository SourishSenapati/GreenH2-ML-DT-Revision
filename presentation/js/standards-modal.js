/**
 * HYDROSENSE™ Engineering Standards & Compliance Verification Modal
 * Binds the 5 Hero credential tags directly to deterministic mathematical calculations,
 * ASME / IEC / ISO / ASTM code clauses, and repository Python scripts.
 */

const standardsDossiers = {
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
      { label: 'Design Internal Pressure', val: '30.0 bar (3.0 MPa)' },
      { label: 'Material Specification', val: 'SA-240 Type 316L Stainless Steel' },
      { label: 'Allowable Stress (S @ 80°C)', val: '115.0 MPa (ASME Sec. II Part D Tab. 1A)' },
      { label: 'Joint Efficiency (E)', val: '1.0 (Full Radiographic Exam UW-11(a))' },
      { label: 'Corrosion Allowance (CA)', val: '1.50 mm (Wet H2 Service)' },
      { label: 'Max Allowable Working Pressure', val: '35.28 bar (MAWP)' },
      { label: 'Div 3 ST-800 Burst Pressure', val: '2,196.4 bar (Safety Margin: 4.88x)' }
    ],
    scriptPath: 'repo_temp/code/engineering_standards/asme_bpvc_calculations.py',
    terminalCmd: 'py repo_temp/code/engineering_standards/asme_bpvc_calculations.py'
  },
  'iec': {
    title: 'IEC 61511 / IEC 61508 Functional Safety & SIL-2 LOPA Dossier',
    badge: 'SIL-2 LOPA VERIFIED',
    standard: 'IEC 61511-1 / IEC 61508-2 (Low Demand Safety Instrumented Systems)',
    clauses: 'Clause 9 [Safety Lifecycle], Clause 11 [SIL Verification & PFD_avg Architecture], Clause 12 [Application Software]',
    summary: 'Quantitative Layer of Protection Analysis (LOPA) calculating Probability of Failure on Demand average (PFD_avg) and Risk Reduction Factor (RRF) for 1oo2 voting architectures.',
    math: `
      $$\\text{1oo2 Redundant Architecture:} \\quad \\mathrm{PFD}_{\\text{avg}} = \\frac{(\\lambda_{\\mathrm{DU}})^2 \\cdot T_1^2}{3} + \\beta \\cdot \\lambda_{\\mathrm{DU}} \\cdot \\frac{T_1}{2}$$
      $$\\text{SIF-01 Crossover Quench:} \\quad \\mathrm{PFD}_{\\text{avg}} = 5.15 \\times 10^{-3} \\implies \\mathbf{\\mathrm{RRF} = 194.3} \\quad [100 \\le \\mathrm{RRF} < 1000 \\implies \\mathbf{\\text{SIL-2 VERIFIED}}]$$
      $$\\text{Safe Failure Fraction (SFF):} \\quad \\mathrm{SFF}_{\\text{sensor}} = 94.5\\%, \\quad \\mathrm{SFF}_{\\text{PLC}} = 99.3\\%, \\quad \\mathrm{SFF}_{\\text{ESD}} = 86.7\\%$$
    `,
    specs: [
      { label: 'Target Functional Safety Level', val: 'IEC 61511 SIL 2' },
      { label: 'SIF-01 Sensor Architecture', val: '1oo2 Dual NDIR Fast H2 Analyzers' },
      { label: 'Logic Solver Architecture', val: 'Siemens S7-1500F Safety PLC (SIL 3 rated)' },
      { label: 'Final Element Response Time', val: '< 500 ms (Spring-return fail-closed ESD)' },
      { label: 'SIF-02 Differential-P PFD_avg', val: '0.008360 (Achieved RRF: 119.6)' },
      { label: 'Dangerous Undetected Rate', val: '1.20 × 10⁻⁶ failures / hr' },
      { label: 'Common Cause Beta Factor', val: '5.0% (β = 0.05)' },
      { label: 'Proof Test Interval (T1)', val: '8,760 hours (1 year full stroke)' }
    ],
    scriptPath: 'repo_temp/code/engineering_standards/iec_61511_sil2_lopa_safety.py',
    terminalCmd: 'py repo_temp/code/engineering_standards/iec_61511_sil2_lopa_safety.py'
  },
  'diff_p': {
    title: '30-Bar Differential Pressure Stack Mechanics & Crossover Dossier',
    badge: 'ISO 22734 / NFPA 2 COMPLIANT',
    standard: 'ISO 22734:2019 (Water Electrolysis) & NFPA 2 (Hydrogen Technologies Code)',
    clauses: 'Clause 4.2.3 [Cross-contamination and flammable mixtures], Clause 4.3.2 [Differential Pressure Structural Limits]',
    summary: 'Pore-bridging membrane deflection under 28.5 bar Delta-P and Fickian/Darcy electrochemical gas crossover ensuring H2-in-O2 remains <0.3 vol% (well below 4.0 vol% LEL).',
    math: `
      $$\\text{Pore-Bridging Membrane Deflection:} \\quad w_{\\text{max}} = \\frac{q \\cdot r_p^4}{64 \\cdot D} = \\frac{(2.85 \\times 10^6)(12.5 \\times 10^{-6})^4}{64(1.61 \\times 10^{-5})} = \\mathbf{0.0001\\ \\mu\\text{m}}$$
      $$\\text{Total Crossover Flux:} \\quad J_{\\text{total}} = \\frac{P_{\\text{perm}}}{t_m} \\Delta P + \\frac{k_{\\text{Darcy}}}{\\mu} c_{\\text{sat}} \\frac{\\Delta P}{t_m} = \\mathbf{1.363 \\times 10^{-4}\\text{ mol/(m}^2\\text{s)}}$$
      $$\\text{Flammability Safety Margin:} \\quad \\%\\mathrm{H}_2\\text{ in }\\mathrm{O}_2 = \\mathbf{0.262\\ \\text{vol}\\%} \\ll 1.0\\% \\text{ limit} \\implies \\mathbf{15.2\\times \\text{ Margin to LEL}}$$
    `,
    specs: [
      { label: 'Cathode Delivery Pressure', val: '30.0 bar (3.0 MPa direct feed)' },
      { label: 'Anode Loop Pressure', val: '1.50 bar (0.15 MPa circulating water)' },
      { label: 'Differential Pressure (Delta-P)', val: '28.5 bar (2.85 MPa continuous load)' },
      { label: 'Membrane Specification', val: '85 µm ePTFE-Reinforced Composite PFSA' },
      { label: 'PTL Mean Pore Diameter', val: '25.0 µm (Sintered Titanium Powder)' },
      { label: 'Membrane Safety Factor on Yield', val: '530x (Max Stress: 0.046 MPa vs 24.5 MPa Yield)' },
      { label: 'Stack Clamping Contact Stress', val: '3.3 MPa (Optimal 2.0 to 4.0 MPa range)' },
      { label: 'Hydrogen Crossover Concentration', val: '0.262 vol% H2 in O2 (Limit: <1.0 vol%)' }
    ],
    scriptPath: 'repo_temp/code/engineering_standards/stack_30bar_differential_mechanics.py',
    terminalCmd: 'py repo_temp/code/engineering_standards/stack_30bar_differential_mechanics.py'
  },
  'iso_purity': {
    title: 'ISO 14687:2019 Grade D Fuel Cell Vehicle Hydrogen Purity Dossier',
    badge: 'FUEL CELL GRADE D PASS',
    standard: 'ISO 14687:2019 / SAE J2719 (Hydrogen Fuel Quality - Grade D)',
    clauses: 'Table 1: Impurity limits for PEM fuel cell road vehicles (O2 <= 5.0 ppm, H2O <= 5.0 ppm, Purity >= 99.97%)',
    summary: 'Catalytic DeOxo exothermic reactor (0.5% Pd/Al2O3) and twin 13X zeolite temperature swing adsorption (TSA) molecular sieve drying towers.',
    math: `
      $$\\text{DeOxo Stoichiometric Reaction:} \\quad 2\\mathrm{H}_2 + \\mathrm{O}_2 \\xrightarrow{\\mathrm{Pd/Al}_2\\mathrm{O}_3} 2\\mathrm{H}_2\\mathrm{O} \\quad (\\Delta H = -285.83\\text{ kJ/mol})$$
      $$\\text{Adiabatic Bed Temperature Rise:} \\quad \\Delta T = \\frac{y_{\\mathrm{O}_2} \\cdot (-\\Delta H)}{C_p} = \\frac{0.0015 \\cdot 285830}{29.1} = \\mathbf{14.73^\\circ\\text{C}}$$
      $$\\text{TSA Molecular Sieve Moisture Stripping:} \\quad \\mathrm{H}_2\\mathrm{O}_{\\text{effluent}} = \\mathbf{1.0\\text{ ppm}} \\implies \\text{Dew Point } \\mathbf{\\le -70.0^\\circ\\text{C}}$$
    `,
    specs: [
      { label: 'Hydrogen Fuel Quality Grade', val: 'ISO 14687:2019 Grade D (PEMFC)' },
      { label: 'Hydrogen Purity Index', val: '≥ 99.97 mol%' },
      { label: 'Inlet Raw O2 Concentration', val: '1,500 ppm (Knockout drum effluent)' },
      { label: 'Purified Product O2 Level', val: '0.5 ppm (< 5.0 ppm ISO standard limit)' },
      { label: 'Purified Product Moisture (H2O)', val: '1.0 ppm (< 5.0 ppm ISO standard limit)' },
      { label: 'Atmospheric Dew Point', val: '≤ -70.0°C (Deep Cryogenic Dryness)' },
      { label: 'DeOxo Catalyst Mass', val: '4,922 kg (0.5 wt% Pd / Al2O3)' },
      { label: 'TSA 13X Zeolite Sorbent Mass', val: '795.3 kg per tower (Dual 12h cycle)' }
    ],
    scriptPath: 'repo_temp/code/engineering_standards/iso_14687_grade_d_purification.py',
    terminalCmd: 'py repo_temp/code/engineering_standards/iso_14687_grade_d_purification.py'
  },
  'astm_titanium': {
    title: 'ASTM B265 Grade 2 Titanium Mechanical & DFM Stamping Dossier',
    badge: 'ASTM B265 DFM VERIFIED',
    standard: 'ASTM B265 (Strip, Sheet & Plate) / ASME BPVC Sec. II Part D Table 1B / DIN 50125',
    clauses: 'Table 2 Mechanical Requirements, DIN Sheet Metal Forming Limits (<18% Thinning Strain)',
    summary: 'Progressive stamping strain analysis for 50-channel bipolar flow fields (250 cm² active area) and ASME Section VIII UG-99 hydrotest verification.',
    math: `
      $$\\text{Engineering Forming Strain:} \\quad e_{\\text{eng}} = \\frac{L_{\\text{developed}} - P_{\\text{pitch}}}{P_{\\text{pitch}}} = \\frac{3.32 - 2.20}{2.20} = \\mathbf{50.92\\%}$$
      $$\\text{Thinning with Normal Anisotropy (}\\bar{r}=1.85\\text{):} \\quad t_{\\text{thinning}} = \\mathbf{16.36\\%} \\quad [\\le 18.0\\% \\implies \\mathbf{\\text{DFM SAFE - ZERO NECKING}}]$$
      $$\\text{Hydraulic Press Forming Force:} \\quad F_{\\text{forming}} = 0.55 \\cdot L_{\\text{total}} \\cdot t_0 \\cdot S_u \\left(1 + \\frac{\\mu}{\\tan \\theta}\\right) = \\mathbf{4,005.8\\text{ kN}} \\implies \\mathbf{408.5\\text{ Ton Press}}$$
    `,
    specs: [
      { label: 'Material Specification', val: 'ASTM B265 Grade 2 (UNS R50400)' },
      { label: 'Room Temp Yield Strength', val: '275.0 MPa (S_y @ 20°C)' },
      { label: 'Room Temp Tensile Strength', val: '345.0 MPa (S_u @ 20°C)' },
      { label: 'ASME Allowable Stress @ 80°C', val: '115.0 MPa (Sec. II Part D Tab. 1B)' },
      { label: 'Sheet Initial Thickness (t0)', val: '0.80 mm Annealed Titanium Sheet' },
      { label: 'Bipolar Channel Pitch / Depth', val: 'Pitch: 2.20 mm | Depth: 0.65 mm' },
      { label: 'Draft Angle / Shoulder Radius', val: 'Draft: 8.5° | Fillet Radius: 0.35 mm' },
      { label: 'Bipolar Envelope Hydrotest', val: '42.39 bar (ASME UG-99 Proof Test)' }
    ],
    scriptPath: 'repo_temp/code/engineering_standards/astm_b265_titanium_dfm.py',
    terminalCmd: 'py repo_temp/code/engineering_standards/astm_b265_titanium_dfm.py'
  }
};

function openStandardModal(key) {
  const dossier = standardsDossiers[key];
  if (!dossier) return;

  const modal = document.getElementById('standardsModal');
  if (!modal) return;

  document.getElementById('stdModalTitle').innerText = dossier.title;
  document.getElementById('stdModalBadge').innerText = dossier.badge;
  document.getElementById('stdModalStandard').innerText = dossier.standard;
  document.getElementById('stdModalClauses').innerText = dossier.clauses;
  document.getElementById('stdModalSummary').innerText = dossier.summary;
  document.getElementById('stdModalScriptPath').innerText = dossier.scriptPath;
  document.getElementById('stdModalTerminalCmd').innerText = dossier.terminalCmd;

  // Math proof
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

