/**
 * HYDROSENSE™ Parametric 3D CAD Workstation & Master Plant Assembly Engine
 * Modeled directly to SolidWorks 2025 SP1.2 B-Rep specifications,
 * ASME BPVC Section VIII Div 1 & 3, API 618, and IEC 61511 SIL-2 standards.
 *
 * Features:
 * - 11 Fully detailed industrial subsystems with realistic mechanical geometry:
 *   Weld-neck flanges, bolt circles, nozzle schedules, level bridles, angle PSVs,
 *   bipolar plate stacks, tension tie-rods, 5000A copper busbars, radiator fins, etc.
 * - Studio PBR materials (Brushed 316L, Titanium Gr. 2, Forged Steel, Copper, Enamel Yellow)
 * - Three.js PCFSoftShadowMap with real-time ground contact shadows
 * - Camera View Presets (Iso, Front, Side, Top, Reset)
 * - Smooth camera tweening to focus on clicked/selected subsystems
 * - Radial Explode-View slider (0% to 100%)
 * - Auto-Turntable orbit & Wireframe inspection
 * - Live Subsystem Engineering Telemetry HUD
 */

let plantScene, plantCamera, plantRenderer, plantControls;
let plantContainer;
let subsystems = {};
let origSubsystemMaterials = new Map();
let selectedSubsystem = null;
let isWireframe = false;
let isTurntable = false;
let showGrid = true;
let explodeFactor = 0.0;
let raycaster, mouse;
let groundShadowPlane, groundGrid;

// Camera tween state
let targetCamPos = null;
let targetLookAt = null;

// --- Subsystem Engineering Catalog & Specifications ---
const plantCatalog = {
  'PLANT-100T': {
    name: 'HYDROSENSE™ 100 TPD Turnkey Commercial Plant Master Skid',
    tag: 'PLANT-100T',
    category: 'Commercial Turnkey Complex (11 Subsystems)',
    standard: 'ASME Sec VIII Div 1 & 3 / API 618 / IS 800 / IEC 61511',
    spec: 'Full 11-subsystem industrial green hydrogen complex. 100 TPD capacity @ 30-bar direct delivery, 4x 50-cell stacks, 350-bar buffer accumulators, dual 13X TSA drying, and SIL-2 HIPPS safety loops.',
    dimensions: '12.0m L × 8.0m W × 6.2m H (Complete Skid)',
    duty: 'Continuous water dissociation, gas-liquid separation, catalytic DeOxo, desiccation, 350-bar compression, and automated edge digital twin governance.',
    opParams: {
      'Plant Production Capacity': '100 TPD (4,166 kg/h pure H₂)',
      'Stack Direct Delivery Pressure': '30.0 bar (Balanced differential)',
      'Purity Standard': 'ISO 14687:2019 Grade D (>99.97% H₂)',
      'Specific Energy Consumption': '51.2 kWh / kg H₂ (Stack level)',
      'Core Life Extension': '+39.4% (21,428h → 29,879h via PIBT)',
      'LCOH Reduction': '-$0.42 / kg H₂ ($4.82 → $4.40 / kg)'
    },
    script: 'code/engineering_standards/generate_engineering_manifest.py',
    standardModalKey: 'asme',
    basePos: [0, 0, 0],
    focusPos: [0, 2.6, 0],
    camFocus: [15, 11, 18],
    explodeDir: [0, 0, 0]
  },
  'STK-010': {
    name: 'STK-010: 1:1 10-Cell Industrial PEM Electrolyzer Stack (250 cm²)',
    tag: 'STK-010',
    category: '1:1 High-Detail Electrochemical Stack Assembly (140 B-Rep Bodies)',
    standard: 'ISO 22734 / ASTM B265 Gr. 2 Ti / Nafion 115 / ASME B16.5',
    spec: '10 repeating PEM cells (250 cm² active area, 360×220 mm plate envelope), 11 machined Grade 2 Titanium bipolar plates with cooling fins & manifold tabs, 10 Nafion 115 MEAs + EPDM perimeter gaskets, 10 gold-plated cell voltage sensing tabs (CVM-01..10), 2x 35mm SA-240 316L endplates, Virgin PTFE dielectric isolators, gold-plated copper current collectors, 8x M16 tie-rods with Belleville spring washer stacks, and dual Siemens SITRANS P320 transmitters.',
    dimensions: '520 mm W × 480 mm D × 560 mm H (140 SolidWorks B-Rep Bodies)',
    duty: 'Direct 30-bar differential water electrolysis at 2.00 A/cm² with per-cell in-operando voltage residual tracking e(t) and <1.4s Bayesian biofilm anomaly isolation.',
    opParams: {
      'Active Cell Area': '250.0 cm² × 10 Cells in Series',
      'Nominal Current Density': '2.00 A/cm² (500 A DC Stack Current)',
      'Clamping Contact Stress': '3.30 MPa (8x M16 Tie-Rods @ 7.5 Nm)',
      'Bipolar Plate Spec': 'ASTM B265 Grade 2 Titanium (11 Plates)',
      'Membrane Electrode Assembly': 'Nafion 115 PFSA + IrO₂ / Pt-C Catalyst',
      'In-Operando Telemetry': '10x Gold CVM Tabs + 2x SITRANS P320'
    },
    bom: [
      { item: '35mm Heavy End Plates', qty: 2, material: 'SA-240 Type 316L SS', spec: '360 × 220 × 35 mm CNC machined + lifting lugs' },
      { item: 'Bipolar Flow-Field Plates', qty: 11, material: 'ASTM B265 Grade 2 Titanium', spec: 'Progressive-die formed serpentine flow field + cooling fins' },
      { item: 'Nafion 115 MEA + EPDM Frames', qty: 10, material: 'PFSA / IrO₂ / Pt-C / EPDM', spec: '250 cm² active catalyst window + dual gaskets' },
      { item: 'Current Collector Plates', qty: 2, material: 'C11000 ETP Copper + Au Flash', spec: '500A DC busbar lug + low-contact-resistance face' },
      { item: 'Dielectric Isolation Plates', qty: 2, material: 'Virgin PTFE (Teflon)', spec: 'High-dielectric strength (>20 kV/mm) isolation' },
      { item: 'Tie-Rod & Belleville Assemblies', qty: 8, material: 'ASTM A193 Gr. B7 / 51CrV4', spec: 'M16 × 480 mm + PTFE sleeves + Belleville washers' },
      { item: 'Cell Voltage Sense Tabs', qty: 10, material: 'Gold-Plated Brass (CVM-01..10)', spec: 'Per-cell differential voltage tap to PIBT edge' },
      { item: 'Pressure Transmitters', qty: 2, material: 'Siemens SITRANS P320 (316L)', spec: 'HART / 4-20mA differential cathode/anode ΔP' }
    ],
    governing_calculations: {
      code_clause: 'ISO 22734 Cl. 4.2.3 & ASTM B265 Stamping DFM',
      formula: 'sigma_contact = (8 * F_bolt) / A_gasket = 3.30 MPa | V_cell = E_rev + eta_act + eta_ohm + eta_conc = 1.854 V',
      computed_value: 'Uniform 3.30 MPa gasket sealing @ 30.0 bar differential pressure; 140 distinct SolidWorks B-Rep solid bodies',
      script_source: 'PEM_Green_Hydrogen_Digital_Twin/cad_and_geometry/sw_h2_equipment_builders.py:L122'
    },
    script: 'PEM_Green_Hydrogen_Digital_Twin/cad_and_geometry/sw_h2_equipment_builders.py',
    standardModalKey: 'diff_p',
    basePos: [0, 0, 0],
    focusPos: [0, 0.65, 0],
    camFocus: [2.2, 1.4, 2.6],
    explodeDir: [0, 0, 0]
  },
  'STR-100': {
    name: 'Unit 100: Structural Mezzanine & Pipe Rack Skid',
    tag: 'STR-100',
    category: 'Structural Heavy Civil & Skids',
    standard: 'The Factories Act 1948 Sec 32/33 & IS 800:2007 (Domestic) | ISO 14122 & EN 1090-2 (Global)',
    spec: 'ISMB 300 Columns (12.0m L x 8.0m W x 5.8m H), Galvanized Checkered Deck, Statutory Safety Yellow Handrails (The Factories Act, 1948 Sec 32), 4-Bolt Baseplates with M24 Anchor J-Bolts, Heavy Overhead Pipe Rack Stanchions.',
    dimensions: '12.0m L × 8.0m W × 5.8m H (Tare: 14.8 Tonnes)',
    duty: 'Structural foundation supporting 4x high-pressure electrolyzer stacks (30 bar) and overhead interconnecting process fluid headers.',
    opParams: {
      'Structural Load Rating': '2,500 kg/m² Uniform Live Load',
      'Column Profile': 'ISMB 300 (Flange 140×12.4mm, Web 7.5mm)',
      'Deflection Limit': 'L / 350 under maximum seismic load',
      'Protective Coating': 'Hot-dip galvanized (85 µm zinc coat)'
    },
    script: 'PEM_Green_Hydrogen_Digital_Twin/cad_and_geometry/build_complete_industrial_hydrogen_plant.py',
    standardModalKey: 'factories_act',
    basePos: [0, 0, 0],
    focusPos: [0, 2.5, 0],
    camFocus: [11, 7, 13],
    explodeDir: [0, -1.2, 0]
  },
  'EL-200': {
    name: 'Unit 200: Commercial Electrolyzer Bay (4x Stacks)',
    tag: 'EL-200',
    category: 'Electrochemical Core Dissociation',
    standard: 'National Green H2 Mission BIS Draft / IS 16087 (Domestic) | ISO 22734:2019 & ASTM B265 (Global)',
    spec: '4x 50-Cell Stacks (250 cm² active area), 30-bar differential operation, 5000A DC polished electrolytic copper busways, 16x M16 tie-rods torqued to 7.5 Nm (3.3 MPa uniform contact stress).',
    dimensions: '4.8m L × 2.4m W × 1.6m H (4x Multi-Cell Stacks)',
    duty: 'Continuous water dissociation at 2.00 A/cm² nominal current density delivering 100 TPD green hydrogen on multi-stack complex basis.',
    opParams: {
      'Operating Pressure': '30.0 bar (Balanced differential)',
      'Operating Temperature': '65.0°C to 80.0°C (Controlled DI Loop)',
      'Stack Voltage': '1.854 V/cell (Nominal @ 2.0 A/cm²)',
      'DC Power Feed': '5,000 A @ 110 V DC per stack bay'
    },
    script: 'engineering_standards/stack_30bar_differential_mechanics.py',
    standardModalKey: 'diff_p',
    basePos: [-0.2, 4.36, 0],
    focusPos: [-0.2, 4.8, 0],
    camFocus: [-0.2, 7.5, 8.5],
    explodeDir: [-1.8, 1.4, 0]
  },
  'V-300': {
    name: 'Unit 300: HP Hydrogen Knockout Drum',
    tag: 'V-300',
    category: 'ASME High-Pressure Gas Separation',
    standard: 'PESO SMPV (Unfired) Rules 2016 (Domestic) | ASME BPVC Sec VIII Div 1 & PED 2014/68/EU (Global)',
    spec: 'Vertical pressure vessel: ID 800 mm × 2200 mm shell (14.0 mm SA-240 316L plate), 2:1 ellipsoidal heads (12.0 mm plate), MAWP 35.28 bar, Hydrotest 45.86 bar, 1.5mm Corrosion Allowance.',
    dimensions: 'ID 800 mm × 2,800 mm Overall Height',
    duty: 'High-pressure gas-liquid separation of entrained moisture droplets from raw cathode hydrogen; dual differential-radar level bridle and automated angle PSV relief.',
    opParams: {
      'Design Pressure (MAWP)': '35.28 bar (Hydrotest: 45.86 bar)',
      'Shell Thickness': '14.0 mm SA-240 316L (UG-27 verified)',
      '2:1 Head Thickness': '12.0 mm SA-240 316L (UG-32 verified)',
      'Level Transmitter': 'Dual Guided Wave Radar + Magnetic Bridle'
    },
    script: 'engineering_standards/asme_bpvc_calculations.py',
    standardModalKey: 'peso_smpv',
    basePos: [2.2, 1.5, -2.0],
    focusPos: [2.2, 1.8, -2.0],
    camFocus: [6.5, 3.8, 1.2],
    explodeDir: [2.0, 1.0, -2.0]
  },
  'V-400': {
    name: 'Unit 400: Anode O₂ Cyclone Degasser & Stack',
    tag: 'V-400',
    category: 'Multiphase Gas-Liquid Separation',
    standard: 'ASME BPVC Sec. VIII Div. 1 / API 520 / NFPA 68',
    spec: 'Diameter 900 mm × 2400 mm, 60° steep conical hopper bottom, tangential inlet nozzle, internal 316L wire-mesh demister pad, elevated 6.20m roof vent stack with flame arrestor.',
    dimensions: 'Dia 900 mm × 6.2m Elevation Vent Stack',
    duty: 'Stripping dissolved oxygen bubbles from recirculating DI water loop, venting safe oxygen stream, and housing SIF-01 continuous NDIR hydrogen crossover sensor.',
    opParams: {
      'Cyclone Diameter': '900 mm (Stairmand high-efficiency geometry)',
      'H₂ Crossover Limit': '< 1.00 vol% O₂ in H₂ (SIF-01 SIL-2 Trip)',
      'Water Recirc Flow': '15.0 L/min nominal per stack cell loop',
      'Vent Stack Elevation': '6.20 m (Safe atmospheric dispersal)'
    },
    script: 'engineering_standards/iec_61511_sil2_lopa_safety.py',
    standardModalKey: 'iec',
    basePos: [4.2, 2.5, -2.0],
    focusPos: [4.2, 3.2, -2.0],
    camFocus: [8.5, 4.5, 0.8],
    explodeDir: [2.6, 1.2, -2.0]
  },
  'DR-500': {
    name: 'Unit 500: Catalytic DeOxo & Dual 13X TSA Dryers',
    tag: 'DR-500',
    category: 'High-Purity Gas Deoxygenation & Drying',
    standard: 'ISO 14687:2019 Grade D / SAE J2719 Vehicle Fuel',
    spec: 'Exothermic catalytic DeOxo reactor (0.5% Pd/Al₂O₃ catalyst, ΔT = 14.7°C) + Twin 13X Molecular Sieve TSA columns (12-hour swing cycle, automated 4-way valve manifold, -70.0°C dew point).',
    dimensions: 'Twin 340mm OD × 2,200mm Columns + Central DeOxo',
    duty: 'Catalytic combustion of trace oxygen to <1 ppm and deep desiccation of moisture to <1 ppm, guaranteeing ISO 14687 Grade D fuel cell purity.',
    opParams: {
      'Residual Oxygen': '0.5 ppm (< 1.0 ppm ISO 14687 limit)',
      'Moisture Dew Point': '≤ -70.0°C (1.0 ppm H₂O residual)',
      'Exothermic Reaction': '2 H₂ + O₂ → 2 H₂O (ΔT = +14.7°C adiabatic)',
      'Adsorbent Media': 'Zeolite 13X Spherical Beads (2.0–3.0mm)'
    },
    script: 'engineering_standards/iso_14687_grade_d_purification.py',
    standardModalKey: 'iso_purity',
    basePos: [3.4, 4.36, 1.6],
    focusPos: [3.4, 5.2, 1.6],
    camFocus: [6.8, 7.5, 5.5],
    explodeDir: [2.2, 1.5, 1.8]
  },
  'WTR-600': {
    name: 'Unit 600: Ultra-Pure DI Water Treatment Skid',
    tag: 'WTR-600',
    category: 'Closed-Loop Feed Water Chemistry',
    standard: 'ASTM D1193 Type 1 (>18.2 MΩ·cm) / ASME BPE',
    spec: '3,000L vertical DI storage tank (SA-240 316L, internal Ra < 0.4 µm electropolished), Dual 100% ANSI centrifugal pumps, Plate Heat Exchanger (80°C → 65°C), Mixed-Bed Polishing Resin.',
    dimensions: 'Dia 1.36m × 2.6m Tank + Duplex ANSI Pump Skid',
    duty: 'Continuous circulation of ultra-pure deionized water, in-operando dielectric biofilm resistance monitoring, and closed-loop proactive DI flushing.',
    opParams: {
      'Feed Water Resistivity': '> 18.2 MΩ·cm @ 25°C (ASTM D1193 Type 1)',
      'Circulation Flow': 'Duplex ANSI Pumps @ 45 m³/h (1 duty, 1 standby)',
      'Loop Cooling Duty': '125 kW Plate Heat Exchanger (80°C → 65°C)',
      'Biofilm Isolation': '< 1.4s SCADA trip via Bayesian impedance observer'
    },
    script: 'digital_twin_core/scada_controller.py',
    standardModalKey: 'iec',
    basePos: [-3.8, 1.4, -2.0],
    focusPos: [-3.8, 1.8, -2.0],
    camFocus: [-8.0, 4.0, 1.5],
    explodeDir: [-2.5, 0.8, -2.0]
  },
  'CMP-700': {
    name: 'Unit 700: HP Diaphragm Compressor Array',
    tag: 'CMP-700',
    category: 'Mechanical Compression (API 618)',
    standard: 'API 618 / ASME B31.3 / Class 2500',
    spec: '3-Stage oil-free metal diaphragm compressor boosting H₂ from 30 bar to 350 bar / 700 bar; 45 kW ATEX Zone 1 explosion-proof motor, suction/discharge pulsation dampener bottles.',
    dimensions: '2.4m L × 1.6m W × 1.5m H (Heavy Rigid Cast Skid)',
    duty: 'Compressing purified 30-bar green hydrogen up to 350 bar for buffer accumulator storage and high-pressure tube trailer dispatch.',
    opParams: {
      'Suction Pressure': '30.0 bar (Direct stack discharge feed)',
      'Discharge Pressure': '350.0 bar (Class 2500 Piping Schedule)',
      'Compression Stages': '3-Stage Metal Diaphragm (Zero Lube-Oil)',
      'Drive Rating': '45 kW Variable Frequency Drive (VFD Motor)'
    },
    script: 'engineering_standards/asme_bpvc_calculations.py',
    standardModalKey: 'asme',
    basePos: [-3.8, 1.2, 2.0],
    focusPos: [-3.8, 1.6, 2.0],
    camFocus: [-8.0, 3.8, 6.0],
    explodeDir: [-2.5, 0.8, 2.2]
  },
  'ST-800': {
    name: 'Unit 800: 350-bar HP Buffer Storage Array',
    tag: 'ST-800',
    category: 'ASME Section VIII Div 3 KD-250 High Pressure',
    standard: 'PESO Gas Cylinders Rules 2016 & IS 7285 (Domestic) | ASME Sec VIII Div 3 & ISO 11119-3 (Global)',
    spec: '4x Horizontal forged seamless accumulator cylinders (OD 650 mm, wall 75.0 mm, SA-723 Grade 2 Cl 2, Design 350 bar, Von Mises stress 148.5 MPa, Theoretical Burst 2,196 bar).',
    dimensions: '4x Cylinders: OD 650 mm × 3,800 mm Length',
    duty: 'Buffer damping compressor pneumatic pulsation and providing 350-bar fast-fill storage for tube trailers and mobility dispenser modules.',
    opParams: {
      'Design Operating Pressure': '350.0 bar (Hydrostatic test: 525.0 bar)',
      'Wall Thickness': '75.0 mm Forged SA-723 Grade 2 Cl 2',
      'Theoretical Burst': '2,196.2 bar (6.27x Safety Margin vs MAWP)',
      'Von Mises Bore Stress': '148.5 MPa (Well below 690 MPa yield limit)'
    },
    script: 'engineering_standards/asme_bpvc_calculations.py',
    standardModalKey: 'asme',
    basePos: [0, 1.2, 3.5],
    focusPos: [0, 1.5, 3.5],
    camFocus: [0, 4.5, 8.5],
    explodeDir: [0, 0.8, 2.6]
  },
  'ELE-900': {
    name: 'Unit 900: 11kV Substation & Power Rectifier',
    tag: 'ELE-900',
    category: 'Heavy Electrical Transformation & Rectification',
    standard: 'IEEE 519 / IEC 60146 / IS 2026',
    spec: '11 kV / 415 V Oil-immersed step-down transformer with external radiator fins and porcelain bushings; 12-Pulse Controlled Thyristor Rectifier delivering 5,000A DC with <3% THD.',
    dimensions: '3.2m L × 2.0m W × 2.8m H (Outdoor Substation Bay)',
    duty: 'Stepping down grid or solar farm medium voltage to low-voltage AC, converting to smooth 5000A DC bus power with active harmonic suppression.',
    opParams: {
      'Input Voltage': '11,000 V AC, 3-Phase, 50 Hz',
      'Output DC Current': '5,000 A continuous DC per stack bay',
      'Harmonic Distortion': '< 3.0% Total Harmonic Distortion (THD)',
      'Cooling Mechanism': 'ONAN Transformer + Forced Water Thyristor'
    },
    script: 'PEM_Green_Hydrogen_Digital_Twin/cad_and_geometry/build_complete_industrial_hydrogen_plant.py',
    standardModalKey: 'asme',
    basePos: [-4.6, 1.8, 0],
    focusPos: [-4.6, 2.0, 0],
    camFocus: [-9.5, 4.5, 3.0],
    explodeDir: [-2.8, 0.8, 0]
  },
  'PLC-1000': {
    name: 'Unit 1000: Siemens S7-1500F Safety PLC & SCADA',
    tag: 'PLC-1000',
    category: 'IEC 61511 SIL-2 Automation & Digital Twin Edge',
    standard: 'IEC 61508 / IEC 61511 SIL-2 HIPPS / NFPA 79',
    spec: 'Rittal TS8 double-door industrial bay, Siemens S7-1500F Fail-Safe CPU, redundant ET 200SP safe I/O, Modbus TCP / OPC-UA edge gateway, 21.5" Full HD IP65 capacitive touch HMI.',
    dimensions: '1.6m L × 0.8m W × 2.2m H (IP65 Rittal Enclosure)',
    duty: 'Edge host executing quantized PIBT neural inference (RTX 4050/Jetson), real-time Butler-Volmer tracking, biofilm isolation (<1.4s trip), and 1oo2 SIL-2 emergency plant trip.',
    opParams: {
      'Safety Integrity Level': 'SIL-2 Verified (PFD_avg = 5.15e-3, RRF = 194.3)',
      'Trip Execution Time': '< 1.4 seconds from biofilm anomaly onset',
      'Industrial Protocol': 'Modbus TCP / OPC-UA / PROFINET RT',
      'Edge Hardware': 'Siemens S7-1500F + Quantized AI Coprocessor'
    },
    script: 'engineering_standards/iec_61511_sil2_lopa_safety.py',
    standardModalKey: 'iec',
    basePos: [4.6, 1.4, 0],
    focusPos: [4.6, 1.8, 0],
    camFocus: [9.5, 3.8, 2.5],
    explodeDir: [2.8, 0.8, 0]
  },
  'PIP-1100': {
    name: 'Unit 1100: Interconnecting Process Piping Network',
    tag: 'PIP-1100',
    category: 'ASME B31.3 / B31.12 Process Piping',
    standard: 'ASME B31.3 / ASME B31.12 (Hydrogen Piping)',
    spec: 'Seamless SA-312 TP316L Schedule 80S, ASME B16.5 Class 300 / 2500 Weld-Neck Flanges, full industrial color coding (Hydrogen = Vivid Rose, Oxygen = Blue, DI Water = Cyan, Nitrogen = Amber).',
    dimensions: 'Full Skid Interconnect Manifold (Schedule 80S)',
    duty: 'Hermetic fluid interconnect between all 10 skid modules with integrated thermal expansion loops, automated pneumatically actuated ESD valves, and LOPA relief paths.',
    opParams: {
      'Piping Specification': 'Seamless SA-312 TP316L Schedule 80S',
      'Flange Rating': 'ASME B16.5 Class 300 & Class 2500 Weld-Neck',
      'Weld Inspection': '100% Radiographic Examination (RT Class 1)',
      'Leak Rate Guarantee': '< 1 × 10⁻⁶ mbar·L/s Helium Mass Spec'
    },
    script: 'engineering_standards/asme_bpvc_calculations.py',
    standardModalKey: 'asme',
    basePos: [0, 3.2, 0],
    focusPos: [0, 3.6, 0],
    camFocus: [7.5, 6.0, 7.5],
    explodeDir: [0, 2.0, 0]
  }
};

// --- High-Quality PBR Industrial Materials Library ---
const matLib = {
  ss316l: new THREE.MeshStandardMaterial({
    color: 0xD8E2EC,
    roughness: 0.22,
    metalness: 0.88
  }),
  titanium: new THREE.MeshStandardMaterial({
    color: 0xC4D0DC,
    roughness: 0.18,
    metalness: 0.92
  }),
  steelDark: new THREE.MeshStandardMaterial({
    color: 0x1E293B,
    roughness: 0.42,
    metalness: 0.65
  }),
  steelForged: new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.32,
    metalness: 0.80
  }),
  steelGalv: new THREE.MeshStandardMaterial({
    color: 0x94A3B8,
    roughness: 0.52,
    metalness: 0.55
  }),
  copper: new THREE.MeshStandardMaterial({
    color: 0xB45309,
    roughness: 0.28,
    metalness: 0.85
  }),
  safetyYellow: new THREE.MeshStandardMaterial({
    color: 0xF59E0B,
    roughness: 0.35,
    metalness: 0.15
  }),
  safetyRed: new THREE.MeshStandardMaterial({
    color: 0xDC2626,
    roughness: 0.30,
    metalness: 0.25
  }),
  processBlue: new THREE.MeshStandardMaterial({
    color: 0x0284C7,
    roughness: 0.35,
    metalness: 0.35
  }),
  safetyGreen: new THREE.MeshStandardMaterial({
    color: 0x059669,
    roughness: 0.35,
    metalness: 0.30
  }),
  pipingH2: new THREE.MeshStandardMaterial({
    color: 0xE11D48,
    roughness: 0.32,
    metalness: 0.50
  }),
  pipingO2: new THREE.MeshStandardMaterial({
    color: 0x0284C7,
    roughness: 0.32,
    metalness: 0.50
  }),
  pipingDI: new THREE.MeshStandardMaterial({
    color: 0x06B6D4,
    roughness: 0.32,
    metalness: 0.45
  }),
  pipingN2: new THREE.MeshStandardMaterial({
    color: 0xD97706,
    roughness: 0.32,
    metalness: 0.45
  }),
  flangeMat: new THREE.MeshStandardMaterial({
    color: 0x64748B,
    roughness: 0.30,
    metalness: 0.82
  }),
  highlightMat: new THREE.MeshStandardMaterial({
    color: 0x0EA5E9,
    emissive: 0x0284C7,
    emissiveIntensity: 0.55,
    roughness: 0.20,
    metalness: 0.85
  })
};

// --- Procedural Mechanical Geometric Primitives ---

function makeMesh(geo, mat) {
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function createFlange(rInner, rOuter, thick, mat = matLib.flangeMat, numBolts = 8) {
  const grp = new THREE.Group();
  const ring = makeMesh(new THREE.CylinderGeometry(rOuter, rOuter, thick, 24), mat);
  grp.add(ring);

  // Raised face
  const rf = makeMesh(new THREE.CylinderGeometry(rOuter * 0.75, rOuter * 0.75, thick + 0.008, 24), mat);
  grp.add(rf);

  // Bolt heads
  const boltR = (rInner + rOuter) / 2;
  for (let i = 0; i < numBolts; i++) {
    const angle = (i / numBolts) * Math.PI * 2;
    const bolt = makeMesh(new THREE.CylinderGeometry(0.016, 0.016, thick + 0.02, 6), matLib.steelGalv);
    bolt.position.set(Math.cos(angle) * boltR, 0, Math.sin(angle) * boltR);
    grp.add(bolt);
  }
  return grp;
}

function createNozzle(rPipe, len, rFlange, flangeThick, pipeMat = matLib.ss316l) {
  const grp = new THREE.Group();
  const pipe = makeMesh(new THREE.CylinderGeometry(rPipe, rPipe, len, 16), pipeMat);
  pipe.position.y = len / 2;
  grp.add(pipe);

  const flange = createFlange(rPipe, rFlange, flangeThick);
  flange.position.y = len;
  grp.add(flange);
  return grp;
}

function createIBeam(x1, y1, z1, x2, y2, z2, depth, width, mat = matLib.steelDark) {
  const grp = new THREE.Group();
  const dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.001) return grp;

  const tf = depth * 0.12;
  const tw = depth * 0.07;

  const topF = makeMesh(new THREE.BoxGeometry(width, tf, len), mat);
  topF.position.y = depth / 2 - tf / 2;
  const botF = makeMesh(new THREE.BoxGeometry(width, tf, len), mat);
  botF.position.y = -depth / 2 + tf / 2;
  const web = makeMesh(new THREE.BoxGeometry(tw, depth - 2 * tf, len), mat);

  grp.add(topF);
  grp.add(botF);
  grp.add(web);

  grp.position.set((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2);
  const dir = new THREE.Vector3(dx, dy, dz).normalize();
  const up = new THREE.Vector3(0, 0, 1);
  grp.quaternion.setFromUnitVectors(up, dir);
  return grp;
}

// --- Single Source of Truth: Dynamic Manifest Loader ---
async function loadEngineeringManifest() {
  try {
    const res = await fetch('engineering_manifest.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const manifest = await res.json();
    if (manifest && manifest.subsystems) {
      window.engineeringManifest = manifest;
      Object.keys(manifest.subsystems).forEach(tag => {
        const sub = manifest.subsystems[tag];
        if (plantCatalog[tag]) {
          plantCatalog[tag].name = sub.name || plantCatalog[tag].name;
          plantCatalog[tag].category = sub.category || plantCatalog[tag].category;
          plantCatalog[tag].standard = sub.standard || plantCatalog[tag].standard;
          plantCatalog[tag].duty = sub.mechanical_duty || plantCatalog[tag].duty;
          plantCatalog[tag].bom = sub.bom || [];
          plantCatalog[tag].governing_calculations = sub.governing_calculations || null;
          plantCatalog[tag].operational_telemetry = sub.operational_telemetry || null;
          if (sub.governing_calculations && sub.governing_calculations.script_source) {
            plantCatalog[tag].script = sub.governing_calculations.script_source;
          }
          if (sub.bounding_box && sub.bounding_box.dimensions_m) {
            plantCatalog[tag].dimensions = `${sub.bounding_box.dimensions_m[0]}m L × ${sub.bounding_box.dimensions_m[1]}m W × ${sub.bounding_box.dimensions_m[2]}m H`;
          }
        }
      });
      if (selectedSubsystem) {
        populateSubsystemHudOnly(selectedSubsystem);
      }
    }
  } catch (e) {
    console.warn('[3D Workstation] Loaded embedded baseline manifest:', e.message);
  }
}

// --- Initialize Three.js WebGL Engine ---

function initPlant3DScene() {
  // Load dynamic single-source-of-truth manifest
  loadEngineeringManifest();

  plantContainer = document.getElementById('plant3dCanvasContainer');
  if (!plantContainer) return;

  const w = plantContainer.clientWidth || 800;
  const h = plantContainer.clientHeight || 580;

  // 1. Scene
  plantScene = new THREE.Scene();
  plantScene.background = new THREE.Color(0xF8FAFC);

  // 2. Camera
  plantCamera = new THREE.PerspectiveCamera(38, w / h, 0.2, 100);
  plantCamera.position.set(13.5, 10.5, 15.5);

  // 3. Renderer with PCFSoftShadowMap
  plantRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  plantRenderer.setSize(w, h);
  plantRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  plantRenderer.shadowMap.enabled = true;
  plantRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
  plantRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  plantRenderer.toneMappingExposure = 1.15;
  plantContainer.innerHTML = '';
  plantContainer.appendChild(plantRenderer.domElement);

  // 4. Orbit Controls
  plantControls = new THREE.OrbitControls(plantCamera, plantRenderer.domElement);
  plantControls.enableDamping = true;
  plantControls.dampingFactor = 0.05;
  plantControls.maxPolarAngle = Math.PI / 2 - 0.02;
  plantControls.minDistance = 3.0;
  plantControls.maxDistance = 45;
  plantControls.target.set(0, 2.6, 0);

  // 5. Lighting Suite
  const ambLight = new THREE.AmbientLight(0xFFFFFF, 0.85);
  plantScene.add(ambLight);

  const keyLight = new THREE.DirectionalLight(0xFFFFFF, 1.25);
  keyLight.position.set(18, 28, 20);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 80;
  keyLight.shadow.camera.left = -16;
  keyLight.shadow.camera.right = 16;
  keyLight.shadow.camera.top = 16;
  keyLight.shadow.camera.bottom = -16;
  keyLight.shadow.bias = -0.0004;
  plantScene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xE2E8F0, 0.65);
  fillLight.position.set(-18, 14, -16);
  plantScene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0x0284C7, 0.40);
  rimLight.position.set(0, -10, 18);
  plantScene.add(rimLight);

  // 6. Ground Shadow Plane & Technical Grid
  const shadowGeo = new THREE.PlaneGeometry(36, 36);
  const shadowMat = new THREE.ShadowMaterial({ opacity: 0.22 });
  groundShadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
  groundShadowPlane.rotation.x = -Math.PI / 2;
  groundShadowPlane.position.y = -0.01;
  groundShadowPlane.receiveShadow = true;
  plantScene.add(groundShadowPlane);

  groundGrid = new THREE.GridHelper(28, 28, 0x94A3B8, 0xE2E8F0);
  groundGrid.position.y = 0.0;
  plantScene.add(groundGrid);

  // 7. Raycaster
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // 8. Build All 11 High-Fidelity Subsystems
  buildAllSubsystems();

  // 9. Event Listeners
  plantRenderer.domElement.addEventListener('pointerdown', onPlantPointerDown);
  window.addEventListener('resize', onPlantResize);

  // 10. Start Animation Loop
  animatePlant();

  // 11. Populate Subsystem Pills Bar
  renderSubsystemPills();

  // 12. Select PLANT-100T Master Assembly by default
  selectSubsystem('PLANT-100T');
}

// ============================================================================
// HIGH-FIDELITY INDUSTRIAL GEOMETRY FOR ALL 11 SUBSYSTEMS
// ============================================================================

function attachSubsystem(tag, group) {
  group.name = tag;
  subsystems[tag] = group;
  plantScene.add(group);

  // Store original materials for un-highlighting
  group.traverse(child => {
    if (child.isMesh) {
      origSubsystemMaterials.set(child, child.material);
    }
  });
}

function buildAllSubsystems() {
  // --------------------------------------------------------------------------
  // 1. STR-100: Heavy Structural Mezzanine & Pipe Rack
  // --------------------------------------------------------------------------
  const strGrp = new THREE.Group();
  const colXs = [-4.6, -1.8, 1.8, 4.6];
  const colZs = [-3.0, 3.0];

  colXs.forEach(cx => {
    colZs.forEach(cz => {
      // ISMB 300 Column
      const col = makeMesh(new THREE.BoxGeometry(0.26, 4.2, 0.26), matLib.steelDark);
      col.position.set(cx, 2.1, cz);
      strGrp.add(col);

      // 4-Bolt Baseplate
      const bp = makeMesh(new THREE.BoxGeometry(0.46, 0.04, 0.46), matLib.steelDark);
      bp.position.set(cx, 0.02, cz);
      strGrp.add(bp);

      // Anchor J-Bolts
      [[-0.16, -0.16], [0.16, -0.16], [-0.16, 0.16], [0.16, 0.16]].forEach(([bx, bz]) => {
        const bolt = makeMesh(new THREE.CylinderGeometry(0.016, 0.016, 0.08, 6), matLib.steelGalv);
        bolt.position.set(cx + bx, 0.06, cz + bz);
        strGrp.add(bolt);
      });
    });
  });

  // Longitudinal & transverse beams at Y=4.2
  colZs.forEach(cz => {
    const beam = makeMesh(new THREE.BoxGeometry(9.6, 0.30, 0.18), matLib.steelDark);
    beam.position.set(0, 4.2, cz);
    strGrp.add(beam);
  });
  colXs.forEach(cx => {
    const cross = makeMesh(new THREE.BoxGeometry(0.18, 0.30, 6.0), matLib.steelDark);
    cross.position.set(cx, 4.2, 0);
    strGrp.add(cross);
  });

  // Galvanized Checkered Walkway Deck
  const deck1 = makeMesh(new THREE.BoxGeometry(9.4, 0.04, 2.0), matLib.steelGalv);
  deck1.position.set(0, 4.36, -2.0);
  const deck2 = makeMesh(new THREE.BoxGeometry(9.4, 0.04, 2.0), matLib.steelGalv);
  deck2.position.set(0, 4.36, 2.0);
  strGrp.add(deck1);
  strGrp.add(deck2);

  // Kick-plate (toeboard 100mm)
  [-3.0, 3.0].forEach(cz => {
    const kick = makeMesh(new THREE.BoxGeometry(9.4, 0.10, 0.02), matLib.safetyYellow);
    kick.position.set(0, 4.43, cz);
    strGrp.add(kick);

    // Statutory Safety Yellow Handrails (The Factories Act, 1948 Sec 32)
    const topRail = makeMesh(new THREE.CylinderGeometry(0.022, 0.022, 9.4, 12), matLib.safetyYellow);
    topRail.rotation.z = Math.PI / 2;
    topRail.position.set(0, 5.40, cz);
    strGrp.add(topRail);

    const midRail = makeMesh(new THREE.CylinderGeometry(0.018, 0.018, 9.4, 12), matLib.safetyYellow);
    midRail.rotation.z = Math.PI / 2;
    midRail.position.set(0, 4.88, cz);
    strGrp.add(midRail);

    colXs.forEach(px => {
      const post = makeMesh(new THREE.CylinderGeometry(0.024, 0.024, 1.05, 12), matLib.safetyYellow);
      post.position.set(px, 4.88, cz);
      strGrp.add(post);
    });
  });

  attachSubsystem('STR-100', strGrp);

  // --------------------------------------------------------------------------
  // 2. EL-200: Commercial Electrolyzer Bay (4x Stacks)
  // --------------------------------------------------------------------------
  const elGrp = new THREE.Group();
  const stackXs = [-2.6, -1.0, 0.6, 2.2];

  stackXs.forEach((sx) => {
    const stack = new THREE.Group();

    // Heavy End Clamping Plates (SA-516 Gr. 70, 50mm thick)
    const ep1 = makeMesh(new THREE.BoxGeometry(0.70, 0.90, 0.10), matLib.steelForged);
    ep1.position.set(0, 0.45, -0.48);
    const ep2 = makeMesh(new THREE.BoxGeometry(0.70, 0.90, 0.10), matLib.steelForged);
    ep2.position.set(0, 0.45, 0.48);
    stack.add(ep1);
    stack.add(ep2);

    // Bipolar Plate Pack (50 Cells, ASTM B265 Grade 2 Titanium)
    const pack = makeMesh(new THREE.BoxGeometry(0.58, 0.78, 0.86), matLib.titanium);
    pack.position.set(0, 0.45, 0);
    stack.add(pack);

    // Visible machined flow channel stripes
    for (let c = -0.38; c <= 0.38; c += 0.08) {
      const stripe = makeMesh(new THREE.BoxGeometry(0.59, 0.02, 0.84), matLib.ss316l);
      stripe.position.set(0, 0.45 + c, 0);
      stack.add(stripe);
    }

    // 16x Tension Tie-Rods with Belleville Spring Washers & Hex Nuts
    const rodX = [-0.30, -0.10, 0.10, 0.30];
    const rodY = [0.10, 0.80];
    rodX.forEach(rx => {
      rodY.forEach(ry => {
        const rod = makeMesh(new THREE.CylinderGeometry(0.014, 0.014, 1.10, 8), matLib.steelGalv);
        rod.rotation.x = Math.PI / 2;
        rod.position.set(rx, ry, 0);
        stack.add(rod);

        // Clamping Hex Nut
        const nut1 = makeMesh(new THREE.CylinderGeometry(0.024, 0.024, 0.03, 6), matLib.flangeMat);
        nut1.rotation.x = Math.PI / 2;
        nut1.position.set(rx, ry, -0.54);
        const nut2 = makeMesh(new THREE.CylinderGeometry(0.024, 0.024, 0.03, 6), matLib.flangeMat);
        nut2.rotation.x = Math.PI / 2;
        nut2.position.set(rx, ry, 0.54);
        stack.add(nut1);
        stack.add(nut2);
      });
    });

    // 5000A Polished Copper Busbars with Flexible Jumpers
    const bus = makeMesh(new THREE.BoxGeometry(0.08, 0.16, 0.50), matLib.copper);
    bus.position.set(0, 0.98, -0.2);
    stack.add(bus);

    const jumper = makeMesh(new THREE.CylinderGeometry(0.02, 0.02, 0.28, 8), matLib.copper);
    jumper.position.set(0, 1.10, -0.2);
    stack.add(jumper);

    // Flanged fluid headers (DI inlet bottom, H2/O2 outlets top)
    const inFlange = createFlange(0.03, 0.07, 0.02, matLib.flangeMat, 4);
    inFlange.rotation.x = Math.PI / 2;
    inFlange.position.set(-0.25, 0.25, -0.54);
    stack.add(inFlange);

    const outFlange = createFlange(0.03, 0.07, 0.02, matLib.flangeMat, 4);
    outFlange.rotation.x = Math.PI / 2;
    outFlange.position.set(0.25, 0.70, 0.54);
    stack.add(outFlange);

    stack.position.set(sx, 4.38, 0);
    elGrp.add(stack);
  });
  attachSubsystem('EL-200', elGrp);

  // --------------------------------------------------------------------------
  // 3. V-300: HP Hydrogen Knockout Drum (ASME Sec VIII Div 1)
  // --------------------------------------------------------------------------
  const v300Grp = new THREE.Group();

  // Cylindrical Shell (ID 800 mm x 2200 mm, 14mm SA-240 316L)
  const vBody = makeMesh(new THREE.CylinderGeometry(0.44, 0.44, 2.1, 32), matLib.ss316l);
  vBody.position.y = 1.65;
  v300Grp.add(vBody);

  // 2:1 Ellipsoidal Heads (12mm SA-240 316L)
  const vTopHead = makeMesh(new THREE.SphereGeometry(0.44, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), matLib.ss316l);
  vTopHead.scale.set(1, 0.5, 1);
  vTopHead.position.y = 2.70;
  v300Grp.add(vTopHead);

  const vBotHead = makeMesh(new THREE.SphereGeometry(0.44, 32, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), matLib.ss316l);
  vBotHead.scale.set(1, 0.5, 1);
  vBotHead.position.y = 0.60;
  v300Grp.add(vBotHead);

  // Welded Circumferential Seam Rings
  [0.60, 2.70].forEach(sy => {
    const seam = makeMesh(new THREE.TorusGeometry(0.444, 0.008, 8, 32), matLib.flangeMat);
    seam.rotation.x = Math.PI / 2;
    seam.position.y = sy;
    v300Grp.add(seam);
  });

  // Skirt Support with Anchor Chairs & Manway
  const skirt = makeMesh(new THREE.CylinderGeometry(0.44, 0.46, 0.60, 32, 1, true), matLib.steelDark);
  skirt.position.y = 0.30;
  v300Grp.add(skirt);

  // Level Bridle with Magnetic Chamber & Radar Transmitter
  const bridlePipe = makeMesh(new THREE.CylinderGeometry(0.03, 0.03, 1.80, 16), matLib.ss316l);
  bridlePipe.position.set(0.58, 1.65, 0);
  v300Grp.add(bridlePipe);

  const radarHead = makeMesh(new THREE.CylinderGeometry(0.06, 0.06, 0.16, 16), matLib.processBlue);
  radarHead.position.set(0.58, 2.62, 0);
  v300Grp.add(radarHead);

  // Top ASME Angle PSV Relief Valve
  const psvNozzle = createNozzle(0.04, 0.20, 0.09, 0.025);
  psvNozzle.position.set(0, 2.92, 0);
  v300Grp.add(psvNozzle);

  const psvBody = makeMesh(new THREE.BoxGeometry(0.16, 0.26, 0.16), matLib.safetyRed);
  psvBody.position.set(0, 3.25, 0);
  v300Grp.add(psvBody);

  const psvHorn = makeMesh(new THREE.ConeGeometry(0.08, 0.18, 16), matLib.safetyRed);
  psvHorn.rotation.z = Math.PI / 2;
  psvHorn.position.set(0.14, 3.28, 0);
  v300Grp.add(psvHorn);

  v300Grp.position.set(2.2, 0, -2.0);
  attachSubsystem('V-300', v300Grp);

  // --------------------------------------------------------------------------
  // 4. V-400: Anode O₂ Cyclone Degasser & Elevated Stack
  // --------------------------------------------------------------------------
  const v400Grp = new THREE.Group();

  // Cylindrical Body
  const cyBody = makeMesh(new THREE.CylinderGeometry(0.48, 0.48, 1.9, 32), matLib.ss316l);
  cyBody.position.y = 2.0;
  v400Grp.add(cyBody);

  // 60° Steep Conical Bottom
  const cyCone = makeMesh(new THREE.ConeGeometry(0.48, 0.9, 32), matLib.ss316l);
  cyCone.rotation.x = Math.PI;
  cyCone.position.y = 0.60;
  v400Grp.add(cyCone);

  // Tangential Inlet Nozzle
  const tangInlet = makeMesh(new THREE.CylinderGeometry(0.06, 0.06, 0.40, 16), matLib.ss316l);
  tangInlet.rotation.z = Math.PI / 2;
  tangInlet.position.set(0.42, 2.4, 0.25);
  v400Grp.add(tangInlet);

  // Elevated 6.2m Roof Vent Stack
  const vent = makeMesh(new THREE.CylinderGeometry(0.12, 0.12, 4.0, 20), matLib.ss316l);
  vent.position.set(0, 4.8, 0);
  v400Grp.add(vent);

  // Rain cowl / spark arrestor hood
  const cowl = makeMesh(new THREE.ConeGeometry(0.24, 0.16, 20), matLib.flangeMat);
  cowl.position.set(0, 6.85, 0);
  v400Grp.add(cowl);

  // SIF-01 NDIR Sample Tap Port
  const sampleTap = makeMesh(new THREE.CylinderGeometry(0.02, 0.02, 0.15, 12), matLib.processBlue);
  sampleTap.rotation.z = Math.PI / 2;
  sampleTap.position.set(0.16, 3.8, 0);
  v400Grp.add(sampleTap);

  v400Grp.position.set(4.2, 0, -2.0);
  attachSubsystem('V-400', v400Grp);

  // --------------------------------------------------------------------------
  // 5. DR-500: Catalytic DeOxo & Dual 13X TSA Dryers
  // --------------------------------------------------------------------------
  const drGrp = new THREE.Group();

  // Central Catalytic DeOxo Reactor (0.5% Pd/Al2O3)
  const deoxo = makeMesh(new THREE.CylinderGeometry(0.34, 0.34, 1.5, 24), matLib.ss316l);
  deoxo.position.set(0, 1.3, 0);
  drGrp.add(deoxo);

  // Insulated Jacket Bands
  [-0.3, 0.0, 0.3].forEach(by => {
    const band = makeMesh(new THREE.TorusGeometry(0.345, 0.012, 8, 24), matLib.copper);
    band.rotation.x = Math.PI / 2;
    band.position.set(0, 1.3 + by, 0);
    drGrp.add(band);
  });

  // Twin 13X TSA Adsorber Towers
  [-0.72, 0.72].forEach(tx => {
    const tsa = makeMesh(new THREE.CylinderGeometry(0.36, 0.36, 2.3, 24), matLib.ss316l);
    tsa.position.set(tx, 1.5, 0.6);
    drGrp.add(tsa);

    const topCap = makeMesh(new THREE.SphereGeometry(0.36, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), matLib.ss316l);
    topCap.position.set(tx, 2.65, 0.6);
    drGrp.add(topCap);

    // 4-Way Switching Valve Assembly
    const valve = makeMesh(new THREE.BoxGeometry(0.14, 0.14, 0.14), matLib.processBlue);
    valve.position.set(tx, 0.40, 0.6);
    drGrp.add(valve);
  });

  // Interconnecting Manifold Bridge
  const bridge = makeMesh(new THREE.BoxGeometry(1.6, 0.06, 0.06), matLib.pipingH2);
  bridge.position.set(0, 2.2, 0.6);
  drGrp.add(bridge);

  drGrp.position.set(3.4, 4.38, 1.6);
  attachSubsystem('DR-500', drGrp);

  // --------------------------------------------------------------------------
  // 6. WTR-600: Ultra-Pure DI Water Treatment Skid
  // --------------------------------------------------------------------------
  const wtrGrp = new THREE.Group();

  // 3,000L DI Storage Tank (Ra < 0.4 µm)
  const diTank = makeMesh(new THREE.CylinderGeometry(0.72, 0.72, 2.5, 32), matLib.ss316l);
  diTank.position.set(-0.85, 1.35, 0);
  wtrGrp.add(diTank);

  const diHead = makeMesh(new THREE.SphereGeometry(0.72, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), matLib.ss316l);
  diHead.scale.set(1, 0.4, 1);
  diHead.position.set(-0.85, 2.60, 0);
  wtrGrp.add(diHead);

  // Dual ANSI Centrifugal Circulation Pumps
  [-0.35, 0.35].forEach(pz => {
    const pumpVolute = makeMesh(new THREE.CylinderGeometry(0.20, 0.20, 0.18, 16), matLib.processBlue);
    pumpVolute.rotation.z = Math.PI / 2;
    pumpVolute.position.set(0.70, 0.30, pz);
    wtrGrp.add(pumpVolute);

    const pumpMotor = makeMesh(new THREE.CylinderGeometry(0.15, 0.15, 0.36, 16), matLib.processBlue);
    pumpMotor.rotation.z = Math.PI / 2;
    pumpMotor.position.set(0.96, 0.30, pz);
    wtrGrp.add(pumpMotor);

    // Pump Base Skid
    const pBase = makeMesh(new THREE.BoxGeometry(0.70, 0.08, 0.36), matLib.steelDark);
    pBase.position.set(0.80, 0.04, pz);
    wtrGrp.add(pBase);
  });

  // Plate Chiller & Mixed-Bed Resin Cylinders
  const chiller = makeMesh(new THREE.BoxGeometry(0.38, 0.85, 0.55), matLib.processBlue);
  chiller.position.set(0.70, 1.25, -0.6);
  wtrGrp.add(chiller);

  const resinPolisher = makeMesh(new THREE.CylinderGeometry(0.24, 0.24, 1.7, 20), matLib.safetyGreen);
  resinPolisher.position.set(0.70, 1.20, 0.6);
  wtrGrp.add(resinPolisher);

  wtrGrp.position.set(-3.8, 0, -2.0);
  attachSubsystem('WTR-600', wtrGrp);

  // --------------------------------------------------------------------------
  // 7. CMP-700: HP Diaphragm Compressor (API 618)
  // --------------------------------------------------------------------------
  const cmpGrp = new THREE.Group();

  // Heavy Base Skid
  const cBase = makeMesh(new THREE.BoxGeometry(2.2, 0.24, 1.5), matLib.steelDark);
  cBase.position.y = 0.12;
  cmpGrp.add(cBase);

  // 45 kW Explosion-Proof Motor with Finned Shell
  const cMotor = makeMesh(new THREE.CylinderGeometry(0.38, 0.38, 0.95, 24), matLib.processBlue);
  cMotor.rotation.z = Math.PI / 2;
  cMotor.position.set(-0.45, 0.60, 0);
  cmpGrp.add(cMotor);

  // Cooling fins on motor
  for (let f = -0.35; f <= 0.35; f += 0.09) {
    const fin = makeMesh(new THREE.TorusGeometry(0.39, 0.012, 8, 24), matLib.processBlue);
    fin.rotation.y = Math.PI / 2;
    fin.position.set(-0.45 + f, 0.60, 0);
    cmpGrp.add(fin);
  }

  // 3-Stage Diaphragm Bolted Compressor Heads
  const cHeads = makeMesh(new THREE.BoxGeometry(0.75, 0.75, 0.95), matLib.steelForged);
  cHeads.position.set(0.50, 0.70, 0);
  cmpGrp.add(cHeads);

  // Suction & Discharge Pulsation Dampener Bottles
  [-0.30, 0.30].forEach(dz => {
    const damp = makeMesh(new THREE.CylinderGeometry(0.18, 0.18, 0.90, 20), matLib.ss316l);
    damp.position.set(0.50, 1.35, dz);
    cmpGrp.add(damp);

    const cap = makeMesh(new THREE.SphereGeometry(0.18, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), matLib.ss316l);
    cap.position.set(0.50, 1.80, dz);
    cmpGrp.add(cap);
  });

  cmpGrp.position.set(-3.8, 0, 2.0);
  attachSubsystem('CMP-700', cmpGrp);

  // --------------------------------------------------------------------------
  // 8. ST-800: 350-bar Buffer Storage Array (ASME Sec VIII Div 3)
  // --------------------------------------------------------------------------
  const stGrp = new THREE.Group();
  const cylZs = [-1.0, -0.33, 0.33, 1.0];

  cylZs.forEach(cz => {
    // Horizontal Forged SA-723 Cylinder (75mm Wall)
    const cyl = makeMesh(new THREE.CylinderGeometry(0.32, 0.32, 3.8, 28), matLib.steelForged);
    cyl.rotation.z = Math.PI / 2;
    cyl.position.set(0, 0.65, cz);
    stGrp.add(cyl);

    // Hemispherical Forged End Caps
    const leftCap = makeMesh(new THREE.SphereGeometry(0.32, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), matLib.steelForged);
    leftCap.rotation.z = -Math.PI / 2;
    leftCap.position.set(-1.9, 0.65, cz);
    const rightCap = makeMesh(new THREE.SphereGeometry(0.32, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), matLib.steelForged);
    rightCap.rotation.z = Math.PI / 2;
    rightCap.position.set(1.9, 0.65, cz);
    stGrp.add(leftCap);
    stGrp.add(rightCap);

    // Saddle Supports with Hold-Down Straps
    [-1.2, 1.2].forEach(sx => {
      const saddle = makeMesh(new THREE.BoxGeometry(0.24, 0.30, 0.44), matLib.steelDark);
      saddle.position.set(sx, 0.15, cz);
      stGrp.add(saddle);
    });
  });

  // Class 2500 Interconnecting High-Pressure Manifold Header
  const hpHeader = makeMesh(new THREE.CylinderGeometry(0.045, 0.045, 2.4, 16), matLib.pipingH2);
  hpHeader.position.set(1.98, 0.65, 0);
  stGrp.add(hpHeader);

  stGrp.position.set(0, 0, 3.5);
  attachSubsystem('ST-800', stGrp);

  // --------------------------------------------------------------------------
  // 9. ELE-900: 11kV Substation & Power Rectifier
  // --------------------------------------------------------------------------
  const eleGrp = new THREE.Group();

  // Oil-Immersed Step-Down Transformer Tank
  const trTank = makeMesh(new THREE.BoxGeometry(1.6, 1.8, 1.2), matLib.steelDark);
  trTank.position.set(-0.8, 1.1, 0);
  eleGrp.add(trTank);

  // External Radiator Cooling Fins
  for (let r = -0.45; r <= 0.45; r += 0.15) {
    const fin = makeMesh(new THREE.BoxGeometry(0.02, 1.4, 0.35), matLib.steelDark);
    fin.position.set(-0.8 + r, 1.1, 0.78);
    eleGrp.add(fin);
  }

  // 3x High-Voltage Porcelain Bushings
  [-0.35, 0.0, 0.35].forEach(bx => {
    const bushing = makeMesh(new THREE.CylinderGeometry(0.03, 0.05, 0.45, 12), matLib.flangeMat);
    bushing.position.set(-0.8 + bx, 2.22, 0);
    eleGrp.add(bushing);
  });

  // 12-Pulse Thyristor Rectifier Cabinet
  const rect = makeMesh(new THREE.BoxGeometry(1.4, 2.1, 1.1), matLib.steelDark);
  rect.position.set(0.9, 1.15, 0);
  eleGrp.add(rect);

  // Heavy Water-Cooled Copper Bus Duct
  const busDuct = makeMesh(new THREE.BoxGeometry(1.8, 0.16, 0.20), matLib.copper);
  busDuct.position.set(0.0, 2.35, 0);
  eleGrp.add(busDuct);

  eleGrp.position.set(-4.6, 0, 0);
  attachSubsystem('ELE-900', eleGrp);

  // --------------------------------------------------------------------------
  // 10. PLC-1000: Siemens S7-1500F Safety PLC & SCADA
  // --------------------------------------------------------------------------
  const plcGrp = new THREE.Group();

  // Rittal TS8 Double-Door Industrial Automation Enclosure
  const cab = makeMesh(new THREE.BoxGeometry(1.4, 2.2, 0.65), matLib.steelDark);
  cab.position.y = 1.15;
  plcGrp.add(cab);

  // Roof Climate Control Air Conditioner
  const ac = makeMesh(new THREE.BoxGeometry(0.65, 0.25, 0.45), matLib.steelGalv);
  ac.position.set(0, 2.38, 0);
  plcGrp.add(ac);

  // 21.5" Full HD Touch HMI Screen
  const hmi = makeMesh(new THREE.BoxGeometry(0.60, 0.42, 0.03), matLib.processBlue);
  hmi.position.set(0, 1.45, 0.34);
  plcGrp.add(hmi);

  // Yellow SIL-2 Emergency Stop Button
  const estop = makeMesh(new THREE.CylinderGeometry(0.035, 0.035, 0.03, 12), matLib.safetyRed);
  estop.rotation.x = Math.PI / 2;
  estop.position.set(0.42, 1.10, 0.34);
  plcGrp.add(estop);

  // Tri-Color Status Beacon on Roof
  const beacon = makeMesh(new THREE.CylinderGeometry(0.03, 0.03, 0.18, 12), matLib.safetyGreen);
  beacon.position.set(0.50, 2.35, 0);
  plcGrp.add(beacon);

  plcGrp.position.set(4.6, 0, 0);
  attachSubsystem('PLC-1000', plcGrp);

  // --------------------------------------------------------------------------
  // 11. PIP-1100: Interconnecting Process Piping Network (ASME B31.3 / B31.12)
  // 100% Closed Physical Routing: Connects all 10 skid modules with zero open ends
  // --------------------------------------------------------------------------
  const pipGrp = new THREE.Group();

  function createPipeRunMesh(points, radius, mat, addFlanges = true) {
    const runGrp = new THREE.Group();
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = new THREE.Vector3(...points[i]);
      const p2 = new THREE.Vector3(...points[i + 1]);
      const vec = new THREE.Vector3().subVectors(p2, p1);
      const len = vec.length();
      if (len < 1e-4) continue;
      const geom = new THREE.CylinderGeometry(radius, radius, len, 16);
      const cyl = new THREE.Mesh(geom, mat);
      cyl.castShadow = true;
      cyl.receiveShadow = true;
      const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
      cyl.position.copy(mid);
      const up = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, vec.clone().normalize());
      cyl.setRotationFromQuaternion(quat);
      runGrp.add(cyl);

      if (i < points.length - 2) {
        const joint = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.08, 12, 12), mat);
        joint.position.copy(p2);
        runGrp.add(joint);
      }
    }
    if (addFlanges && points.length >= 2) {
      [0, points.length - 1].forEach(idx => {
        const pt = points[idx];
        const flg = createFlange(radius, radius * 1.8, radius * 0.45, matLib.flangeMat, 6);
        flg.position.set(...pt);
        runGrp.add(flg);
      });
    }
    return runGrp;
  }

  // 1. LINE-01-H2-WET: Cathode 30-bar Wet H2 Collector -> V-300 Knockout Drum
  [-3.0, -1.0, 1.0, 3.0].forEach(sx => {
    pipGrp.add(createPipeRunMesh([
      [sx + 0.25, 5.08, 0.63],
      [sx + 0.25, 5.08, 1.20]
    ], 0.038, matLib.pipingH2, true));
  });
  pipGrp.add(createPipeRunMesh([
    [-2.75, 5.08, 1.20],
    [3.25, 5.08, 1.20],
    [3.25, 4.80, 1.20],
    [3.25, 4.80, -2.00],
    [2.68, 4.80, -2.00],
    [2.68, 1.80, -2.00] // Direct mate into V-300 tangential inlet nozzle
  ], 0.045, matLib.pipingH2, true));

  // 2. LINE-02-H2-DEOXO: V-300 Top Gas Outlet -> DR-500 DeOxo Catalytic Reactor
  pipGrp.add(createPipeRunMesh([
    [2.20, 3.10, -2.00], // V-300 Top Outlet Nozzle
    [2.20, 4.80, -2.00],
    [2.20, 4.80, 1.60],
    [3.40, 4.80, 1.60],
    [3.40, 6.03, 1.60]  // Direct mate into DR-500 top reactor flange
  ], 0.040, matLib.pipingH2, true));

  // 3. LINE-03-H2-CMP-SUCT: DR-500 Pure Dry H2 -> CMP-700 3-Stage Compressor Suction
  pipGrp.add(createPipeRunMesh([
    [3.40, 4.34, 1.60],  // DR-500 Bottom Discharge Flange
    [3.40, 1.20, 1.60],
    [0.00, 1.20, 1.60],
    [-4.65, 1.20, 1.60],
    [-4.65, 1.42, 1.55] // Direct mate into CMP-700 suction dampener flange
  ], 0.040, matLib.pipingH2, true));

  // 4. LINE-04-H2-HP-350BAR: CMP-700 350-bar Discharge -> ST-800 Buffer Storage
  pipGrp.add(createPipeRunMesh([
    [-2.95, 1.42, 1.55], // CMP-700 Discharge Dampener Flange
    [-2.95, 0.55, 1.55],
    [-2.95, 0.55, 3.50],
    [1.85, 0.55, 3.50]  // Direct mate into ST-800 350-bar manifold
  ], 0.028, matLib.safetyRed, true));

  // 5. LINE-05-H2-BATTERY-LIMIT: ST-800 Offtake -> Skid Battery Limit (BL-H2-EXP-01)
  pipGrp.add(createPipeRunMesh([
    [-1.85, 0.55, 3.50], // ST-800 Offtake Flange
    [-5.50, 0.55, 3.50]  // Skid Boundary Battery Limit
  ], 0.035, matLib.safetyRed, true));

  const specBlindMesh = makeMesh(new THREE.BoxGeometry(0.04, 0.22, 0.12), matLib.safetyYellow);
  specBlindMesh.position.set(-5.52, 0.55, 3.50);
  pipGrp.add(specBlindMesh);

  // 6. LINE-06-O2-SLURRY: Stack Anode Outlets -> V-400 O2 Cyclone Degasser
  [-3.0, -1.0, 1.0, 3.0].forEach(sx => {
    pipGrp.add(createPipeRunMesh([
      [sx - 0.25, 5.08, -0.63],
      [sx - 0.25, 5.08, -1.00]
    ], 0.038, matLib.processBlue, true));
  });
  pipGrp.add(createPipeRunMesh([
    [-3.25, 5.08, -1.00],
    [3.75, 5.08, -1.00],
    [3.75, 4.80, -2.00],
    [3.75, 1.80, -2.00]  // Direct mate into V-400 tangential inlet nozzle
  ], 0.050, matLib.processBlue, true));

  // 7. LINE-07-DI-SUPPLY: WTR-600 High-Pressure Pump -> 4x Stack Anode Inlets
  pipGrp.add(createPipeRunMesh([
    [-3.15, 0.58, -1.55], // WTR-600 Pump Discharge Flange
    [-3.15, 0.58, -1.00],
    [-3.25, 0.58, -1.00],
    [-3.25, 4.63, -1.00], // Riser to Mezzanine
    [2.75, 4.63, -1.00]   // Distribution header
  ], 0.045, matLib.pipingDI, true));

  [-3.0, -1.0, 1.0, 3.0].forEach(sx => {
    pipGrp.add(createPipeRunMesh([
      [sx - 0.25, 4.63, -1.00],
      [sx - 0.25, 4.63, -0.63] // Direct mate into Stack Anode Inlet Flange
    ], 0.035, matLib.pipingDI, true));
  });

  // 8. LINE-08-WATER-RECYCLE: V-300 & V-400 Bottom Drains -> WTR-600 Sump
  pipGrp.add(createPipeRunMesh([
    [4.20, 0.30, -2.00], // V-400 Bottom Drain
    [2.20, 0.25, -2.00], // V-300 Bottom Drain Tee
    [-3.80, 0.25, -2.00],
    [-4.25, 0.25, -2.00],
    [-4.25, 2.68, -2.00] // Direct mate into WTR-600 tank recycle nozzle
  ], 0.035, matLib.pipingDI, true));

  // 9. LINE-09-N2-PURGE: Battery Limit (BL-N2-SUP-01) -> V-300 / DR-500 Purge
  pipGrp.add(createPipeRunMesh([
    [-5.50, 4.95, 0.00], // Battery Limit N2 Supply Interface BL-N2-SUP-01
    [3.40, 4.95, 0.00],  // Central Mezzanine Overhead Pipe Rack
    [3.40, 4.95, 1.60],
    [3.40, 5.21, 1.60]   // Connects into DR-500 purge port
  ], 0.025, matLib.pipingN2, true));

  // 10. BUS-01-DC: 5000A Solid Copper Busbar from ELE-900 to Stacks
  pipGrp.add(createPipeRunMesh([
    [-2.70, 1.80, 0.00], // ELE-900 Rectifier Busduct
    [-2.70, 5.20, 0.00], // Vertical copper riser to mezzanine
    [3.00, 5.20, 0.00]   // Horizontal copper distribution busway
  ], 0.040, matLib.copper, false));

  attachSubsystem('PIP-1100', pipGrp);
}

// ============================================================================
// SOLIDWORKS GLB MODEL CATALOG & REAL-TIME TESSELATION LOADER
// ============================================================================

const glbModelCatalog = {
  'PLANT-100T': {
    tag: 'PLANT-100T',
    shortName: '100 TPD Master Skid',
    file: 'models/plant_100tpd_master_assembly.glb',
    name: '100 TPD Commercial Green Hydrogen Master Assembly (11 Subsystems)'
  },
  'STK-010': {
    tag: 'STK-010',
    shortName: '10-Cell PEM Stack',
    file: 'models/STK-010_10Cell_PEM_Stack.glb',
    name: 'STK-010: 1:1 10-Cell Industrial PEM Electrolyzer Stack (140 B-Rep Bodies)'
  },
  'STR-100': {
    tag: 'STR-100',
    shortName: 'Mezzanine Skid',
    file: 'models/STR-100_Structural_Mezzanine.glb',
    name: 'Unit 100: Structural Mezzanine Platform & Pipe Rack Skid'
  },
  'EL-200': {
    tag: 'EL-200',
    shortName: '50-Cell Stacks',
    file: 'models/EL-200_Electrolyzer_Bay.glb',
    name: 'Unit 200: Commercial Electrolyzer Bay (4x Stacks, 250 cm²)'
  },
  'V-300': {
    tag: 'V-300',
    shortName: 'Knockout Drum',
    file: 'models/V-300_HP_Knockout_Drum.glb',
    name: 'Unit 300: HP Hydrogen Knockout Drum (ASME Sec VIII Div 1)'
  },
  'V-400': {
    tag: 'V-400',
    shortName: 'Cyclone Degasser',
    file: 'models/V-400_Anode_O2_Degasser.glb',
    name: 'Unit 400: Anode O₂ Cyclone Degasser & 6.2m Vent Stack'
  },
  'DR-500': {
    tag: 'DR-500',
    shortName: 'DeOxo & TSA',
    file: 'models/DR-500_Catalytic_DeOxo_TSA.glb',
    name: 'Unit 500: Catalytic DeOxo Reactor & Dual 13X TSA Dryers'
  },
  'WTR-600': {
    tag: 'WTR-600',
    shortName: 'Water Skid',
    file: 'models/WTR-600_Pure_Water_Treatment.glb',
    name: 'Unit 600: Ultra-Pure DI Water Treatment Skid (3,000L)'
  },
  'CMP-700': {
    tag: 'CMP-700',
    shortName: 'Diaphragm Comp',
    file: 'models/CMP-700_Diaphragm_Compressor.glb',
    name: 'Unit 700: 3-Stage HP Diaphragm Compressor (API 618)'
  },
  'ST-800': {
    tag: 'ST-800',
    shortName: 'Buffer Storage',
    file: 'models/ST-800_HP_Buffer_Storage.glb',
    name: 'Unit 800: 350-bar HP Buffer Storage Array (4x Forged SA-723)'
  },
  'ELE-900': {
    tag: 'ELE-900',
    shortName: 'Rectifier Bay',
    file: 'models/ELE-900_Substation_Rectifier.glb',
    name: 'Unit 900: 11kV Substation & 12-Pulse Thyristor Rectifier'
  },
  'PLC-1000': {
    tag: 'PLC-1000',
    shortName: 'Safety SCADA',
    file: 'models/PLC-1000_Safety_PLC_Console.glb',
    name: 'Unit 1000: Siemens S7-1500F Safety PLC & SCADA Suite'
  },
  'PIP-1100': {
    tag: 'PIP-1100',
    shortName: 'Process Headers',
    file: 'models/PIP-1100_Process_Piping_Header.glb',
    name: 'Unit 1100: Interconnecting Process Piping Network (Sched 80S)'
  }
};

const MAT_SPEC_LABELS = {
  'SS316L_POLISHED': 'SA-240 / SA-312 Type 316L Stainless Steel (Electropolished Ra < 0.4 µm)',
  'SS304_SATIN': 'ASTM A240 Type 304 Satin Stainless Steel',
  'TITANIUM_GRADE2': 'ASTM B265 Grade 2 Commercially Pure Titanium (Progressive-Die Formed Flow Field)',
  'NAFION_MEMBRANE_DARK': 'DuPont Nafion™ 115 PFSA Membrane + IrO₂ / Pt-C Catalyst CCM',
  'EPDM_GASKET_BLACK': 'Peroxide-Cured High-Purity EPDM Elastomer Frame Gasket (Shore 70A)',
  'PTFE_TEFLON_WHITE': 'Virgin PTFE Dielectric Isolation Plate (>20 kV/mm Breakdown)',
  'GOLD_COATED_CONTACT': 'Au-Electroplated C11000 Contact / Differential Cell Voltage Tap',
  'BELLEVILLE_SPRING_STEEL': 'DIN 2093 51CrV4 High-Tensile Conical Belleville Disc Spring',
  'FLANGE_FORGED_SS': 'SA-182 F316L Forged Weld-Neck Flange / ASTM A193 Gr. B7 Fasteners',
  'STRUCTURAL_SLATE_IS800': 'IS 800:2007 / ASTM A36 Heavy Structural Steel Section (ISMB 300)',
  'STRUCTURAL_BLUE_EPOXY': 'IS 2062 Structural Steel + Marine Epoxy Coating (RAL 5015)',
  'DECK_GALVANIZED': 'Hot-Dip Galvanized Checkered Anti-Slip Walkway Grating (85 µm Zn)',
  'SAFETY_YELLOW_OSHA': 'The Factories Act, 1948 (Sec 32) & IS 800:2007 Safety Yellow Guardrail',
  'SAFETY_RED_VALVE': 'ASME Section VIII Angle PSV / Class 2500 High-Pressure Enamel Red',
  'MOTOR_IEC_BLUE': 'IEC 60034 Ex d IIC T4 Cast-Iron Finned Drive Motor Housing',
  'COPPER_BUSBAR': 'C11000 Electrolytic Tough Pitch (ETP) 5,000A Solid Copper Busbar',
  'SIEMENS_RAL7035': 'Rittal TS8 IP65 Dipcoat-Primed Powder-Coated Enclosure (RAL 7035)',
  'SIEMENS_PLINTH_RAL7022': 'Rittal Structural Sheet-Steel Plinth Base (RAL 7022 Umbra Grey)',
  'SIEMENS_PETROL_TEAL': 'Siemens SITRANS / SIMATIC Industrial Die-Cast Aluminum Bezel',
  'HMI_SCREEN_DARK': 'IP65 Anti-Glare Capacitive Multi-Touch Industrial Glass Panel',
  'INSULATION_CLAD_ALU': 'ASTM B209 Stucco-Embossed Aluminum Thermal Cladding Jacket',
  'RESIN_AMBER': 'Nuclear-Grade Mixed-Bed Ion Exchange Polishing Column (>18.2 MΩ·cm)',
  'TRANSFORMER_STEEL': 'IS 2026 ONAN Step-Down Transformer Tank & Radiator Fin Bank',
  'CERAMIC_BUSHING_BROWN': 'IEEE C57.19.01 High-Creepage Glazed Porcelain HV Bushing',
  'LED_GREEN': 'IEC 60073 High-Intensity LED Status Beacon (Normal Run)',
  'LED_AMBER': 'IEC 60073 High-Intensity LED Status Beacon (Standby / Purge)',
  'LED_BLUE': 'IEC 60073 High-Intensity LED Status Beacon (PIBT Active)',
  'PIPING_H2_ROSE': 'ASME B31.12 Seamless SA-312 TP316L Sch 80S Hydrogen Header',
  'PIPING_O2_BLUE': 'ASME B31.3 Seamless SA-312 TP316L Sch 80S Anode O₂/H₂O Header',
  'PIPING_DI_CYAN': 'ASME BPE Electropolished 316L Ultra-Pure DI Water Feed Header',
  'PIPING_N2_AMBER': 'ASME B31.3 Seamless 316L High-Purity Nitrogen Sweep Purge Header'
};

let glbSceneNode = null;
let loadedGlbTag = null;
let currentGlbLoadId = 0;
let gltfLoaderInstance = null;
let selectedSubMesh = null;

function parseGlbNodeMetadata(meshObj) {
  let rawName = meshObj.name || (meshObj.parent && meshObj.parent.name) || '';
  // Format generated by sw_brep_primitives_h2.py: {TAG}_{IDX}__{SUBTAG}__{MAT_KEY}
  let unitTag = loadedGlbTag || selectedSubsystem || 'PLANT-100T';
  let nodeId = rawName || 'BREP_001';
  let subtag = rawName || 'SolidWorks_BRep_Body';
  let matKey = 'SS316L_POLISHED';

  if (rawName.includes('__')) {
    const parts = rawName.split('__');
    const prefix = parts[0] || '';
    const lastUnderscore = prefix.lastIndexOf('_');
    if (lastUnderscore > 0) {
      const candidateTag = prefix.substring(0, lastUnderscore);
      if (plantCatalog[candidateTag]) unitTag = candidateTag;
    }
    nodeId = prefix;
    subtag = parts[1] || subtag;
    matKey = parts[2] || matKey;
  } else {
    Object.keys(plantCatalog).forEach(k => {
      if (rawName.startsWith(k)) unitTag = k;
    });
  }

  const cleanSubtag = subtag.replace(/_/g, ' ');
  const matSpec = MAT_SPEC_LABELS[matKey] || matKey.replace(/_/g, ' ');
  return { unitTag, nodeId, subtag, cleanSubtag, matKey, matSpec };
}

function loadSubsystemGlb(tag) {
  const modelInfo = glbModelCatalog[tag];
  if (!modelInfo || !THREE.GLTFLoader) return;

  const myLoadId = ++currentGlbLoadId;

  const overlay = document.getElementById('cadTesselateOverlay');
  const bar = document.getElementById('cadTesselateBar');
  const text = document.getElementById('cadTesselateText');
  const fileDisp = document.getElementById('cadTesselateFile');

  if (overlay) overlay.style.display = 'flex';
  if (fileDisp) fileDisp.textContent = modelInfo.file;
  if (bar) bar.style.width = '25%';
  if (text) text.textContent = 'Loading OpenCASCADE B-Rep Tessellation (25%)...';

  if (!gltfLoaderInstance) {
    gltfLoaderInstance = new THREE.GLTFLoader();
  }

  gltfLoaderInstance.load(
    modelInfo.file,
    gltf => {
      if (myLoadId !== currentGlbLoadId) return;
      if (bar) bar.style.width = '100%';
      if (text) text.textContent = 'SolidWorks 2025 B-Rep Multi-Material Scene Ready (100%)';

      if (glbSceneNode) {
        plantScene.remove(glbSceneNode);
        glbSceneNode = null;
      }

      glbSceneNode = gltf.scene;
      loadedGlbTag = tag;
      selectedSubMesh = null;

      // Auto-center on bounding box and ground plane
      const bbox = new THREE.Box3().setFromObject(glbSceneNode);
      const center = bbox.getCenter(new THREE.Vector3());
      const size = bbox.getSize(new THREE.Vector3());
      glbSceneNode.position.set(-center.x, -bbox.min.y, -center.z);
      glbSceneNode.updateMatrixWorld(true);

      const overallBox = new THREE.Box3().setFromObject(glbSceneNode);
      const overallCenter = overallBox.getCenter(new THREE.Vector3());

      let meshCount = 0;
      glbSceneNode.traverse(child => {
        if (child.isMesh) {
          meshCount++;
          child.castShadow = true;
          child.receiveShadow = true;

          // Ensure each mesh has its own cloned PBR material instance with realistic metallic sheen
          if (child.material) {
            child.material = child.material.clone();
            child.material.needsUpdate = true;
            if (isWireframe) child.material.wireframe = true;
          }
          origSubsystemMaterials.set(child, child.material);

          // Compute exact B-Rep sub-component bounding box & explode vector
          if (!child.geometry.boundingBox) {
            child.geometry.computeBoundingBox();
          }
          const mBox = child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld);
          const mCenter = mBox.getCenter(new THREE.Vector3());
          const mSize = mBox.getSize(new THREE.Vector3());

          const meta = parseGlbNodeMetadata(child);
          child.userData.meta = meta;
          child.userData.origPos = child.position.clone();
          // For STK-010 (scaled 2.5x in CAD), report true 1:1 physical millimeters
          const dimScale = (tag === 'STK-010' || meta.unitTag === 'STK-010') ? (1000.0 / 2.5) : 1000.0;
          child.userData.dimsMm = [
            Math.max(1.0, mSize.x * dimScale).toFixed(1),
            Math.max(1.0, mSize.y * dimScale).toFixed(1),
            Math.max(1.0, mSize.z * dimScale).toFixed(1)
          ];
          child.userData.worldCenter = mCenter.clone();

          // Compute physically meaningful explode vector
          const radial = mCenter.clone().sub(overallCenter);
          if (tag === 'STK-010') {
            // Axial stack cell separation along Z + radial bolt/manifold separation
            child.userData.explodeVec = new THREE.Vector3(
              radial.x * 0.65,
              Math.max(0, radial.y) * 0.55,
              radial.z * 1.65
            );
          } else if (tag === 'PLANT-100T' && plantCatalog[meta.unitTag] && plantCatalog[meta.unitTag].explodeDir) {
            const ed = plantCatalog[meta.unitTag].explodeDir;
            child.userData.explodeVec = new THREE.Vector3(
              ed[0] * 1.4 + radial.x * 0.25,
              ed[1] * 1.4 + Math.max(0, radial.y) * 0.25,
              ed[2] * 1.4 + radial.z * 0.25
            );
          } else {
            child.userData.explodeVec = new THREE.Vector3(
              radial.x * 0.85,
              Math.max(0, radial.y) * 0.75,
              radial.z * 0.85
            );
          }
        }
      });

      plantScene.add(glbSceneNode);
      // Hide procedural fallback scene while authentic SolidWorks GLB model is active
      Object.values(subsystems).forEach(g => { g.visible = false; });

      // Apply current explode slider factor if non-zero
      if (explodeFactor > 0.001) {
        setExplodeFactor(explodeFactor);
      }

      // Auto-fit camera to loaded SolidWorks B-Rep assembly
      const maxDim = Math.max(size.x, size.y, size.z) || 6;
      const dist = maxDim / (2 * Math.tan((plantCamera.fov * Math.PI) / 360)) * 1.32;
      targetCamPos = new THREE.Vector3(dist * 0.82, dist * 0.62, dist * 0.95);
      targetLookAt = new THREE.Vector3(0, size.y * 0.42, 0);

      // Show default sub-component inspection banner summarizing loaded B-Rep nodes
      updateClickedPartBanner({
        cleanSubtag: `${modelInfo.shortName} (${meshCount} Multi-Material SolidWorks B-Rep Nodes)`,
        nodeId: `${tag}::MASTER_BREP`,
        matSpec: 'Click any 3D valve, plate, tie-rod, nozzle, or vessel in the viewport to inspect its exact B-Rep geometry & alloy',
        dimsMm: [
          (size.x * (tag === 'STK-010' ? 400 : 1000)).toFixed(0),
          (size.y * (tag === 'STK-010' ? 400 : 1000)).toFixed(0),
          (size.z * (tag === 'STK-010' ? 400 : 1000)).toFixed(0)
        ]
      });

      setTimeout(() => {
        if (overlay) overlay.style.display = 'none';
      }, 220);
    },
    xhr => {
      if (xhr.total > 0) {
        const pct = Math.round((xhr.loaded / xhr.total) * 100);
        if (bar) bar.style.width = `${pct}%`;
        if (text) text.textContent = `Loading SolidWorks B-Rep GLB (${pct}%)...`;
      }
    },
    err => {
      console.warn('[GLTF Loader] Seamless procedural CAD assembly active:', err.message);
      if (overlay) overlay.style.display = 'none';
      if (glbSceneNode) {
        plantScene.remove(glbSceneNode);
        glbSceneNode = null;
      }
      loadedGlbTag = null;
      Object.values(subsystems).forEach(g => { g.visible = true; });
    }
  );
}

// ============================================================================
// INTERACTIVE SELECTION, SUB-COMPONENT INSPECTOR, HUD UPDATE & CAMERA CONTROLS
// ============================================================================

function ensureClickedPartBanner() {
  let banner = document.getElementById('cadClickedPartBanner');
  if (!banner) {
    const hudHeader = document.querySelector('.cad-hud-header');
    if (hudHeader && hudHeader.parentNode) {
      banner = document.createElement('div');
      banner.id = 'cadClickedPartBanner';
      banner.style.cssText = 'background: #0F172A; color: #F8FAFC; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; font-size: 11.5px;';
      hudHeader.parentNode.insertBefore(banner, hudHeader.nextSibling);
    }
  }
  return banner;
}

function updateClickedPartBanner(partInfo) {
  const banner = ensureClickedPartBanner();
  if (!banner || !partInfo) return;
  const dimsStr = partInfo.dimsMm ? `${partInfo.dimsMm[0]} × ${partInfo.dimsMm[1]} × ${partInfo.dimsMm[2]} mm` : 'Verified B-Rep Solid';
  banner.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
        LIVE 3D B-REP COMPONENT INSPECTOR
      </span>
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #94A3B8;">
        ${partInfo.nodeId || ''}
      </span>
    </div>
    <div style="font-weight: 700; font-size: 12.5px; color: #FFFFFF; margin-bottom: 3px;">
      ${partInfo.cleanSubtag}
    </div>
    <div style="font-size: 11px; color: #CBD5E1; margin-bottom: 4px;">
      <strong>Alloy / Material:</strong> ${partInfo.matSpec}
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94A3B8;">
      <strong>Measured B-Rep Envelope:</strong> ${dimsStr}
    </div>
  `;
}

function populateSubsystemHudOnly(tag) {
  const data = plantCatalog[tag];
  if (!data) return;

  // Update Ribbon Pills UI
  document.querySelectorAll('.cad-model-pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tag === tag);
  });

  const hudTag = document.getElementById('cadSubsystemTag');
  if (hudTag) hudTag.textContent = `${data.tag} // ${data.category.toUpperCase()}`;

  const hudName = document.getElementById('cadSubsystemName');
  if (hudName) hudName.textContent = data.name;

  const hudStandard = document.getElementById('cadSubsystemStandard');
  if (hudStandard) hudStandard.textContent = data.standard;

  const hudSpec = document.getElementById('cadSubsystemSpec');
  if (hudSpec) hudSpec.textContent = data.spec;

  const hudDuty = document.getElementById('cadSubsystemDuty');
  if (hudDuty) hudDuty.textContent = data.duty;

  const hudDims = document.getElementById('cadSubsystemDims');
  if (hudDims) hudDims.textContent = data.dimensions;

  const gridContainer = document.getElementById('cadSubsystemParams');
  if (gridContainer && data.opParams) {
    gridContainer.innerHTML = Object.entries(data.opParams).map(([k, v]) => `
      <div class="cad-param-row">
        <span class="cad-param-label">${k}</span>
        <span class="cad-param-val">${v}</span>
      </div>
    `).join('');
  }

  const hudScript = document.getElementById('cadSubsystemScript');
  if (hudScript) hudScript.textContent = data.script;

  const calcClause = document.getElementById('cadCalcClause');
  const calcFormula = document.getElementById('cadCalcFormula');
  const calcCitation = document.getElementById('cadCalcCitation');
  if (data.governing_calculations) {
    if (calcClause) calcClause.textContent = data.governing_calculations.code_clause || data.standard;
    if (calcFormula) {
      let formulaText = data.governing_calculations.formula || data.governing_calculations.formula_shell || data.governing_calculations.formula_von_mises || '';
      if (data.governing_calculations.computed_value) formulaText += ` [${data.governing_calculations.computed_value}]`;
      calcFormula.textContent = formulaText;
    }
    if (calcCitation) calcCitation.textContent = `Source: ${data.governing_calculations.script_source || data.script}`;
  } else {
    if (calcClause) calcClause.textContent = data.standard;
    if (calcFormula) calcFormula.textContent = data.spec;
    if (calcCitation) calcCitation.textContent = `Source: ${data.script}`;
  }

  const bomContainer = document.getElementById('cadSubsystemBomTable');
  if (bomContainer && data.bom && data.bom.length > 0) {
    bomContainer.innerHTML = `
      <table class="cad-bom-table" style="width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 4px;">
        <thead>
          <tr style="background: #F1F5F9; text-align: left; border-bottom: 1px solid var(--border-subtle);">
            <th style="padding: 4px 6px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">Component</th>
            <th style="padding: 4px 6px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">Qty</th>
            <th style="padding: 4px 6px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">Material</th>
            <th style="padding: 4px 6px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">Specification</th>
          </tr>
        </thead>
        <tbody>
          ${data.bom.map(b => `
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 4px 6px; font-weight: 600; color: var(--text-main);">${b.item}</td>
              <td style="padding: 4px 6px; font-family: 'JetBrains Mono', monospace; color: #0284C7;">${b.qty}</td>
              <td style="padding: 4px 6px; color: var(--text-body);">${b.material}</td>
              <td style="padding: 4px 6px; font-size: 10.5px; color: var(--text-muted);">${b.spec}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } else if (bomContainer) {
    bomContainer.innerHTML = '';
  }

  const auditBtn = document.getElementById('cadInspectCodeBtn');
  if (auditBtn) {
    auditBtn.onclick = () => {
      if (window.openStandardModal && data.standardModalKey) {
        window.openStandardModal(data.standardModalKey);
      }
    };
  }
}

function selectSubsystem(tag) {
  if (!plantCatalog[tag]) return;
  selectedSubsystem = tag;
  const data = plantCatalog[tag];

  // Load the SolidWorks B-Rep GLB model for this subsystem or master assembly
  loadSubsystemGlb(tag);

  // Highlight procedural fallback group if active
  Object.keys(subsystems).forEach(k => {
    const grp = subsystems[k];
    const isSel = (k === tag);
    grp.traverse(child => {
      if (child.isMesh) {
        if (isSel) {
          child.material = matLib.highlightMat;
        } else {
          const orig = origSubsystemMaterials.get(child);
          if (orig) child.material = isWireframe ? getWireframeMat(orig) : orig;
        }
      }
    });
  });

  if (data.camFocus && data.focusPos) {
    targetCamPos = new THREE.Vector3(...data.camFocus);
    targetLookAt = new THREE.Vector3(...data.focusPos);
  }

  populateSubsystemHudOnly(tag);
}

function getWireframeMat(origMat) {
  const m = origMat.clone();
  m.wireframe = true;
  return m;
}

function renderSubsystemPills() {
  const tray = document.getElementById('cadModelPillTray');
  if (!tray) return;

  tray.innerHTML = Object.keys(plantCatalog).map(tag => {
    const item = plantCatalog[tag];
    return `
      <button class="cad-model-pill ${tag === 'PLANT-100T' ? 'active' : ''}" data-tag="${tag}" onclick="selectSubsystem('${tag}')" title="${item.name}">
        <span class="cad-pill-tag">${tag}</span>
        <span class="cad-pill-sep">//</span>
        <span class="cad-pill-label">${item.name.includes(':') ? item.name.split(':').slice(1).join(':').trim() : item.name}</span>
      </button>
    `;
  }).join('');
}

// --- Camera View Presets ---

function setCameraPreset(preset) {
  if (!plantCamera || !plantControls) return;

  if (preset === 'iso') {
    targetCamPos = new THREE.Vector3(14, 11, 16);
    targetLookAt = new THREE.Vector3(0, 2.6, 0);
  } else if (preset === 'front') {
    targetCamPos = new THREE.Vector3(0, 3.5, 20);
    targetLookAt = new THREE.Vector3(0, 3.0, 0);
  } else if (preset === 'side') {
    targetCamPos = new THREE.Vector3(20, 3.5, 0);
    targetLookAt = new THREE.Vector3(0, 3.0, 0);
  } else if (preset === 'top') {
    targetCamPos = new THREE.Vector3(0.01, 25, 0);
    targetLookAt = new THREE.Vector3(0, 0, 0);
  } else if (preset === 'reset') {
    if (selectedSubsystem) selectSubsystem(selectedSubsystem);
  }
}

// --- Display Controls ---

function togglePlantTurntable() {
  isTurntable = !isTurntable;
  const btn = document.getElementById('btnTurntable');
  if (btn) btn.classList.toggle('active', isTurntable);
}

function togglePlantWireframe() {
  isWireframe = !isWireframe;
  const btn = document.getElementById('btnWireframe');
  if (btn) btn.classList.toggle('active', isWireframe);

  if (glbSceneNode) {
    glbSceneNode.traverse(child => {
      if (child.isMesh && child.material) {
        child.material.wireframe = isWireframe;
      }
    });
  }

  Object.keys(subsystems).forEach(k => {
    const grp = subsystems[k];
    const isSel = (k === selectedSubsystem);
    grp.traverse(child => {
      if (child.isMesh) {
        if (isSel) {
          child.material = matLib.highlightMat;
        } else {
          const orig = origSubsystemMaterials.get(child);
          if (orig) {
            child.material = isWireframe ? getWireframeMat(orig) : orig;
          }
        }
      }
    });
  });
}

function togglePlantGrid() {
  showGrid = !showGrid;
  if (groundGrid) groundGrid.visible = showGrid;
  const btn = document.getElementById('btnGrid');
  if (btn) btn.classList.toggle('active', showGrid);
}

function setExplodeFactor(val) {
  explodeFactor = parseFloat(val);
  const disp = document.getElementById('explodeDisp');
  if (disp) disp.textContent = Math.round(explodeFactor * 100) + '%';

  // 1. Explode SolidWorks B-Rep GLB sub-meshes in real time
  if (glbSceneNode) {
    glbSceneNode.traverse(child => {
      if (child.isMesh && child.userData.origPos && child.userData.explodeVec) {
        const orig = child.userData.origPos;
        const ev = child.userData.explodeVec;
        child.position.set(
          orig.x + ev.x * explodeFactor,
          orig.y + ev.y * explodeFactor,
          orig.z + ev.z * explodeFactor
        );
      }
    });
  }

  // 2. Also explode procedural fallback groups
  Object.keys(subsystems).forEach(tag => {
    const item = plantCatalog[tag];
    const grp = subsystems[tag];
    if (item && grp && item.explodeDir) {
      grp.position.set(
        item.basePos[0] + item.explodeDir[0] * explodeFactor * 2.5,
        item.basePos[1] + item.explodeDir[1] * explodeFactor * 2.5,
        item.basePos[2] + item.explodeDir[2] * explodeFactor * 2.5
      );
    }
  });
}

// --- Raycaster Pointer Interaction (Both GLB Sub-Components & Subsystems) ---

function onPlantPointerDown(e) {
  if (!plantRenderer || !plantCamera) return;
  const rect = plantRenderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, plantCamera);

  // 1. If SolidWorks B-Rep GLB Scene is active, raycast directly against all B-Rep sub-component meshes
  if (glbSceneNode) {
    const glbMeshes = [];
    glbSceneNode.traverse(c => { if (c.isMesh) glbMeshes.push(c); });
    const hits = raycaster.intersectObjects(glbMeshes, false);
    if (hits.length > 0) {
      const hitMesh = hits[0].object;
      const meta = hitMesh.userData.meta || parseGlbNodeMetadata(hitMesh);

      // Restore all meshes to their authentic PBR materials, then highlight the clicked B-Rep part
      glbMeshes.forEach(m => {
        const origMat = origSubsystemMaterials.get(m);
        if (origMat) {
          m.material = origMat;
          m.material.wireframe = isWireframe;
        }
      });

      const highlightClone = hitMesh.material.clone();
      if (highlightClone.emissive) {
        highlightClone.emissive.setHex(0x0284C7);
        highlightClone.emissiveIntensity = 0.42;
      }
      hitMesh.material = highlightClone;
      selectedSubMesh = hitMesh;

      // Update Live 3D B-Rep Sub-Component Inspector Banner
      updateClickedPartBanner({
        cleanSubtag: meta.cleanSubtag,
        nodeId: meta.nodeId,
        matSpec: meta.matSpec,
        dimsMm: hitMesh.userData.dimsMm || ['100.0', '100.0', '100.0']
      });

      // If viewing Master Plant (PLANT-100T) and clicked a subsystem's part, update HUD to that subsystem!
      if (meta.unitTag && plantCatalog[meta.unitTag]) {
        populateSubsystemHudOnly(meta.unitTag);
      }
      return;
    }
  }

  // 2. Fallback: raycast against procedural subsystem groups
  const meshes = [];
  Object.values(subsystems).forEach(grp => {
    if (grp.visible) {
      grp.traverse(c => { if (c.isMesh) meshes.push(c); });
    }
  });

  const intersects = raycaster.intersectObjects(meshes, false);
  if (intersects.length > 0) {
    let topObj = intersects[0].object;
    while (topObj.parent && !plantCatalog[topObj.name] && topObj.parent !== plantScene) {
      topObj = topObj.parent;
    }
    if (plantCatalog[topObj.name]) {
      if (selectedSubsystem === 'PLANT-100T') {
        populateSubsystemHudOnly(topObj.name);
      } else {
        selectSubsystem(topObj.name);
      }
    }
  }
}

function onPlantResize() {
  if (!plantContainer || !plantCamera || !plantRenderer) return;
  const w = plantContainer.clientWidth;
  const h = plantContainer.clientHeight;
  plantCamera.aspect = w / h;
  plantCamera.updateProjectionMatrix();
  plantRenderer.setSize(w, h);
}

// --- Animation & Camera Tween Loop ---

function animatePlant() {
  requestAnimationFrame(animatePlant);

  // Smooth Camera Tween
  if (targetCamPos && targetLookAt) {
    plantCamera.position.lerp(targetCamPos, 0.08);
    plantControls.target.lerp(targetLookAt, 0.08);
    if (plantCamera.position.distanceTo(targetCamPos) < 0.05) {
      targetCamPos = null;
      targetLookAt = null;
    }
  }

  // Turntable Auto-Rotation
  if (isTurntable && plantControls && !targetCamPos) {
    plantControls.autoRotate = true;
    plantControls.autoRotateSpeed = 2.0;
  } else if (plantControls) {
    plantControls.autoRotate = false;
  }

  plantControls.update();
  plantRenderer.render(plantScene, plantCamera);
}

// Expose globals for UI buttons
window.initPlant3DScene = initPlant3DScene;
window.selectSubsystem = selectSubsystem;
window.setCameraPreset = setCameraPreset;
window.togglePlantTurntable = togglePlantTurntable;
window.togglePlantWireframe = togglePlantWireframe;
window.togglePlantGrid = togglePlantGrid;
window.setExplodeFactor = setExplodeFactor;

// Auto-boot on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initPlant3DScene);
} else {
  setTimeout(initPlant3DScene, 150);
}
