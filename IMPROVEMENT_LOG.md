# IMPROVEMENT LOG: Continuous Enhancement & Engineering History

## [v3.1.0] - 2026-11-06 - Production-Grade 3D CAD Workstation & Typography Unslop
### Added
- **Parametric Solid Model & Assembly Workstation (FinnoaQ Architecture)**:
  - Redesigned Section 4 to match and exceed industrial standard `portal-nine-self-55.vercel.app`.
  - Subsystem Switcher Ribbon Bar on top with 11 discrete pills (`STR-100` through `PIP-1100`).
  - 2-Column Split Workstation: 8-column 3D WebGL Canvas + 4-column Subsystem Engineering Inspector Panel.
  - Camera View Presets with smooth tween interpolation: `Iso`, `Front`, `Side`, `Top`, and `Reset`.
  - Interactive Display Controls: `Auto-Rotate` (turntable), `Wireframe`, `Ground Grid`, and `Explode` (0% to 100% radial expansion).
  - Studio Three.js PBR Rendering: Real-time ground contact shadows (`PCFSoftShadowMap`), 3-point key/fill/rim lighting, and realistic industrial materials (Brushed 316L, Grade 2 Titanium, Forged Steel, Electrolytic Copper, Enamel Yellow).
  - Mechanical geometric fidelity: Weld-neck flanges with bolt circles, nozzle schedules, level bridles with differential radar, angle PSVs with discharge horns, 50-cell stack packs with machined flow channel grooves, 16x tension tie-rods with hex nuts, 5000A busbars with flexible jumpers, 3-stage compressor heads with finned motor, and ASME B31.12 color-coded piping network.
  - Subsystem Engineering Inspector with live telemetry, operational parameters, ASME/API/IEC code cross-references, and direct modal trigger.

### Changed / Unslop
- **Eliminated All Microscopic/Squished Text in Fig 7 & Fig 8**:
  - Rebuilt Figure 7 via `rebuild_clean_paper_graphs.py` into a pure 2-panel 300 DPI scientific CAE plot (Panel A: Stamping Profile, Panel B: ASME Sec VIII Wall Thickness vs Operating Pressure), completely removing the squished 5-column panel (c).
  - Rebuilt Figure 8 into a pure full-width 300 DPI Customer ROI Payback Curve with breakeven marker at Month 4.8 and shaded positive cash flow zone, completely eliminating small text boxes from inside the image.
  - Replaced the cramped 1-row 5-column sector cards in Section 6 with `.sector-spacious-grid` (`minmax(360px, 1fr)`), featuring large bold badges (`SEC-01` to `SEC-05`), 17px bold titles, clear uppercase labels, and readable 13.5px body text (`#1E293B`, line-height 1.55).
  - Verified visual rendering via headless browser screenshot testing at 1440x900 resolution (`verified_cad_workstation.png`, `verified_multi_plant_section.png`, `verified_techno_economics_section.png`).

## [v3.0.0] - 2026-11-06 - Complete Ocean Boiling & Deterministic Engineering Overhaul
### Added
- **5 Python Engineering Standards Engines**:
  - `asme_bpvc_calculations.py`: ASME Sec VIII Div 1 UG-27/UG-32 & Div 3 KD-250 sizing.
  - `iec_61511_sil2_lopa_safety.py`: Quantitative PFD_avg calculation for 1oo2 SIFs.
  - `stack_30bar_differential_mechanics.py`: Pore-bridging stress and ISO 22734 crossover kinetics.
  - `iso_14687_grade_d_purification.py`: Stoichiometric DeOxo mass balance and twin 13X TSA bed sizing.
  - `astm_b265_titanium_dfm.py`: Titanium Gr 2 progressive stamping strain and tonnage calculations.
  - `run_full_engineering_audit.py`: Master test suite outputting certified `engineering_audit_certificate.json`.
- **Three.js WebGL Interactive 3D Plant Model Viewer**:
  - Full 3D parametric scene representing all 11 subsystems with orbit/pan controls.
  - Raycaster click-to-inspect highlighting components and revealing live engineering specs.
  - Radial explode view slider (`0%` to `100%`) and wireframe/X-ray mode.
- **Interactive Engineering Verification Modals**:
  - Clicking any of the 5 standards tags opens the live calculation modal displaying governing formulas, input parameters, and direct code links.
- **7 Structured Memory Files (MF System)**:
  - `PROJECT_VISION.md`, `ARCHITECTURE.md`, `SECURITY.md`, `PERFORMANCE.md`, `UX.md`, `TESTING.md`, `IMPROVEMENT_LOG.md`.

### Changed / Unslop
- **Eliminated Fake Certification Claims**: Replaced static marketing buzzwords ("Certified") with verifiable compliance specifications linked directly to calculation scripts.
- **Procurement & EPC Re-Framing**: Converted adversarial startup cliches ("Monopoly Architecture", "Why Competition is for Losers") into industrial plant reliability language (TCO, LCOH, Battery Limits, RAM, MTBF).
- **Harden KaTeX Math Rendering**: Removed single-dollar auto-render delimiters to eliminate parser collisions with currency amounts (`$420,000`, `-$0.42/kg`), adding responsive overflow scroll rules.

