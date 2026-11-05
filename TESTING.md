# TESTING: Test Coverage, Engineering Audits & Regression Suites

## 1. Automated Engineering Standards Test Suite (`run_full_engineering_audit.py`)
Deterministic mathematical verification across 5 industrial codebases:
1. **ASME BPVC Section VIII Div 1 (UG-27, UG-32) & Div 3 (KD-250)**:
   - Validates V-300 shell thickness: $t_{\text{calc}} = 10.60\text{ mm} + 1.5\text{ mm CA} = 12.10\text{ mm} \le 14\text{ mm}$ plate.
   - Validates V-300 head thickness: $t_{\text{calc}} = 10.46\text{ mm} + 1.5\text{ mm CA} = 11.96\text{ mm} \le 12\text{ mm}$ plate.
   - Validates ST-800 350-bar forged cylinder Lame von Mises stress: $\sigma_{\text{vm}} = 148.48\text{ MPa} \le 725\text{ MPa}$ yield (Safety margin: 4.88x).
2. **IEC 61511 / IEC 61508 Functional Safety & SIL-2 LOPA**:
   - SIF-01 (Anode Crossover Quench): $PFD_{\text{avg}} = 0.005150$, $RRF = 194.3$ (SIL 2 verified).
   - SIF-02 (Stack Delta-P Interlock): $PFD_{\text{avg}} = 0.008360$, $RRF = 119.6$ (SIL 2 verified).
3. **30-Bar Differential Pressure Stack Mechanics & ISO 22734**:
   - Membrane pore-bridging stress: $0.046\text{ MPa}$ (Safety margin: 530x on 24.5 MPa reinforced PFSA).
   - Clamping force on PTL: $3.3\text{ MPa}$ (within 2–4 MPa optimal conductivity window).
   - H2 crossover in O2: $0.262\ \text{vol}\%$ (well below 1.0 vol% ISO 22734 threshold; 15.2x margin to LEL).
4. **ISO 14687:2019 Grade D Fuel Quality**:
   - DeOxo reactor effluent: $\text{O}_2 = 0.5\ \text{ppm}$ ($\le 5.0\ \text{ppm}$ ISO limit).
   - Dual 13X TSA dryer product: $\text{H}_2\text{O} = 1.0\ \text{ppm}$ ($\le 5.0\ \text{ppm}$ limit, dew point $-70^\circ\text{C}$).
5. **ASTM B265 Grade 2 Titanium DFM Stamping**:
   - Progressive stamping thinning: $16.36\%$ ($< 18.0\%$ safe limit; no necking).
   - Hydrostatic proof pressure: $42.39\text{ bar}$.

## 2. Regression & Frontend Tests
- **JavaScript Syntax Verification**: All frontend scripts (`main.js`, `cad-studio.js`, `plant-3d-scene.js`, `ml-workbench.js`, `roi-calculator.js`, `telemetry-scope.js`, `deck-modal.js`) verified via Node syntax compilation (`node -c`).
- **LaTeX Math Containment**: Zero auto-render collisions between single dollar currency symbols (`$420,000`, `-$0.42/kg`) and LaTeX proofs; overflow containment enabled via `.katex-display { overflow-x: auto }`.

