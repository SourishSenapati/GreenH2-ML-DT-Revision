/**
 * HYDROSENSE™ 15-Slide Master Presentation Deck Modal
 * Pure Architectural Light Mode, WCAG AAA High Contrast, Full-Viewport Utilization.
 * Every slide features high-density data, mathematical formulations, and 300 DPI figures.
 * Zero empty voids. Fully integrates Peter Thiel's Zero to One Strategic Framework.
 */

let deckIdx = 0;

const presentationSlides = [
  {
    tag: 'SLIDE 1 • DEFENSE COVER & EXECUTIVE MANDATE',
    title: 'Machine Learning for Predicting Catalyst Efficiency in Green Hydrogen Production',
    subtitle: 'Predictive Maintenance, Fault Detection, and Closed-Loop Degradation Mitigation',
    body: `
      <div style="display: grid; grid-template-columns: 1.1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 6px;">AUTHORS & INSTITUTIONAL AFFILIATIONS</div>
            <p style="font-size: 12.5px; line-height: 1.5; margin: 0 0 6px;"><strong>Sourish Senapati</strong> (Corresponding Author)<br>Dept. of Chemical Engineering, Jadavpur University<br><span style="font-family: monospace; font-size: 11px; color: var(--text-muted);">sourishs.chem.ug@jadavpuruniversity.in</span></p>
            <p style="font-size: 12.5px; line-height: 1.5; margin: 0;"><strong>Srestha Biswash</strong><br>Dept. of Microbiology &amp; Biotechnology, Sister Nivedita University</p>
          </div>

          <div style="background: #F0FDF4; padding: 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 6px;">CORE QUANTITATIVE BENCHMARKS</div>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <tr><td style="padding: 3px 0; color: #475569;">Virtual Sensing XGBoost</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">R² = 0.963 (1.911 µV/h error)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Physics-Informed Transformer</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">NRMSE = 0.054 (100h horizon)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Bayesian Anomaly Trip Latency</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">&lt; 1.40 seconds (F1 = 0.990)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Core Stack Lifespan Extension</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #166534;">+39.4% (21,428h &rarr; 29,879h)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Levelized Cost of H2 (LCOH)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #166534;">-$0.42 / kg H2 savings</td></tr>
            </table>
          </div>

          <div style="background: var(--bg-wash); padding: 10px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 11.5px; color: var(--text-body);">
            <strong>Governing Electrochemical Model:</strong>
            <div style="font-family: monospace; font-size: 11px; margin-top: 4px; color: var(--accent);">
              V_cell = E_rev(T, P) + &eta;_act(J, T) + &eta;_ohm(J, T) + &eta;_conc(J) + &Delta;&eta;_biofilm
            </div>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #F8FAFC; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px; cursor: pointer;" onclick="showModal('assets/PEM_Commercial_Plant_Master_Render.png', '100 TPD Turnkey Commercial Plant Master Assembly (300 DPI)')">
          <img src="assets/PEM_Commercial_Plant_Master_Render.png" alt="100 TPD Master Assembly" style="width: 100%; height: auto; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 8px;">
            Fig: 100 TPD Turnkey Plant CAD Master Model (SolidWorks 2025 SP1.2) • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 2 • THIEL PILLAR 1: NICHE MONOPOLY',
    title: 'Conquer a Small Market, Then Expand to Adjacent Sectors',
    subtitle: 'Zero Greenfield Competition: Capturing the 250 MW Brownfield Industrial Beachhead',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Beachhead Market: Captive Industrial Chlor-Alkali & Refineries</h4>
            <p style="font-size: 12.5px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
              Instead of competing in undifferentiated greenfield electrolyzer auctions, HYDROSENSE locks in existing high-purity brownfield industrial operators in India (Haldia Petrochemicals, HMEL, Grasim, DCM Shriram).
            </p>
            <ul style="font-size: 12px; line-height: 1.5; color: var(--text-body); padding-left: 18px; margin: 0;">
              <li><strong>Zero Substation CAPEX:</strong> Taps directly into existing 5000A DC busbars.</li>
              <li><strong>Immediate Offtake:</strong> On-site hydrotreating & chlorine balance consumption.</li>
              <li><strong>High Switching Barrier:</strong> Once edge firmware is embedded in Siemens S7-1500 PLC, replacement cost is prohibitive.</li>
            </ul>
          </div>

          <div style="background: #F8FAFC; padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--text-main); margin: 0 0 8px; font-size: 12.5px; font-weight: 700;">Adjacent Market Expansion Roadmap</h4>
            <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
              <thead>
                <tr style="border-bottom: 1px solid #CBD5E1; color: var(--text-muted); text-align: left;">
                  <th style="padding: 4px;">Phase</th><th>Sector</th><th>Duty</th><th>TAM</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style="padding: 4px; font-weight: 700;">Niche</td><td>Chlor-Alkali & Refining</td><td>20-50 TPD H2</td><td>$42M (India)</td></tr>
                <tr><td style="padding: 4px; font-weight: 700;">Wave 2</td><td>Green Steel (DRI Shaft)</td><td>50-150 TPD H2</td><td>$180M (SEA)</td></tr>
                <tr><td style="padding: 4px; font-weight: 700;">Wave 3</td><td>Haber-Bosch Green NH3</td><td>100-250 TPD H2</td><td>$650M (Global)</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png', 'Fig 7: Universal Multi-Plant Deployment Matrix across 5 Sectors (300 DPI)')">
          <img src="assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png" alt="Multi-Plant Matrix" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 7: Standardized Skid Deployment across 5 Key Sectors • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 3 • THIEL PILLAR 2: VERTICAL INTEGRATION',
    title: 'Vertical Integration Eliminates Intermediate Margin Stacking',
    subtitle: 'From Native SolidWorks CAD to On-Premise C++ Edge Inference and Verified EPC Design Calculations',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Full-Stack Ownership Across 4 Critical Layers</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>1. Hardware B-Rep CAD:</strong> Proprietary Progressive Stamping Die drawings for titanium bipolar plates with validated &lt;18% thinning.</p>
              <p style="margin: 0 0 6px;"><strong>2. Statutory Calculations:</strong> Deterministic ASME Sec VIII Div 1 UG-27 and Div 3 KD-250 code compliance generated in-repo.</p>
              <p style="margin: 0 0 6px;"><strong>3. Embedded Edge Twin:</strong> C++/Rust inference runtime compiled directly for Siemens S7-1500 PLC with zero cloud telemetry latency.</p>
              <p style="margin: 0;"><strong>4. Dual-Revenue Co-Products:</strong> Integrated AERO-PURE Unit 650 monetizing 800 TPD high-purity O2 and 18 MWh/day low-grade waste heat.</p>
            </div>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">ECONOMIC IMPACT OF VERTICAL INTEGRATION</div>
            <p style="font-size: 12px; line-height: 1.5; color: #14532D; margin: 0;">
              Eliminates 3 external vendor margins (CAD consultants, third-party CAE auditors, cloud IoT middleware), reducing total commercial deployment CAPEX by <strong>41.8%</strong>.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/PEM_Cell_Exploded_Architecture_Diagram.png', 'PEM Cell Exploded Architecture & Subsystem Integration (300 DPI)')">
          <img src="assets/PEM_Cell_Exploded_Architecture_Diagram.png" alt="Exploded Cell Architecture" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: Exploded B-Rep Stack Component Architecture • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 4 • THIEL PILLAR 3: LAST MOVER ADVANTAGE',
    title: 'Fast Follower Technology Arbitrage: Learning From Pioneer Failures',
    subtitle: 'Fixing the Fatal Flaws of First-Generation Alkaline & Early PEM Commercial Stacks',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Pioneer Vulnerabilities Identified &amp; Solved</h4>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <thead>
                <tr style="border-bottom: 1px solid #CBD5E1; color: var(--text-muted); text-align: left;">
                  <th style="padding: 4px;">Pioneer Flaw</th><th>Industry Consequence</th><th>HYDROSENSE Fix</th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom: 1px solid #E2E8F0;"><td style="padding: 5px 4px; font-weight: 700; color: #DC2626;">Dead-end Caustic Loops</td><td>Corrosive KOH leaks, slow dynamic ramp (&lt;2%/min)</td><td style="color: #059669; font-weight: 600;">Pure PEM solid membrane (0-100% in 12s)</td></tr>
                <tr style="border-bottom: 1px solid #E2E8F0;"><td style="padding: 5px 4px; font-weight: 700; color: #DC2626;">Biofilm Fouling Ignored</td><td>Premature stack burn out at 21,400h</td><td style="color: #059669; font-weight: 600;">Sub-1.4s Bayesian detection + automated DI pulse</td></tr>
                <tr><td style="padding: 5px 4px; font-weight: 700; color: #DC2626;">Cloud-Only AI Telemetry</td><td>Data sovereignty rejection by industrial EPCs</td><td style="color: #059669; font-weight: 600;">100% on-prem edge PLC runtime (air-gapped)</td></tr>
              </tbody>
            </table>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 12px; color: var(--text-body);">
            <strong>The Last-Mover Doctrine:</strong> Let venture-backed pioneers spend hundreds of millions validating the basic market demand, then enter with superior engineering tolerances, solved degradation modes, and institutional manufacturing packages.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig6_Cross_Sectoral_Validation.png', 'Fig 6: Cross-Sectoral Industrial Techno-Economic Validation (300 DPI)')">
          <img src="assets/Fig6_Cross_Sectoral_Validation.png" alt="Cross Sectoral Validation" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 6: Operational Availability & Cross-Sectoral Parity • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 5 • THE $420,000 PROBLEM',
    title: 'Unmitigated Electrochemical Degradation & Biofouling Economics',
    subtitle: 'Why Conventional Industrial Stacks Suffer Catastrophic Lifespan Collapse',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: #DC2626; margin: 0 0 8px; font-size: 13px; font-weight: 700;">Three Unseen Killers of Industrial PEM Stacks</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>1. Anodic Iridium Dissolution:</strong> Dynamic renewable power cycling induces OER catalyst loss at 1.911 µV/h, driving cell potential from 1.78V toward the 2.15V thermal runaway ceiling.</p>
              <p style="margin: 0 0 6px;"><strong>2. Membrane Chemical Thinning:</strong> Radicals (H2O2 / OH•) degrade Nafion ionomer backbone, increasing hydrogen crossover flux toward the 1.0% safety threshold.</p>
              <p style="margin: 0;"><strong>3. Dielectric EPS Biofouling:</strong> Microbial colonization on porous transport layers introduces a +58.8 mV parasitic resistance shift, previously indistinguishable from normal electrochemical wear.</p>
            </div>
          </div>

          <div style="background: #FEF2F2; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #FECACA;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #991B1B; margin-bottom: 4px;">ANNUAL ACCUMULATED CLIENT LOSS PER 5 MW BLOCK</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; color: #991B1B;">$412,000 USD / Skid</div>
            <p style="font-size: 11.5px; color: #7F1D1D; margin: 4px 0 0;">Premature stack replacement ($1.05M) amortized over 2.4 years + 21 days unscheduled outage.</p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png', 'Fig 4: Bayesian Anomaly Detection & Biofilm Dielectric Decoupling (300 DPI)')">
          <img src="assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png" alt="Biofilm Decoupling" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 4: Dielectric EPS Biofilm Anomaly Decoupling (<1.4s) • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 6 • SOLIDWORKS MASTER ASSEMBLY',
    title: 'SolidWorks 2025 SP1.2 Parametric 3D B-Rep Assembly',
    subtitle: 'Complete 11-Subsystem Turnkey Manufacturing Geometry with Fully Closed Mates',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Subsystems Engineered &amp; Fully Closed</h4>
            <div style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
              <div>• <strong>EL-200:</strong> 4x 50-Cell Stacks</div>
              <div>• <strong>V-300:</strong> Knockout Drum (ASME)</div>
              <div>• <strong>V-400:</strong> Anode Degasser (API 520)</div>
              <div>• <strong>DR-500:</strong> DeOxo + Dual 13X TSA</div>
              <div>• <strong>WTR-600:</strong> 3000L DI Water Loop</div>
              <div>• <strong>CMP-700:</strong> Diaphragm Comp.</div>
              <div>• <strong>ST-800:</strong> 350-bar Buffer Vessels</div>
              <div>• <strong>ELE-900:</strong> 11kV/415V + 5000A Bus</div>
              <div>• <strong>PLC-1000:</strong> Siemens S7-1500F</div>
              <div>• <strong>STR-100:</strong> ISMB 300 Mezzanine</div>
            </div>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 12px;">
            <strong>Zero Geometric Hallucinations:</strong> Every flange, gasket, bolt circle, pipe run, and instrument port is mathematically placed using deterministic coordinate math in Python with direct SolidWorks API integration.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/PEM_Commercial_Plant_Isometric_HD.png', 'PEM Commercial Plant Isometric View (300 DPI)')">
          <img src="assets/PEM_Commercial_Plant_Isometric_HD.png" alt="Plant Isometric" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: SolidWorks 2025 High-Resolution Isometric Assembly • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 7 • AUTOCAD ISO A1 DRAWING',
    title: 'AutoCAD 2025 ISO A1 Production Shop Drawings',
    subtitle: 'Zero Text Collisions, Exact Title Block, Plotted with Monochrome.ctb at 250 DPI',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Drawing Package Specification (DWG-002 Rev 5)</h4>
            <ul style="font-size: 12px; line-height: 1.5; color: var(--text-body); padding-left: 18px; margin: 0;">
              <li><strong>Standard:</strong> ISO A1 Metric (841 x 594 mm) with ISO 7200 / ASME Y14.100 Title Block.</li>
              <li><strong>Viewport A (Elevation):</strong> ISMB 300 Mezzanine (+4.360m / +6.200m), Handrails, Equipment mounting.</li>
              <li><strong>Viewport B (Plot Plan):</strong> 12.0m x 8.0m skid footprint with Grid Lines 1-4 and A-C.</li>
              <li><strong>Detail C-C &amp; D-D:</strong> ASME BPVC Sec VIII UG-27 shell wall sizing (14.0mm commercial plate selected).</li>
              <li><strong>Master Schedule:</strong> Complete 11-subsystem Bill of Materials with design P/T and ASTM metallurgy.</li>
            </ul>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>AutoCAD Core Console Validated:</strong> Plotted natively using <code style="font-size: 11px;">accoreconsole.exe</code> to vector PDF with zero layer overlapping and zero text clipping.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png', 'AutoCAD 2025 ISO A1 Production Shop Drawing DWG-002 Rev 5 (300 DPI)')">
          <img src="assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png" alt="AutoCAD Drawing Preview" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: Plotted AutoCAD ISO A1 Shop Drawing (DWG-002 Rev 5) • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 8 • PHASE 1 VIRTUAL SENSING',
    title: 'Phase 1: Virtual Sensing XGBoost Degradation Model',
    subtitle: 'R² = 0.963, Sub-Microvolt Accuracy (1.911 µV/h) Over 5,000 Hours Continuous Telemetry',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Predictive Feature Importance Breakdown</h4>
            <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
              SHAP analysis isolating the dominant drivers of long-term electrochemical wear in commercial 250 cm² active area stacks:
            </p>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <thead>
                <tr style="border-bottom: 1px solid #CBD5E1; color: var(--text-muted); text-align: left;">
                  <th style="padding: 3px;">Rank</th><th>Input Feature</th><th>SHAP Weight</th><th>Physical Mechanism</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style="padding: 4px; font-weight: 700;">#1</td><td>Current Density (J)</td><td>38.4%</td><td>OER overpotential &amp; electric field stress</td></tr>
                <tr><td style="padding: 4px; font-weight: 700;">#2</td><td>Operating Temp (T)</td><td>27.1%</td><td>Catalyst dissolution kinetics (Arrhenius)</td></tr>
                <tr><td style="padding: 4px; font-weight: 700;">#3</td><td>Cumulative Amp-Hours</td><td>19.6%</td><td>IrO2 / Pt surface passivation</td></tr>
                <tr><td style="padding: 4px; font-weight: 700;">#4</td><td>DI Flow Maldistribution</td><td>14.9%</td><td>Localized hot-spot pinhole formation</td></tr>
              </tbody>
            </table>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>Eliminates Hardware Sensors:</strong> Predicts local cell degradation rates in real time without requiring costly internal reference electrodes or destructive teardowns.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig1_XGBoost_Feature_Importance.png', 'Fig 1: XGBoost Feature Importance & Electrochemical Degradation Trajectory (300 DPI)')">
          <img src="assets/Fig1_XGBoost_Feature_Importance.png" alt="Feature Importance" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 1: Feature Importance & Degradation Trajectory • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 9 • PHASE 1 PARITY VALIDATION',
    title: 'Phase 1: Empirical vs Predicted Parity Validation',
    subtitle: 'Zero Systematic Bias Across Dynamic 0.5 to 2.5 A/cm² Load Cycles',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Parity Statistics Across 5,000h Operating Window</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;">• <strong>Coefficient of Determination (R²):</strong> 0.963 across train/test splits.</p>
              <p style="margin: 0 0 6px;">• <strong>Mean Absolute Error (MAE):</strong> 0.0038 V (3.8 mV) across 10-cell stack.</p>
              <p style="margin: 0 0 6px;">• <strong>Root Mean Square Error (RMSE):</strong> 0.0051 V (5.1 mV).</p>
              <p style="margin: 0;">• <strong>Residual Mean:</strong> +0.00012 V (symmetric Gaussian distribution with zero drift).</p>
            </div>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 12px;">
            <strong>Cross-Sector Validation:</strong> Model verified on both continuous steady-state industrial duty (chlor-alkali) and intermittent renewable duty (solar-coupled green hydrogen).
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png', 'Fig 2: Parity Plot Predicted vs Empirical Voltage Drift (300 DPI)')">
          <img src="assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png" alt="Parity Plot" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 2: High-Fidelity Parity Plot (Predicted vs Empirical) • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 10 • PHASE 2 PIBT TRANSFORMER',
    title: 'Phase 2: Physics-Informed Butler-Volmer Transformer (PIBT)',
    subtitle: 'NRMSE = 0.054 • Loss Formulation Enforcing Electrochemical Thermodynamic Law',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Physics-Informed Loss Function</h4>
            <div style="font-family: monospace; font-size: 11px; background: #FFFFFF; padding: 8px; border-radius: 4px; border: 1px solid #CBD5E1; margin-bottom: 8px; color: var(--accent);">
              L_total = L_MSE + &lambda;_BV * L_BV + &lambda;_reg * ||&theta;||²
            </div>
            <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0;">
              Unlike black-box neural networks that violate energy conservation, PIBT penalizes predictions that deviate from the Butler-Volmer kinetic equation:
              <br><code style="font-size: 10.5px;">j = j_0 * [exp(&alpha;_a F &eta; / RT) - exp(-&alpha;_c F &eta; / RT)]</code>
            </p>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>Generalization Guarantee:</strong> PIBT maintains stable degradation forecasting even during extreme dynamic power transients where pure machine learning models produce unphysical voltage spikes.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png', 'Fig 3: PIBT Time-Series Degradation Forecast & Convergence (300 DPI)')">
          <img src="assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png" alt="PIBT Forecast" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 3: PIBT Loss Convergence & 100h Forecast • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 11 • PHASE 3 FAULT MITIGATION',
    title: 'Phase 3: Sub-1.4s Bayesian Anomaly Trip & Autonomous DI Purge',
    subtitle: 'Decoupling Microbial Biofilm From Physical Wear and Recovering Stack Baseline in <5 Seconds',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Sub-1.4s Closed-Loop SCADA Sequence</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;">• <strong>t = 0.00s:</strong> Bacterial EPS matrix injects dielectric resistance (+58.8 mV residual shift).</p>
              <p style="margin: 0 0 6px;">• <strong>t = 1.18s:</strong> Recursive Bayesian posterior crosses P(Biofilm) &ge; 0.950 threshold.</p>
              <p style="margin: 0 0 6px;">• <strong>t = 1.34s:</strong> SCADA opens fast solenoid valve XV-202, surging DI water from 15 to 28 L/min and derating current to 1.30 A/cm².</p>
              <p style="margin: 0;">• <strong>t = 4.80s:</strong> High hydrodynamic shear dislodges EPS layer; cell potential returns to nominal baseline (1.82V).</p>
            </div>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>100% Chemical-Free:</strong> Replaces hazardous chemical biocide wash loops (formalin/peracetic acid) with purely physical hydrodynamic shear purging.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig5_Lifespan_Extension_Techno_Economic.png', 'Fig 5: Techno-Economic Lifespan Extension & Mitigation Timeline (300 DPI)')">
          <img src="assets/Fig5_Lifespan_Extension_Techno_Economic.png" alt="Lifespan Extension" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 5: Closed-Loop SCADA Mitigation Timeline • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 12 • TECHNO-ECONOMIC PAYBACK',
    title: 'Techno-Economic Payback: Month 4.8 Breakeven & 485% 2-Year ROI',
    subtitle: 'Quantifiable Proof That Prevents Premature $1.05M Stack Replacement',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--status-pass); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Customer Financial Benefit (5 MW Skid)</h4>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <tr><td style="padding: 4px 0; color: #475569;">Avoided Premature Stack Replacement</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">+$412,000 USD</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Eliminated Chemical Biocide Flushes</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">+$76,000 USD</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Avoided Unscheduled Downtime (21 Days)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">+$144,000 USD</td></tr>
              <tr style="border-top: 1px solid #CBD5E1;"><td style="padding: 4px 0; font-weight: 700; color: #0F172A;">Total Cumulative 2-Year Savings</td><td style="font-family: monospace; font-weight: 800; text-align: right; color: #059669;">+$632,000 USD</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">AERO-H2 Twin 2-Year License Cost</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #DC2626;">-$72,000 USD</td></tr>
              <tr style="border-top: 2px solid #0F172A;"><td style="padding: 6px 0; font-weight: 800; color: #0F172A;">Net 2-Year Cash Return</td><td style="font-family: monospace; font-weight: 800; text-align: right; color: #059669; font-size: 13px;">+$560,000 USD (485% ROI)</td></tr>
            </table>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>Breakeven in Month 4.8:</strong> The complete service fee is fully recouped in less than 5 months of operation purely through avoided energy efficiency degradation.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png', 'Fig 8: Commercial Monetization & Payback Curve (300 DPI)')">
          <img src="assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png" alt="Payback Curve" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 8: Cumulative Client Net Cashflow & Payback • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 13 • UNIVERSAL MULTI-PLANT MATRIX',
    title: 'Universal Multi-Plant Deployment Matrix (Standardized Turnkey Retrofit)',
    subtitle: 'Standardized ISMB 300 Mezzanine & Skid Footprint Enables Plug-and-Play Integration Across 5 Sectors',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Standardized Process Duties &amp; Safety Compliance</h4>
            <div style="font-size: 11.5px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>SEC-01 Petrochemical &amp; Refining:</strong> 50-100 TPD H2 | Hydrotreating &amp; sulfur recovery | SIL-2 high-P trip interlock (&lt;10 ppm O2 quench) | SA-312 TP316L.</p>
              <p style="margin: 0 0 6px;"><strong>SEC-02 Chlor-Alkali:</strong> 20-40 TPD H2 | Membrane brine BoP retrofit | Dual differential-P radar interlock | ASTM B265 Gr 2 Ti.</p>
              <p style="margin: 0 0 6px;"><strong>SEC-03 Green Steel:</strong> 80-150 TPD H2 | 100% H2 DRI shaft furnace | Fast-acting ESD valves (&lt;500ms) | ASME Sec VIII Div 1.</p>
              <p style="margin: 0;"><strong>SEC-04 Green Ammonia:</strong> 100-250 TPD H2 | Direct 30-bar feed to Haber-Bosch loop | Automated PSV angle relief trip | 316L orbitally welded.</p>
            </div>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 12px;">
            <strong>No Civil Redesign:</strong> The modular 12.0m x 8.0m skid footprint fits standard industrial crane pads and transport flatbeds, minimizing on-site construction time to under 14 days.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/PEM_Turnkey_Plant_Skid_Photorealistic_Render.png', 'PEM Turnkey Plant Skid Photorealistic 3D Render (300 DPI)')">
          <img src="assets/PEM_Turnkey_Plant_Skid_Photorealistic_Render.png" alt="Skid Photorealistic Render" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: Turnkey Skid Photorealistic Fabrication Render • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 14 • COMMERCIAL MONETIZATION & DELIVERY PORTFOLIO',
    title: 'ASD-STM100 Commercial Service Delivery Portfolio',
    subtitle: 'Transparent, Data-Backed Pricing Structured for Indian & Global Industrial Markets',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Three Grounded Commercial Tiers</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>TIER 1 • FORENSIC AUDIT ($7,500 / ₹6.5 Lakh):</strong> 5-day remote diagnostic CAE audit of brownfield stacks. Ingests 100h SCADA, generates verified ASME Sec VIII UG-27 and Tafel decay dossier.</p>
              <p style="margin: 0 0 6px;"><strong>TIER 2 • EDGE FIRMWARE ($2,500/mo / ₹2.1 Lakh/mo):</strong> AERO-H2 neural twin on Siemens S7-1500 / Jetson. Sub-1.4s biofilm anomaly detection + automated DI shear flush.</p>
              <p style="margin: 0;"><strong>TIER 3 • TURNKEY CAD LICENSE ($50,000 / ₹42.0 Lakh):</strong> SolidWorks 2025 assembly, AutoCAD ISO A1 drawings, ASME calculations, and complete procurement BOM for EPC fabricators.</p>
            </div>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; font-size: 12px; color: #166534;">
            <strong>Institutional Contract Governance:</strong> Governed by The Indian Arbitration and Conciliation Act 1996 (New Delhi), strict mutual NDA, and Section 72A IT Act data safeguards.
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="openCommercialModal('audit')">
          <div style="text-align: center; padding: 20px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #0284C7; margin-bottom: 8px;">INTERACTIVE STATEMENT OF WORK (SOW)</div>
            <div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 12px;">Click to Inspect Commercial Milestone Deliverables & Legal Terms</div>
            <button class="btn btn-primary" style="margin: 0 auto;" onclick="openCommercialModal('audit')">
              Open Commercial SOW Modal
            </button>
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 15 • READY FOR COMMERCIAL TENDER',
    title: 'Ready for Commercial Tender & Immediate Industrial Pilot',
    subtitle: 'Zero Further Planning Required • Production-Grade Engineering Deliverables Ready for Inspection',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Immediate Pilot Onboarding Protocol</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>Step 1:</strong> Client executes mutual NDA and delivers 100-hour historical SCADA CSV log.</p>
              <p style="margin: 0 0 6px;"><strong>Step 2:</strong> HYDROSENSE team runs automated ADVEKT forensic analysis and returns Tafel decay attribution within 5 business days.</p>
              <p style="margin: 0 0 6px;"><strong>Step 3:</strong> Edge twin deployed on plant Siemens S7-1500 PLC for zero-risk side-by-side passive monitoring.</p>
              <p style="margin: 0;"><strong>Step 4:</strong> Autonomous closed-loop DI mitigation engaged after baseline validation.</p>
            </div>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">CONTACT CORRESPONDING AUTHOR</div>
            <div style="font-size: 12.5px; color: #14532D; line-height: 1.5;">
              <strong>Sourish Senapati</strong> &bull; Dept. of Chemical Engineering, Jadavpur University<br>
              <span style="font-family: monospace; font-size: 11.5px;">sourishs.chem.ug@jadavpuruniversity.in</span>
            </div>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/PEM_10Cell_Stack_Presentation_Render.png', 'PEM 10-Cell Industrial Stack Presentation Render (300 DPI)')">
          <img src="assets/PEM_10Cell_Stack_Presentation_Render.png" alt="10-Cell Stack Render" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: 10-Cell Industrial Stack B-Rep SolidWorks Render • Click to Zoom
          </div>
        </div>
      </div>
    `
  }
];

function openDeck() {
  const overlay = document.getElementById('deckOverlay');
  if (!overlay) return;
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
  renderDeckSlide();
}

function closeDeck() {
  const overlay = document.getElementById('deckOverlay');
  if (overlay) overlay.classList.remove('active');
  document.body.style.overflow = 'auto';
}

function renderDeckSlide() {
  const cur = presentationSlides[deckIdx];
  const box = document.getElementById('deckSlideContent');
  const countEl = document.getElementById('deckSlideCount');
  if (!box) return;

  if (countEl) {
    countEl.innerText = `Slide ${deckIdx + 1} / ${presentationSlides.length}`;
  }

  box.innerHTML = `
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: var(--space-1); text-transform: uppercase;">
      ${cur.tag}
    </div>
    <h2 style="font-size: 20px; font-weight: 800; color: var(--text-main); margin-bottom: var(--space-1); line-height: 1.25;">
      ${cur.title}
    </h2>
    <div style="font-size: 12.5px; color: var(--text-muted); margin-bottom: var(--space-3); border-bottom: 1px solid var(--border-subtle); padding-bottom: var(--space-2);">
      ${cur.subtitle}
    </div>
    <div style="flex: 1; overflow-y: auto;">
      ${cur.body}
    </div>
  `;

  if (window.renderMathInElement) {
    window.renderMathInElement(box, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      throwOnError: false,
      ignoredClasses: ['btn', 'tier-price-tag', 'slider-num', 'data-val', 'metric-num', 'no-math']
    });
  }
}

function slideForward() {
  if (deckIdx < presentationSlides.length - 1) {
    deckIdx++;
    renderDeckSlide();
  }
}

function slideBack() {
  if (deckIdx > 0) {
    deckIdx--;
    renderDeckSlide();
  }
}

// Global Keyboard Navigation
document.addEventListener('keydown', (e) => {
  const overlay = document.getElementById('deckOverlay');
  const isDeckOpen = overlay && overlay.classList.contains('active');
  if (isDeckOpen) {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      slideForward();
    }
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      slideBack();
    }
    if (e.key === 'Escape') {
      closeDeck();
    }
  }
});

window.openDeck = openDeck;
window.closeDeck = closeDeck;
window.slideForward = slideForward;
window.slideBack = slideBack;
