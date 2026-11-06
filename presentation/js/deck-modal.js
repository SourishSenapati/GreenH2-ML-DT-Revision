/**
 * HYDROSENSE™ 20-Slide Master Presentation Deck Modal
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
      <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 6px;">AUTHORS &amp; INSTITUTIONAL AFFILIATIONS</div>
            <p style="font-size: 12.5px; line-height: 1.5; margin: 0 0 6px;"><strong>Sourish Senapati</strong> (Corresponding Author)<br>Dept. of Chemical Engineering, Jadavpur University<br><span style="font-family: monospace; font-size: 11px; color: var(--text-muted);">sourishs.chem.ug@jadavpuruniversity.in</span></p>
            <p style="font-size: 12.5px; line-height: 1.5; margin: 0;"><strong>Srestha Biswash</strong><br>Dept. of Microbiology &amp; Biotechnology, Sister Nivedita University</p>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 6px;">CORE QUANTITATIVE BENCHMARKS</div>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <tr><td style="padding: 3px 0; color: #475569;">Virtual Sensing XGBoost</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">R² = 0.963 (1.911 µV/h error)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Physics-Informed Transformer</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">NRMSE = 0.054 (100h horizon)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Bayesian Anomaly Trip Latency</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">&lt; 1.40 seconds (F1 = 0.990)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Core Stack Lifespan Extension</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #166534;">+39.4% (21,428h &rarr; 29,879h)</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Levelized Cost of H2 (LCOH)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #166534;">-$0.42 / kg H2 savings</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Customer Payback Horizon</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #166534;">Month 4.8 (485% 2-Yr ROI)</td></tr>
            </table>
          </div>

          <div style="background: var(--bg-wash); padding: 10px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); font-size: 11.5px; color: var(--text-body);">
            <strong>Governing Electrochemical Model:</strong>
            <div style="font-family: monospace; font-size: 11px; margin-top: 4px; color: var(--accent);">
              V_cell = E_rev(T, P) + &eta;_act(J, T) + &eta;_ohm(J, T) + &eta;_conc(J) + &Delta;&eta;_biofilm
            </div>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px; cursor: pointer;" onclick="showModal('assets/PEM_Commercial_Plant_Master_Render.png', '100 TPD Turnkey Commercial Plant Master Assembly (300 DPI)')">
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
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Beachhead Market: Captive Industrial Chlor-Alkali &amp; Refineries</h4>
            <p style="font-size: 12.5px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
              Instead of competing in undifferentiated greenfield electrolyzer auctions, HYDROSENSE locks in existing high-purity brownfield industrial operators in India (Haldia Petrochemicals, HMEL, Grasim, DCM Shriram).
            </p>
            <ul style="font-size: 12px; line-height: 1.5; color: var(--text-body); padding-left: 18px; margin: 0;">
              <li><strong>Zero Substation CAPEX:</strong> Taps directly into existing 5000A DC busbars.</li>
              <li><strong>Immediate Offtake:</strong> On-site hydrotreating &amp; chlorine balance consumption.</li>
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
                <tr><td style="padding: 4px; font-weight: 700;">Niche</td><td>Chlor-Alkali &amp; Refining</td><td>20-50 TPD H2</td><td>$42M (India)</td></tr>
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
              <p style="margin: 0;"><strong>4. Dual-Revenue Co-Products:</strong> Integrated AERO-PURE Unit 650 monetizing ASTM D1193 ultrapure water and waste heat.</p>
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

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--text-main); margin: 0 0 6px; font-size: 12px; font-weight: 700;">The Last-Mover Arbitrage Play</h4>
            <p style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); margin: 0;">
              First movers burned billions validating membrane chemistry. We build the final durable operating layer that eliminates early failures and scales across legacy facilities.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig6_Cross_Sectoral_Validation.png', 'Fig 6: Cross-Sectoral Validation Across Scales (300 DPI)')">
          <img src="assets/Fig6_Cross_Sectoral_Validation.png" alt="Cross Sectoral Validation" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 6: Validation Across Benchtop &amp; Industrial Cells • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 5 • THE $420,000 DEGRADATION PROBLEM',
    title: 'The Commercial Barrier: Levelized Cost of Hydrogen (LCOH)',
    subtitle: 'Accelerated Catalyst Ripening & Unscheduled Plant Outages',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="background: var(--bg-wash); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 8px;">CAPEX &amp; PRECIOUS METALS</div>
          <h4 style="font-size: 15px; font-weight: 700; color: var(--text-main); margin: 0 0 10px;">Noble Metal Concentration</h4>
          <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
            Stack CAPEX represents <strong>40-55%</strong> of total plant installed capital cost.
          </p>
          <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
            <li>Platinum (Pt) at cathode (HER)</li>
            <li>Iridium (IrO2) at anode (OER)</li>
            <li>Dynamic power cycling accelerates anodic Iridium dissolution at 1.911 µV/h.</li>
          </ul>
        </div>

        <div style="background: var(--bg-wash); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #D97706; margin-bottom: 8px;">DYNAMIC RENEWABLE CYCLING</div>
          <h4 style="font-size: 15px; font-weight: 700; color: var(--text-main); margin: 0 0 10px;">Intermittent Power Shocks</h4>
          <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
            Solar &amp; wind intermittency triggers rapid load steps (0 to 100% in seconds).
          </p>
          <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
            <li>Severe local potential spikes (>2.2V)</li>
            <li>Ostwald ripening &amp; Pt band precipitation</li>
            <li>Premature stack condemnation before 22,000h design life</li>
          </ul>
        </div>

        <div style="background: #FEF2F2; padding: 16px; border-radius: var(--radius-md); border: 1px solid #FECACA; display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #DC2626; margin-bottom: 8px;">COMMERCIAL COST PENALTY</div>
          <h4 style="font-size: 15px; font-weight: 700; color: #991B1B; margin: 0 0 10px;">The $420k Unmitigated Replacement</h4>
          <p style="font-size: 12px; line-height: 1.5; color: #7F1D1D; margin: 0 0 8px;">
            Unscheduled stack replacement forces a $420,000 CAPEX write-off on a 5 MW plant.
          </p>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid #FECACA; margin-top: auto;">
            <div style="font-size: 11px; color: #991B1B; font-weight: 700;">HYDROSENSE Impact:</div>
            <div style="font-size: 12px; color: #166534; font-weight: 700; margin-top: 2px;">Extends stack life to 29,879h (+39.4%), defers CAPEX by 1.15 years.</div>
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 6 • DUAL DEGRADATION PHYSICS',
    title: 'Dual Degradation Physics: Nanoscale Ripening vs. Dielectric Biofilm',
    subtitle: 'Deconvoluting Irreversible Electrochemical Losses from Reversible Biological Obstructions',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="background: var(--bg-wash); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 8px;">KINETIC LOSS MODE 1</div>
          <h3 style="font-size: 16px; font-weight: 700; color: var(--text-main); margin: 0 0 8px;">Electrochemical Catalyst Ripening &amp; Dissolution</h3>
          <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
            Under prolonged high-potential operation (>1.85V vs RHE), anodic iridium oxide (IrO2) undergoes active dissolution into aqueous Ir(III)/Ir(IV) complexes:
          </p>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 11px; color: var(--accent); margin-bottom: 8px;">
            IrO2 + 4H+ + 2e- &rarr; Ir2+ + 2H2O (Tafel Slope b = 42.1 &rarr; 68.4 mV/dec)
          </div>
          <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
            <li>ECSA contracts from 48.2 m²/g to 26.1 m²/g.</li>
            <li>Irreversible voltage shift requiring gradual current redistribution.</li>
          </ul>
        </div>

        <div style="background: var(--bg-wash); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #059669; margin-bottom: 8px;">OHMIC LOSS MODE 2</div>
          <h3 style="font-size: 16px; font-weight: 700; color: var(--text-main); margin: 0 0 8px;">Biological Biofilm (EPS) Ohmic Fouling</h3>
          <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
            Oligotrophic colonizers (*Pseudomonas aeruginosa*) adhere to Titanium Porous Transport Layers (PTL), secreting extracellular polymeric substances (EPS):
          </p>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 11px; color: #059669; margin-bottom: 8px;">
            &Delta;&eta;_biofilm = J &times; (R_EPS + R_contact) &asymp; +58.8 mV step jump
          </div>
          <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
            <li>Creates dielectric insulating barrier across microporous channels.</li>
            <li><strong>Reversible:</strong> Fully stripped via automated DI water shear pulse.</li>
          </ul>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 7 • PHASE 1 VIRTUAL SENSING XGBOOST',
    title: 'Phase 1 Feature Gain: Reverse-Engineering Nanoscale Morphology',
    subtitle: 'Non-Invasive Virtual Sensing Without Destructive Cell Probes',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Materials Informatics Feature Hierarchy</h4>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <thead>
                <tr style="border-bottom: 1px solid #CBD5E1; color: var(--text-muted); text-align: left;">
                  <th style="padding: 4px;">Latent Feature</th><th>Gain</th><th>Physical Mechanism</th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom: 1px solid #E2E8F0;"><td style="padding: 5px 4px; font-weight: 700;">Tafel Slope (b)</td><td style="font-family: monospace; color: var(--accent);">38.2%</td><td>OER charge-transfer kinetics</td></tr>
                <tr style="border-bottom: 1px solid #E2E8F0;"><td style="padding: 5px 4px; font-weight: 700;">ECSA Decay (m²/g)</td><td style="font-family: monospace; color: var(--accent);">21.5%</td><td>Nanoscale active site area loss</td></tr>
                <tr style="border-bottom: 1px solid #E2E8F0;"><td style="padding: 5px 4px; font-weight: 700;">Porosity Index (&epsilon;)</td><td style="font-family: monospace; color: var(--accent);">14.7%</td><td>PTL mass transport resistance</td></tr>
                <tr><td style="padding: 5px 4px; font-weight: 700;">Macroscopic Sensors</td><td style="font-family: monospace; color: var(--text-muted);">25.6%</td><td>Current density, flow, inlet temp</td></tr>
              </tbody>
            </table>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">GPU-RESIDENT INFERENCE</div>
            <p style="font-size: 12px; line-height: 1.5; color: #14532D; margin: 0;">
              Over <strong>74% of model gain</strong> originates from latent electrochemical features. Inference runs in &lt;0.4 ms on standard PLC edge accelerators.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig1_XGBoost_Feature_Importance.png', 'Fig 1: XGBoost Feature Importance & Gain Hierarchy (300 DPI)')">
          <img src="assets/Fig1_XGBoost_Feature_Importance.png" alt="Feature Importance" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 1: Relative Feature Importance via Tree SHAP • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 8 • PHASE 1 PARITY VALIDATION',
    title: 'Sub-Microvolt Parity Plot: Predicted vs Empirical Degradation',
    subtitle: 'Strict 1:1 Unity Alignment with Zero Systematic Bias (R² = 0.963, 1.911 µV/h)',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Parity Verification Metrics</h4>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <tr><td style="padding: 4px 0; color: #475569;">Coefficient of Determination (R²)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #059669;">0.963</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Root Mean Squared Error (RMSE)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">1.911 µV/h</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Mean Absolute Percentage Error</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">2.14%</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Residual Distribution</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">Zero-centered Gaussian</td></tr>
              <tr><td style="padding: 4px 0; color: #475569;">Heteroscedasticity Score</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #059669;">p = 0.42 (Homoscedastic)</td></tr>
            </table>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--text-main); margin: 0 0 6px; font-size: 12px; font-weight: 700;">Zero Systematic Bias Confirmed</h4>
            <p style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); margin: 0;">
              Points track the 45° dashed unity line identically across both low-wear and severe dynamic degradation regimes, validating model transferability.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png', 'Fig 2: Parity Plot Predicted vs Empirical Catalyst Decay (300 DPI)')">
          <img src="assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png" alt="Parity Plot" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 2: 1:1 Parity Alignment Across 1,200 Operating Points • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 9 • PHASE 2 PIBT TRANSFORMER',
    title: 'Long-Horizon Prognostics: Physics-Informed Transformer (PIBT)',
    subtitle: 'Thermodynamic Bounding via Butler-Volmer Attention Mechanism (NRMSE = 0.054)',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Thermodynamic Loss Penalty</h4>
            <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
              Standard LSTM/GRU models drift into unphysical predictions over 100h+ forecasting windows. PIBT penalizes predictions violating Butler-Volmer kinetics:
            </p>
            <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: var(--accent); margin-bottom: 8px;">
              L_total = L_MSE + &lambda; &times; || V_pred - [E_rev + &eta;_act + &eta;_ohm] ||²
            </div>
            <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
              <li>Maintains NRMSE = 0.054 across a 100-hour rolling horizon.</li>
              <li>Monte Carlo Dropout generates rigorous 95% Bayesian confidence bounds.</li>
            </ul>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">PREVENTING GRADIENT DRIFT</div>
            <p style="font-size: 12px; line-height: 1.5; color: #14532D; margin: 0;">
              Eliminates the catastrophic false alarms and runaway predictions common in unconstrained black-box models.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png', 'Fig 3: PIBT Long-Horizon Time Series Forecast (300 DPI)')">
          <img src="assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png" alt="PIBT Time Series" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 3: 100h Degradation Forecast with 95% Confidence Band • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 10 • PHASE 3 FAULT MITIGATION',
    title: 'Real-Time Biofilm Isolation: Decoupled Bayesian Classifier',
    subtitle: 'Sub-1.40s Anomaly Trip Latency & Elimination of Chemical Biocides (F1 = 0.990)',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Dynamic Bayesian Posterior</h4>
            <p style="font-size: 12px; line-height: 1.5; color: var(--text-body); margin: 0 0 8px;">
              Continuously computes the posterior probability of biological fouling versus physical catalyst degradation:
            </p>
            <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: var(--accent); margin-bottom: 8px;">
              P(Biofilm | e_t) = [P(e_t | Biofilm) &times; P(Biofilm)] / P(e_t)
            </div>
            <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-muted); padding-left: 16px; margin: 0;">
              <li>When P(Biofilm) &ge; 0.95, trips autonomous high-shear DI water solenoid.</li>
              <li>Trip latency verified at <strong>&lt; 1.40 seconds</strong>.</li>
              <li>F1-Score: <strong>0.990</strong> (Zero false positive membrane shutdowns).</li>
            </ul>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">ELIMINATING CHEMICAL BIOCIDES</div>
            <p style="font-size: 12px; line-height: 1.5; color: #14532D; margin: 0;">
              Eliminates toxic chemical biocides (glutaraldehyde, sodium hypochlorite) that poison ionomer membranes and violate environmental effluent regulations.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png', 'Fig 4: Bayesian Anomaly Detection of EPS Biofilm (300 DPI)')">
          <img src="assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png" alt="Bayesian Biofilm Isolation" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 4: Dual-Panel Anomaly Detection &amp; SCADA Trip Trigger • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 11 • CLOSED-LOOP MITIGATION TIMELINE',
    title: 'Closed-Loop Remediation & +39.4% Lifespan Extension',
    subtitle: 'Automated High-Shear DI Surge Restoring Normal Cell Potential (+8,451 Hours)',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">In-Operando Mitigation Sequence</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>t = 0.0s:</strong> Anomaly detected; voltage residual jumps +58.8 mV.</p>
              <p style="margin: 0 0 6px;"><strong>t = 1.4s:</strong> Bayesian threshold crossed (P &ge; 0.95); SCADA opens high-shear flush valve FV-601.</p>
              <p style="margin: 0 0 6px;"><strong>t = 4.8s:</strong> Pure DI surge sweeps PTL channel surface, shearing EPS matrix.</p>
              <p style="margin: 0;"><strong>t = 8.2s:</strong> Overpotential drops back to baseline 1.82V; stack returned to full duty.</p>
            </div>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">OPERATIONAL LIFESPAN EXTENSION</div>
            <p style="font-size: 12.5px; line-height: 1.5; color: #14532D; margin: 0;">
              Extends stack life from <strong>21,428h to 29,879h</strong> (+39.4% increase, +8,451 operating hours), saving $420,000 in replacement costs.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig5_Lifespan_Extension_Techno_Economic.png', 'Fig 5: Techno-Economic Lifespan Extension Comparison (300 DPI)')">
          <img src="assets/Fig5_Lifespan_Extension_Techno_Economic.png" alt="Lifespan Extension" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 5: Unmitigated Run-to-Failure vs. Closed-Loop Mitigation • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 12 • TECHNO-ECONOMIC PAYBACK & CUSTOMER ROI',
    title: 'Customer ROI Payback Model: Month 4.8 Breakeven',
    subtitle: '485% 2-Year ROI on 5 MW Commercial Skid ($560,000 Net Benefit)',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Financial Cash-Flow Breakdown (5 MW Skid)</h4>
            <table style="width: 100%; font-size: 11.5px; border-collapse: collapse;">
              <tr><td style="padding: 3px 0; color: #475569;">Avoided Stack Replacement CAPEX</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #059669;">+$412,000</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Eliminated Chemical Biocide Flushes</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #059669;">+$76,000</td></tr>
              <tr><td style="padding: 3px 0; color: #475569;">Avoided Unscheduled Plant Outage (21d)</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #059669;">+$144,000</td></tr>
              <tr style="border-top: 1px solid #CBD5E1;"><td style="padding: 4px 0; font-weight: 700; color: #0F172A;">Gross 2-Year Cash Savings</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #0F172A;">+$632,000</td></tr>
              <tr><td style="padding: 3px 0; color: #DC2626;">AERO-H2 Edge Twin 2-Year License</td><td style="font-family: monospace; font-weight: 700; text-align: right; color: #DC2626;">-$72,000</td></tr>
              <tr style="border-top: 2px solid #059669;"><td style="padding: 4px 0; font-weight: 800; color: #166534;">Net 2-Year Client Benefit</td><td style="font-family: monospace; font-weight: 800; text-align: right; color: #166534;">+$560,000</td></tr>
            </table>
          </div>

          <div style="background: #F0FDF4; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">RAPID PAYBACK HORIZON</div>
            <p style="font-size: 12px; line-height: 1.5; color: #14532D; margin: 0;">
              Breakeven is reached in <strong>Month 4.8</strong>. Over 2 years of commercial duty, the plant yields a <strong>485% ROI</strong>.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png', 'Fig 8: Commercial Payback Curve & Service Matrix (300 DPI)')">
          <img src="assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png" alt="ROI Payback Curve" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 8: Audited Customer ROI Payback Curve • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 13 • MULTI-PLANT DFM MATRIX',
    title: 'Universal Multi-Plant Deployment Matrix Across 5 Sectors',
    subtitle: 'Standardized ISMB 300 Mezzanine Skid Footprint for Plug-and-Play Industrial Retrofit',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1.15fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Standardized DFM Specifications</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>Bipolar Plate Stamping:</strong> Progressive die stamping of ASTM B265 Grade 2 titanium sheet (0.80mm). Draft angle 8.5°, 16.36% localized thinning (&lt;18% safe limit).</p>
              <p style="margin: 0 0 6px;"><strong>Pressure Vessel Sizing:</strong> Knockout Drum V-300 sized via ASME Sec VIII Div 1 UG-27 (14.0mm SA-240 316L plate selected vs 12.1mm required, 45.86 bar hydrotest).</p>
              <p style="margin: 0;"><strong>Universal Footprint:</strong> 12.0m x 8.0m standardized skid layout fits existing industrial civil slabs without foundation redesign.</p>
            </div>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--text-main); margin: 0 0 6px; font-size: 12px; font-weight: 700;">Multi-Industry Interoperability</h4>
            <p style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); margin: 0;">
              Retrofit configurations verified across Refineries, Chlor-Alkali, Green Steel, Fertilizers, and Heavy Mobility.
            </p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png', 'Fig 7: DFM Progressive Stamping & Multi-Plant Matrix (300 DPI)')">
          <img src="assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png" alt="DFM Multi-Plant Matrix" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig 7: DFM Stamping Tooling &amp; Multi-Plant Matrix • Click to Zoom
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 14 • SOLIDWORKS 2025 MASTER ASSEMBLY',
    title: '100 TPD Commercial Plant Mechanical & Process Architecture',
    subtitle: 'Native SolidWorks 2025 SP1.2 B-Rep Assembly & STEP AP214 Models',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/PEM_Commercial_Plant_Isometric_HD.png', 'SolidWorks 2025 100 TPD Commercial Plant Isometric View (300 DPI)')">
          <img src="assets/PEM_Commercial_Plant_Isometric_HD.png" alt="Isometric Master Assembly" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: SolidWorks 2025 Master Plant Assembly (Isometric) • Click to Zoom
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: var(--space-2); overflow-y: auto;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 2px;">11 PRODUCTION SUBSYSTEMS (ISA-5.1 TAGS)</div>
          <table style="width: 100%; font-size: 10.5px; border-collapse: collapse;">
            <thead>
              <tr style="border-bottom: 1px solid #CBD5E1; color: var(--text-muted); text-align: left;">
                <th style="padding: 3px;">Tag</th><th>Subsystem</th><th>Specs</th><th>Standard</th>
              </tr>
            </thead>
            <tbody>
              <tr><td style="padding: 3px; font-weight: 700;">STR-100</td><td>Mezzanine Frame</td><td>12x8x5.8m ISMB 300</td><td>IS 800 / AISC</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">EL-200</td><td>Electrolyzer Bay</td><td>4x 50-cell, 5000A DC</td><td>ISO 22734</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">V-300</td><td>HP Knockout Drum</td><td>30 bar, demister pad</td><td>ASME Sec VIII</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">V-400</td><td>Anode O2 Degasser</td><td>6.2m vent stack</td><td>API 520 / 521</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">DR-500</td><td>Catalytic DeOxo</td><td>&lt;0.5 ppm O2, -70°C Td</td><td>ISO 14687 Gr D</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">WTR-600</td><td>DI Water Treatment</td><td>3000L, duplex pumps</td><td>ASTM D1193</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">CMP-700</td><td>HP Compressor</td><td>3-stage metal diaphragm</td><td>API 618 (350 bar)</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">ST-800</td><td>Buffer Storage</td><td>4x 350-bar cascade</td><td>ASME Sec VIII Div 3</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">ELE-900</td><td>Substation Center</td><td>11kV/400V + 5000A SCR</td><td>IEEE 519 / IEC</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">PLC-1000</td><td>Siemens Safety PLC</td><td>S7-1500F fail-safe</td><td>IEC 61511 SIL-2</td></tr>
              <tr><td style="padding: 3px; font-weight: 700;">PIP-1100</td><td>Process Piping</td><td>Sch 80S 316L flanged</td><td>ASME B31.3 / B31.12</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 15 • AUTOCAD 2025 PRODUCTION DRAWINGS',
    title: 'AutoCAD 2025 ISO A1 Master Shop Drawings & Schedules',
    subtitle: 'Multi-Angle Orthographic Elevations, Equipment Schedules, and Piping Elevations',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 10px; cursor: pointer;" onclick="showModal('assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png', 'AutoCAD 2025 ISO A1 Master Engineering Drawing (300 DPI)')">
          <img src="assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png" alt="AutoCAD Master Drawing" style="width: 100%; max-height: 380px; object-fit: contain; border-radius: 4px; border: 1px solid #CBD5E1;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
            Fig: AutoCAD 2025 ISO A1 Sheet (DWG-PEM-H2-COMM-002) • Click to Zoom
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: var(--space-3);">
          <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--accent); margin: 0 0 8px; font-size: 13px; font-weight: 700;">Production Sheet Layout (841 x 594 mm)</h4>
            <div style="font-size: 12px; line-height: 1.5; color: var(--text-body);">
              <p style="margin: 0 0 6px;"><strong>Viewport A:</strong> Mezzanine front elevation (+6.200m RO, +4.360m TO, +0.000m TO grade level).</p>
              <p style="margin: 0 0 6px;"><strong>Viewport B:</strong> Top plot plan equipment layout (12,000mm span, Grids 1 to 4 and A to C).</p>
              <p style="margin: 0 0 6px;"><strong>Viewport C:</strong> Detail Section D-D: ASME Sec VIII UG-27 wall sizing &amp; ASTM B265 titanium flow channel profile.</p>
              <p style="margin: 0;"><strong>Viewport D:</strong> Equipment schedule and fabrication bill of materials.</p>
            </div>
          </div>

          <div style="background: #F8FAFC; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <h4 style="color: var(--text-main); margin: 0 0 4px; font-size: 12px; font-weight: 700;">Plotted via AutoCAD Accoreconsole</h4>
            <p style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); margin: 0;">
              Exported as vector PDF with 1:1 scale geometry, exact title blocks, and verified layer hierarchy.
            </p>
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 16 • QUANTITATIVE ENGINEERING CALCULATIONS',
    title: 'Deterministic Mechanical & Safety Sizing Formulations',
    subtitle: 'Executable Code Verification Across ASME, ISO, and ASTM Governing Standards',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">ASME BPVC SEC VIII DIV 1 (UG-27)</div>
          <p style="font-size: 11.5px; line-height: 1.5; margin: 0 0 4px;">Knockout Drum V-300 Shell Wall Thickness:</p>
          <div style="background: #FFFFFF; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: var(--accent); margin-bottom: 6px;">
            t = (P &times; R) / (S &times; E - 0.6 &times; P) + CA = 12.10 mm &rarr; 14.0 mm plate selected
          </div>
          <p style="font-size: 11px; color: var(--text-muted); margin: 0;">Resulting MAWP = 35.28 bar (+17.6% margin). Hydrotest pressure = 45.86 bar (1.3x).</p>
        </div>

        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #059669; margin-bottom: 4px;">ASME BPVC SEC VIII DIV 3 (KD-250)</div>
          <p style="font-size: 11.5px; line-height: 1.5; margin: 0 0 4px;">350-Bar Buffer Accumulators (ST-800):</p>
          <div style="background: #FFFFFF; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: #059669; margin-bottom: 6px;">
            &sigma;_hoop = P &times; (r_o² + r_i²) / (r_o² - r_i²) = 176.2 MPa (Yield: 860 MPa)
          </div>
          <p style="font-size: 11px; color: var(--text-muted); margin: 0;">Safety factor = 4.88x on yield stress. Calculated burst pressure = 2,196.4 bar.</p>
        </div>

        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #D97706; margin-bottom: 4px;">30-BAR DIFFERENTIAL STACK MECHANICS</div>
          <p style="font-size: 11.5px; line-height: 1.5; margin: 0 0 4px;">Membrane Extrusion &amp; Pore Bridging Stress:</p>
          <div style="background: #FFFFFF; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: #D97706; margin-bottom: 6px;">
            &sigma;_bridge = &Delta;P &times; (d_pore / 2t_mem)² = 0.046 MPa (Yield: 24.5 MPa)
          </div>
          <p style="font-size: 11px; color: var(--text-muted); margin: 0;">Safety margin = 530x against extrusion into PTL fiber felt.</p>
        </div>

        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #DC2626; margin-bottom: 4px;">ISO 14687 GRADE D PURITY (DR-500)</div>
          <p style="font-size: 11.5px; line-height: 1.5; margin: 0 0 4px;">Catalytic DeOxo &amp; Dual 13X TSA Dryers:</p>
          <div style="background: #FFFFFF; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; font-size: 10.5px; color: #DC2626; margin-bottom: 6px;">
            Residual O2 &le; 0.5 ppm (&Delta;T = 14.7°C) &bull; Dew Point &le; -70.0°C (&le; 1 ppm H2O)
          </div>
          <p style="font-size: 11px; color: var(--text-muted); margin: 0;">Exceeds SAE J2719 and ISO 14687 fuel-cell vehicle requirements.</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 17 • DUAL INDIAN & GLOBAL REGULATORY COMPLIANCE',
    title: 'Dual Compliance: The Factories Act, 1948 & PESO SMPV 2016 Harmonized Globally',
    subtitle: 'Seamless Domestic Execution Across India Harmonized 1:1 with Global Export Standards',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-2); background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">PRIMARY INDIAN STATUTORY FRAMEWORK</div>
          <div style="font-size: 11.5px; line-height: 1.5; color: var(--text-body);">
            <p style="margin: 0 0 6px;"><strong>The Factories Act, 1948 (Sec 21, 32, 33):</strong> Mandatory belt guards for CMP-700; mezzanine handrails with 1050mm top rails, 550mm intermediate rails, 100mm toeboards; non-slip IS 800 grating.</p>
            <p style="margin: 0 0 6px;"><strong>IS 800:2007 (Steel Construction):</strong> Limit state design for ISMB 300 beams. Calculated deflection = 18.42mm (well within L/350 = 34.28mm limit).</p>
            <p style="margin: 0 0 6px;"><strong>IS 875:2015 &amp; IS 1893:2016:</strong> Basic wind speed V_b = 50 m/s (coastal Dahej/Haldia) and Seismic Zone IV factors.</p>
            <p style="margin: 0;"><strong>PESO SMPV Rules, 2016:</strong> Governs V-300 HP separator (Rules 18, 19, 21); water jacket hydrotest at 45.86 bar; dual full-capacity PSVs.</p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: var(--space-2); background: #F0FDF4; padding: 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">GLOBAL EXPORT HARMONIZATION MATRIX</div>
          <table style="width: 100%; font-size: 10.5px; border-collapse: collapse;">
            <thead>
              <tr style="border-bottom: 1px solid #BBF7D0; color: #166534; text-align: left;">
                <th style="padding: 3px;">Indian Statutory Code</th><th>Global Export Equivalent</th>
              </tr>
            </thead>
            <tbody>
              <tr><td style="padding: 4px 3px; font-weight: 600;">The Factories Act Sec 32 / IS 800</td><td>ISO 14122-1..4 &bull; EN 1090-2 (EXC2)</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">PESO SMPV Rules 2016</td><td>ASME BPVC Sec VIII &bull; EU PED Mod H</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">NGHM BIS Draft Safety Code</td><td>ISO 22734:2019 &bull; NFPA 2</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">IS 16087 Automotive H2 Purity</td><td>ISO 14687:2019 Grade D &bull; SAE J2719</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">IS/IEC 60079 Zone 1/2 Electrical</td><td>ATEX 2014/34/EU (Ex d / Ex i)</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">IS/IEC 61511 (SIL-2 Safety)</td><td>IEC 61508 / 61511 SIL-2 (PFD 0.0051)</td></tr>
              <tr><td style="padding: 4px 3px; font-weight: 600;">PNGRB T4S Regulations</td><td>ASME B31.3 Cat M &bull; ASME B31.12</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 18 • COMMERCIAL MONETIZATION PORTFOLIO',
    title: 'Transparent Commercial Service Delivery Tiers',
    subtitle: 'ASD-STM100 High-Margin Digital Delivery & Turnkey Fabrication Licensing',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">TIER 1 • FORENSIC AUDIT</div>
          <h4 style="font-size: 14px; font-weight: 800; color: var(--text-main); margin: 0 0 4px;">Plant Health Audit &amp; CAE</h4>
          <div style="font-size: 16px; font-weight: 800; color: #059669; margin-bottom: 2px;">$7,500 <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">/ ₹6.2 Lakh</span></div>
          <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">5 Business Days Remote Turnaround</div>
          <ul style="font-size: 11px; line-height: 1.45; color: var(--text-body); padding-left: 14px; margin: 0;">
            <li>100h remote SCADA ingestion</li>
            <li>Butler-Volmer Tafel &amp; ECSA decay attribution</li>
            <li>ADVEKT CFD multiphase flow check</li>
            <li>ASME UG-27 shell thickness validation</li>
            <li>25-page deterministic forensic dossier</li>
          </ul>
        </div>

        <div style="background: #F0FDF4; padding: 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0; display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #166534; margin-bottom: 4px;">TIER 2 • EDGE FIRMWARE</div>
          <h4 style="font-size: 14px; font-weight: 800; color: #14532D; margin: 0 0 4px;">AERO-H2 Edge Digital Twin</h4>
          <div style="font-size: 16px; font-weight: 800; color: #166534; margin-bottom: 2px;">$2,500/mo <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">or $25k Perpetual</span></div>
          <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Annual SaaS or One-Time OEM License</div>
          <ul style="font-size: 11px; line-height: 1.45; color: #14532D; padding-left: 14px; margin: 0;">
            <li>Deployed on Siemens S7-1500 / Jetson</li>
            <li>In-operando &lt;1.4s biofilm anomaly trip</li>
            <li>Physics-constrained voltage forecasting</li>
            <li>+39.4% core stack life extension</li>
            <li>Eliminates chemical biocide wash loops</li>
            <li>99.9% uptime SLA commitment</li>
          </ul>
        </div>

        <div style="background: #FFFBEB; padding: 14px; border-radius: var(--radius-md); border: 1px solid #FDE68A; display: flex; flex-direction: column;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #B45309; margin-bottom: 4px;">TIER 3 • EPC FABRICATION</div>
          <h4 style="font-size: 14px; font-weight: 800; color: #78350F; margin: 0 0 4px;">Turnkey DFM CAD License</h4>
          <div style="font-size: 16px; font-weight: 800; color: #B45309; margin-bottom: 2px;">$45,000 <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">/ ₹37.5 Lakh</span></div>
          <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Optional $280k Full 100 TPD BEDP</div>
          <ul style="font-size: 11px; line-height: 1.45; color: #78350F; padding-left: 14px; margin: 0;">
            <li>Full SolidWorks 2025 assembly (.SLDASM)</li>
            <li>AutoCAD ISO A1 shop drawings (DWG/PDF)</li>
            <li>Progressive stamping die drawings (Ti plate)</li>
            <li>ASME Section VIII Div 1/3 calculations</li>
            <li>The Factories Act mezzanine design</li>
            <li>Target: EPC fabricators (L&T, Technip)</li>
          </ul>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 19 • MASTER PROCUREMENT BOM',
    title: 'Tier-1 Industrial Vendor Bill of Materials (BOM)',
    subtitle: 'Reconciled Component Specifications from Verified Tier-1 Industrial Manufacturers',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 6px;">MECHANICAL &amp; STRUCTURAL HARDWARE</div>
          <ul style="font-size: 11.5px; line-height: 1.5; color: var(--text-body); padding-left: 16px; margin: 0;">
            <li><strong>Structural Steel:</strong> Tata Steel ISMB 300 / IS 2062 Grade B (Hot-Dip Galvanized 85 µm)</li>
            <li><strong>Bipolar Flow Plates:</strong> Sandvik / ATI ASTM B265 Grade 2 Titanium Sheet (0.80mm cold-rolled)</li>
            <li><strong>Separator Pressure Vessels:</strong> Larsen &amp; Toubro Heavy Eng (SA-240 Type 316L, 14.0mm plate)</li>
            <li><strong>Catalytic DeOxo Reactor:</strong> BASF / Johnson Matthey 0.5% wt Pd/Al2O3 extruded catalyst pellets</li>
            <li><strong>High-Pressure Compressors:</strong> PDC Machines / Hofer Metal Diaphragm Compressor (350 bar rated)</li>
            <li><strong>Buffer Storage:</strong> Everest Kanto Cylinder (EKC) / IS 7285 Type 1 seamless Cr-Mo steel</li>
            <li><strong>Process Valves &amp; Fittings:</strong> Swagelok 316L medium-pressure cone &amp; thread valves</li>
          </ul>
        </div>

        <div style="background: #F0FDF4; padding: 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 6px;">ELECTRICAL, CONTROLS &amp; MEAs</div>
          <ul style="font-size: 11.5px; line-height: 1.5; color: #14532D; padding-left: 16px; margin: 0;">
            <li><strong>Solid Electrolyte Membrane:</strong> Chemours Nafion HP / N117 ePTFE reinforced composite (85 µm)</li>
            <li><strong>Porous Transport Layers:</strong> Bekaert Titanium sintered fiber felt with microporous layer (PTL)</li>
            <li><strong>Safety PLC Controller:</strong> Siemens SIMATIC S7-1500F Fail-Safe CPU 1518F-4 PN/DP</li>
            <li><strong>DC Power Rectifier:</strong> ABB / Schneider Electric 12-Pulse Thyristor Rectifier (5000A DC)</li>
            <li><strong>High-Purity Busbars:</strong> Hindalco C11000 Electrolytic Tough Pitch (ETP) 99.9% solid copper</li>
            <li><strong>NDIR Hydrogen Transmitters:</strong> Crowcon / Draeger Polytron 8720 fast NDIR detection (&lt;500ms)</li>
            <li><strong>Water Purification Pumps:</strong> Grundfos CRN 316L multi-stage centrifugal sanitary pumps</li>
          </ul>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 20 • CONCLUSIONS & ROADMAP',
    title: 'Conclusions & Immediate Commercial Implementation Roadmap',
    subtitle: 'Validated Engineering Deliverables Ready for Commercial Tender & Site Deployment',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); height: 100%;">
        <div style="display: flex; flex-direction: column; gap: var(--space-3); background: var(--bg-wash); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">CORE SCIENTIFIC &amp; TECHNICAL CONCLUSIONS</div>
          <div style="font-size: 11.5px; line-height: 1.5; color: var(--text-body);">
            <p style="margin: 0 0 6px;"><strong>Virtual Sensing Accuracy:</strong> GPU-resident XGBoost reverse-engineers initial catalyst morphology with sub-microvolt accuracy (1.911 µV/h error, R² = 0.963).</p>
            <p style="margin: 0 0 6px;"><strong>Thermodynamic Generalization:</strong> Butler-Volmer physics constraints eliminate unphysical degradation drift over 100h horizons (NRMSE = 0.054).</p>
            <p style="margin: 0 0 6px;"><strong>Decoupled Biological Defense:</strong> Dielectric EPS biofilm anomalies isolated in &lt;1.40s, triggering autonomous DI water recovery without biocide chemicals.</p>
            <p style="margin: 0;"><strong>Commercial Payback:</strong> +39.4% core stack life extension saves $420,000 in replacement CAPEX, lowering LCOH by -$0.42/kg H2 with Month 4.8 breakeven.</p>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: var(--space-2); background: #F0FDF4; padding: 14px; border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 4px;">IMMEDIATE INDUSTRIAL ONBOARDING PROTOCOL</div>
          <div style="font-size: 11.5px; line-height: 1.5; color: #14532D;">
            <p style="margin: 0 0 4px;"><strong>Step 1:</strong> Mutual NDA executed; client provides 100h min-resolution operational CSV logs.</p>
            <p style="margin: 0 0 4px;"><strong>Step 2:</strong> ADVEKT engine deconvolutes Tafel slope and delivers validated ASME/PESO dossier in 5 days.</p>
            <p style="margin: 0 0 4px;"><strong>Step 3:</strong> Containerized C++ runtime compiled onto on-prem Siemens S7-1500F PLC for zero-risk side-by-side passive tracking.</p>
            <p style="margin: 0 0 8px;"><strong>Step 4:</strong> Autonomous DI surge solenoid interlock engaged after baseline calibration.</p>
          </div>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid #BBF7D0; margin-top: auto;">
            <div style="font-size: 11px; font-weight: 700; color: #166534;">CONTACT CORRESPONDING AUTHOR:</div>
            <div style="font-size: 12px; font-weight: 700; color: #0F172A; margin-top: 2px;">Sourish Senapati</div>
            <div style="font-size: 11px; color: var(--text-muted);">Dept. of Chemical Engineering, Jadavpur University &bull; sourishs.chem.ug@jadavpuruniversity.in</div>
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
