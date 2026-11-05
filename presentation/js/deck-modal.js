/**
 * HYDROSENSE™ 15-Slide Master Presentation Deck Modal
 * Pure Architectural Light Mode, WCAG AAA High Contrast,
 * Zero Dark Voids, Keyboard Navigable (Arrow Keys / Escape).
 */

let deckIdx = 0;

const presentationSlides = [
  {
    tag: 'SLIDE 1 • DEFENSE COVER',
    title: 'Machine Learning for Predicting Catalyst Efficiency in Green Hydrogen Production',
    subtitle: 'Predictive Maintenance, Fault Detection, and Closed-Loop Degradation Mitigation',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--accent); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Academic Affiliations</h4>
          <p style="font-size: 13px; line-height: 1.5;"><strong>Sourish Senapati</strong> (Corresponding Author)<br>Dept. of Chemical Engineering, Jadavpur University<br><span style="font-family: monospace; font-size: 11.5px; color: var(--text-muted);">sourishs.chem.ug@jadavpuruniversity.in</span></p>
          <p style="margin-top: 10px; font-size: 13px; line-height: 1.5;"><strong>Srestha Biswash</strong><br>Dept. of Microbiology & Biotech, Sister Nivedita University</p>
        </div>
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-pass); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Commercial & Technical Summary</h4>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 1:</strong> Virtual sensing XGBoost (R² = 0.963, sub-microvolt accuracy: 1.911 µV/h).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 2:</strong> PIBT Transformer anchored by Butler-Volmer constraint (NRMSE = 0.054).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 3:</strong> Decoupled Bayesian fault isolation (&lt;1.4s trip latency, F1 = 0.990).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Payback:</strong> +39.4% lifespan extension (21,428h &rarr; 29,879h, -$0.42/kg H₂).</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 2 • THE $420,000 PROBLEM',
    title: 'The Multi-Megawatt Degradation Dilemma',
    subtitle: 'Why Unmitigated Dynamic Cycling Destroys Commercial PEM Electrolyzers',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div>
          <p style="margin-bottom: 12px; font-size: 13.5px; line-height: 1.5;">Under renewable power fluctuations (solar PV / wind ramps), operating current spikes unpredictably from 0.2 to 2.5 A/cm². This triggers rapid Ostwald ripening and catalyst dissolution.</p>
          <ul style="margin-left: 20px; line-height: 1.8; font-size: 13px; color: var(--text-body);">
            <li>Unmitigated degradation rates exceed <strong>15 to 22 µV/h</strong>.</li>
            <li>Premature stack teardowns cost <strong>$420,000 per 5 MW core</strong> every 2.45 years.</li>
            <li>Generic SCADA alarms trip only after irreversible membrane dry-out or pinhole rupture.</li>
          </ul>
        </div>
        <div style="background: var(--status-alert-bg); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--status-alert-border);">
          <h4 style="color: var(--status-alert); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">The Hidden Biofilm Failure Mode</h4>
          <p style="font-size: 13px; line-height: 1.5; color: var(--text-main);">Industrial feed water loops develop dielectric extracellular polymeric substance (EPS) biofilms within 600 hours. Legacy operators mistake this for irreversible catalyst degradation, conducting unnecessary manual teardowns and applying toxic chemical biocides.</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 3 • SOLIDWORKS MASTER ASSEMBLY',
    title: 'Commercial Turnkey Hydrogen Plant (100 TPD)',
    subtitle: '11 Subsystems Parameterized in SolidWorks 2025 SP1.2 (B-Rep Geometry)',
    body: `
      <div style="text-align: center;">
        <img src="assets/PEM_Commercial_Plant_Catalog_Render.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="SolidWorks Plant Render">
        <p style="font-size: 13px; color: var(--text-muted);">Full parametric B-Rep CAD assembly compliant with ASME BPVC Section VIII Div 1 &amp; 3, API 618, and ISMB 300 structural skids.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 4 • AUTOCAD ISO A1 DRAWING',
    title: 'AutoCAD Plot Plan & Engineering Schedules',
    subtitle: 'DWG-PEM-H2-COMMERCIAL-PLANT-002: Fully Documented P&ID and LOPA Safety Loops',
    body: `
      <div style="text-align: center;">
        <img src="assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="AutoCAD Drawing">
        <p style="font-size: 13px; color: var(--text-muted);">ISO A1 shop drawing complete with ISA-5.1 sensor tagging, hazardous area classification, and IEC 61511 SIL-2 trip schedules.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 5 • PHASE 1 VIRTUAL SENSING',
    title: 'XGBoost Feature Importance Attribution',
    subtitle: 'Sub-Microvolt Accuracy on Anode Kinetic Decay Descriptors',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig1_XGBoost_Feature_Importance.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 1">
        <p style="font-size: 13px; color: var(--text-muted);">Tafel Slope + ECSA + PTL Porosity account for &gt;74.4% of total kinetic degradation variance (R² = 0.963, NRMSE = 0.046).</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 6 • PHASE 1 PARITY VALIDATION',
    title: 'Empirical Parity Plot (n = 140 Samples)',
    subtitle: 'Sub-Microvolt Fidelity Across Galvanostatic Operating Regimes',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 2">
        <p style="font-size: 13px; color: var(--text-muted);">Statistical validation: R² = 0.9630, RMSE = 1.9110 µV/h, Max Residual &lt; 1.4 µV/h bounded by ±5% and ±10% tolerance bands.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 7 • PHASE 2 PIBT TRANSFORMER',
    title: 'Physics-Informed Bayesian Degradation Tracking',
    subtitle: 'Butler-Volmer Thermodynamic Loss Prevents Divergence Under Fluctuating Loads',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 3">
        <p style="font-size: 13px; color: var(--text-muted);">31.4% error reduction over unconstrained LSTM baselines (NRMSE = 0.054) with dynamic 95% Bayesian Monte Carlo confidence intervals.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 8 • PHASE 3 FAULT MITIGATION',
    title: 'Dual-Panel Decoupled Biofilm Anomaly Detection',
    subtitle: 'Isolating Dielectric Biofilm from Electrochemical Decay (<1.4s Latency, F1 = 0.990)',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 4">
        <p style="font-size: 13px; color: var(--text-muted);">Instantaneous SCADA trip at t=60.02h throttles current to 1.3 A/cm² and boosts DI water to 28 L/min, eliminating toxic chemical flushes.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 9 • TECHNO-ECONOMIC PAYBACK',
    title: '+39.4% Stack Lifespan Extension',
    subtitle: 'Extends Core Life by 8,451 Hours & Reduces LCOH by $0.42/kg H₂',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig5_Lifespan_Extension_Techno_Economic.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 5">
        <p style="font-size: 13px; color: var(--text-muted);">Core stack life extended from 21,428h to 29,879h (3.41 years full load), deferring $420,000 in replacement CAPEX per 5 MW unit.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 10 • CROSS-SECTORAL GENERALIZATION',
    title: 'Multi-Scale Validation Across 4 Regimes',
    subtitle: 'Consistent >93% Accuracy from 25 cm² Single Cells to 100 TPD Pilot Plants',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig6_Cross_Sectoral_Validation.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 6">
        <p style="font-size: 13px; color: var(--text-muted);">Proven transferability across benchtop single cells, 250 cm² 10-cell industrial stacks, chlor-alkali brine cells, and multi-MW arrays.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 11 • DFM MULTI-PLANT ARCHITECTURE',
    title: 'Design for Manufacturing (DFM) & Multi-Plant Retrofit',
    subtitle: 'Progressive Stamping of Titanium Plates & ASME Sizing for 5 Heavy Sectors',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 7">
        <p style="font-size: 13px; color: var(--text-muted);">Validated ASTM B265 Gr. 2 channel profile (w=1.2mm, d=0.8mm, &theta;=7°), ASME UG-27 shell sizing (30 bar), and 5-sector deployment matrix.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 12 • COMMERCIAL MONETIZATION',
    title: 'ASD-STM100 Commercialization & ROI Model',
    subtitle: 'Zero Cash Burn, High-Margin Digital Delivery, Breakeven in Month 4.8',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 8">
        <p style="font-size: 13px; color: var(--text-muted);">3-tier monetization portfolio (Tier 1: $7.5k audit, Tier 2: $2.5k/mo edge SaaS, Tier 3: $45k EPC license) delivering 485% 2-year client ROI.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 13 • DUAL-REVENUE AERO-PURE 650',
    title: 'AERO-PURE Unit 650 Ultrafiltration',
    subtitle: 'Repurposing Industrial Wastewater for Electrolysis & Secondary Water Sales',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--accent); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Primary Electrolyzer Duty</h4>
          <p style="font-size: 13px; line-height: 1.6;">• Produces ASTM D1193 Type 1 ultrapure water (&gt;18.2 M&Omega;·cm).</p>
          <p style="font-size: 13px; line-height: 1.6;">• Cuts TOC &lt; 20 ppb and eliminates multivalent metal ions.</p>
          <p style="font-size: 13px; line-height: 1.6;">• Protects titanium PTLs and membrane electrode assemblies.</p>
        </div>
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-pass); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Secondary Revenue Monetization</h4>
          <p style="font-size: 13px; line-height: 1.6;">• Filters cooling tower blowdown and boiler feed water.</p>
          <p style="font-size: 13px; line-height: 1.6;">• Adds $85,000/yr in secondary purified water utility revenue.</p>
          <p style="font-size: 13px; line-height: 1.6;">• Turns electrolyzer water CAPEX into an independent profit center.</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 14 • ZERO TO ONE COMPETITIVE MOAT',
    title: 'Defensibility & Category Dominance',
    subtitle: 'How HYDROSENSE Compounds Network Effects and Data Moats',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-notice); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">The Compounding Bayesian Prior</h4>
          <p style="font-size: 13px; line-height: 1.6;">Every plant operating our firmware reports anonymized decay trajectories to our cloud prior library. New installations start with thousands of hours of pre-trained priors, making late-entry competitors obsolete.</p>
        </div>
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-pass); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">ASME &amp; CAD B-Rep Lock-In</h4>
          <p style="font-size: 13px; line-height: 1.6;">EPC contractors build directly to our SolidWorks 2025 SP1.2 drawings. Once the ISMB 300 skids and piping tie-ins are installed, our edge firmware is the native operating system.</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 15 • CONCLUSION & TENDER CALL-TO-ACTION',
    title: 'Ready for Commercial Deployment',
    subtitle: '5-Day Remote Audits Available for 5 MW to 100 MW Industrial Operators',
    body: `
      <div style="text-align: center; padding: 40px var(--space-4);">
        <h3 style="font-size: 24px; color: var(--text-main); margin-bottom: 10px;">Deploy HYDROSENSE™ on Your Facility</h3>
        <p style="font-size: 15px; color: var(--text-muted); max-width: 680px; margin: 0 auto 24px;">
          Full engineering dossiers, SolidWorks CAD assemblies, AutoCAD shop drawings, and edge firmware licenses are ready for immediate tender.
        </p>
        <div style="display: flex; justify-content: center; gap: var(--space-4);">
          <a href="Green_Hydrogen_ML_Presentation.pptx" class="btn btn-primary" download>Download .PPTX Deck</a>
          <button class="btn btn-secondary" onclick="closeDeck()">Explore Web Portal</button>
        </div>
      </div>
    `
  }
];

function openDeck() {
  const overlay = document.getElementById('deckOverlay');
  if (overlay) {
    overlay.classList.add('active');
    deckIdx = 0;
    renderDeckSlide();
  }
}

function closeDeck() {
  const overlay = document.getElementById('deckOverlay');
  if (overlay) overlay.classList.remove('active');
}

function renderDeckSlide() {
  const cur = presentationSlides[deckIdx];
  const badge = document.getElementById('slideBadge');
  if (badge) badge.innerText = `Slide ${deckIdx + 1} / ${presentationSlides.length}`;

  const box = document.getElementById('deckSlideBox');
  if (!box) return;

  box.innerHTML = `
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">
      ${cur.tag}
    </div>
    <h2 style="font-size: 22px; font-weight: 800; color: var(--text-main); margin-bottom: 2px;">
      ${cur.title}
    </h2>
    <div style="font-size: 13.5px; color: var(--text-muted); margin-bottom: var(--space-4); border-bottom: 1px solid var(--border-subtle); padding-bottom: var(--space-2);">
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
        {left: '$', right: '$', display: false}
      ]
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
