/**
 * HYDROSENSE™ 15-Slide Master Presentation Deck Modal
 * Pure Architectural Light Mode, WCAG AAA High Contrast,
 * Zero Dark Voids, Keyboard Navigable (Arrow Keys / Escape).
 * Fully Integrates Peter Thiel's Zero to One Strategic Framework.
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
          <p style="margin-top: 10px; font-size: 13px; line-height: 1.5;"><strong>Srestha Biswash</strong><br>Dept. of Microbiology &amp; Biotech, Sister Nivedita University</p>
        </div>
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-pass); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Commercial &amp; Technical Summary</h4>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 1:</strong> Virtual sensing XGBoost (R² = 0.963, sub-microvolt accuracy: 1.911 µV/h).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 2:</strong> PIBT Transformer anchored by Butler-Volmer constraint (NRMSE = 0.054).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Phase 3:</strong> Decoupled Bayesian fault isolation (&lt;1.4s trip latency, F1 = 0.990).</p>
          <p style="font-size: 13px; line-height: 1.6;">• <strong>Payback:</strong> +39.4% lifespan extension (21,428h &rarr; 29,879h, -$0.42/kg H₂).</p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 2 • THIEL PILLAR 1: NICHE MONOPOLY',
    title: 'Conquer a Small Market, Then Expand to Adjacent Sectors',
    subtitle: 'Zero Greenfield Competition: Capturing the 250 MW Brownfield Industrial Beachhead',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--accent); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">The Beachhead Monopoly (Chlor-Alkali &amp; Refineries)</h4>
          <p style="font-size: 13px; line-height: 1.55; margin-bottom: 8px;">Rather than burning billions competing for speculative greenfield tenders, HYDROSENSE captures existing brownfield chlor-alkali and refinery hydrogen loops:</p>
          <ul style="margin-left: 18px; line-height: 1.7; font-size: 12.5px; color: var(--text-body);">
            <li><strong>250 MW addressable fleet</strong> across India and Southeast Asia.</li>
            <li><strong>$0 new substation CAPEX:</strong> Plugs directly into existing 5,000A DC busbars and 30-bar delivery headers.</li>
            <li><strong>High urgency pain:</strong> Biofilm-induced trips and unmonitored degradation cost operators $420,000 per stack teardown.</li>
          </ul>
        </div>
        <div style="background: #FFFFFF; padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: #0284C7; margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Sequential Adjacent Market Expansion</h4>
          <p style="font-size: 13px; line-height: 1.55; margin-bottom: 8px;">Once cash-flow positive and entrenched with compounding Bayesian fleet priors, the modular architecture scales outward:</p>
          <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px;">
            <div style="padding: 6px 10px; background: #F8FAFC; border-left: 3px solid #0284C7; border-radius: 2px;"><strong>Step 1:</strong> Brownfield Chlor-Alkali &amp; Refineries (Immediate cash-flow)</div>
            <div style="padding: 6px 10px; background: #F8FAFC; border-left: 3px solid #D97706; border-radius: 2px;"><strong>Step 2:</strong> Green Steel DRI Shaft Furnaces (100% H₂ Direct Reduced Iron)</div>
            <div style="padding: 6px 10px; background: #F8FAFC; border-left: 3px solid #059669; border-radius: 2px;"><strong>Step 3:</strong> Heavy Mobility &amp; Fueling (350/700 bar ISO 14687 Grade D)</div>
            <div style="padding: 6px 10px; background: #F8FAFC; border-left: 3px solid #7C3AED; border-radius: 2px;"><strong>Step 4:</strong> GW-Scale Utility Green Ammonia &amp; Synthetic E-Fuels</div>
          </div>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 3 • THIEL PILLAR 2: VERTICAL INTEGRATION',
    title: 'Vertical Integration Eliminates EPC Margin Destruction',
    subtitle: 'Full-Stack Ownership: Kinetic Priors + Edge AI Firmware + SolidWorks DFM + DI Chemistry',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--status-alert-bg); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--status-alert-border);">
          <h4 style="color: var(--status-alert); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">Conventional Fragmented EPC Architecture (Fails)</h4>
          <p style="font-size: 12.5px; line-height: 1.6; color: var(--text-main);">
            • <strong>Disjointed Supply Chain:</strong> Stack OEM blames compressor vendor; compressor vendor blames SCADA contractor.<br>
            • <strong>Layered EPC Markups:</strong> 2.0x to 2.5x markups passed to asset owner with zero performance liability.<br>
            • <strong>Unmonitored Stack Degradation:</strong> Disconnected generic PLCs fail to catch sub-microvolt degradation or biofilm drift.
          </p>
        </div>
        <div style="background: #F0FDF4; padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid #BBF7D0;">
          <h4 style="color: #15803D; margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">HYDROSENSE Vertically Integrated Architecture</h4>
          <p style="font-size: 12.5px; line-height: 1.6; color: var(--text-main);">
            • <strong>Layer 1 (Physics Core):</strong> Butler-Volmer and ECSA electrochemical state estimators.<br>
            • <strong>Layer 2 (Edge Firmware):</strong> AERO-H2 compiled directly for Siemens S7-1500F / STM32.<br>
            • <strong>Layer 3 (Mechanical Reality):</strong> Turnkey SolidWorks 2025 SP1.2 B-Rep skids &amp; ASME Sec VIII Div 1 &amp; 3 vessels.<br>
            • <strong>Layer 4 (Water Chemistry):</strong> Closed-loop AERO-PURE Unit 650 with automated DI surge mitigations.
          </p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 4 • THIEL PILLAR 3: LAST MOVER ADVANTAGE',
    title: 'The Last Mover Advantage: Synthesizing Early Bird Inventions',
    subtitle: 'Let Early Birds Burn Billions on Hardware; Standardize the Missing Intelligent Layer',
    body: `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); margin-top: var(--space-4);">
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--accent); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">What Early Birds De-Risked (Capital Burnt)</h4>
          <p style="font-size: 12.5px; line-height: 1.6;">
            • <strong>Electrochemical Chemistry:</strong> Nel, ITM, and Plug Power invested billions de-risking Nafion 117 membranes, Pt/C cathode formulations, and IrO₂ anode catalysts.<br>
            • <strong>Industrial SCADA Standards:</strong> Siemens and Schneider standardized PROFINET, SIL-2 safety PLC architectures, and 12-pulse thyristor rectifiers.<br>
            • <em>Result:</em> The physical hardware is now commoditized and reliable when operated at static steady-state.
          </p>
        </div>
        <div style="background: var(--bg-wash); padding: var(--space-5); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
          <h4 style="color: var(--status-pass); margin-bottom: 8px; font-size: 13.5px; font-weight: 700;">The Decisive Missing Innovation We Invented</h4>
          <p style="font-size: 12.5px; line-height: 1.6;">
            • <strong>Dynamic Degradation Mitigation:</strong> Early birds collapsed under renewable power ramps because generic PLCs could not model kinetic decay.<br>
            • <strong>The PIBT Transformer Digital Twin:</strong> We engineered the deterministic software and CAD layer that makes early bird hardware commercially viable.<br>
            • <strong>Compounding Fleet Moat:</strong> +39.4% core life extension (21,428h &rarr; 29,879h) and -$0.42/kg H₂ LCOH capture the dominant market share.
          </p>
        </div>
      </div>
    `
  },
  {
    tag: 'SLIDE 5 • THE $420,000 PROBLEM',
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
    tag: 'SLIDE 6 • SOLIDWORKS MASTER ASSEMBLY',
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
    tag: 'SLIDE 7 • AUTOCAD ISO A1 DRAWING',
    title: 'AutoCAD Plot Plan &amp; Engineering Schedules',
    subtitle: 'DWG-PEM-H2-COMM-002 Rev 5: Fully Documented P&amp;ID, ASME UG-27 Proofs &amp; LOPA Safety',
    body: `
      <div style="text-align: center;">
        <img src="assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="AutoCAD Drawing">
        <p style="font-size: 13px; color: var(--text-muted);">ISO A1 shop drawing complete with ISA-5.1 sensor tagging, hazardous area classification, and IEC 61511 SIL-2 trip schedules.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 8 • PHASE 1 VIRTUAL SENSING',
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
    tag: 'SLIDE 9 • PHASE 1 PARITY VALIDATION',
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
    tag: 'SLIDE 10 • PHASE 2 PIBT TRANSFORMER',
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
    tag: 'SLIDE 11 • PHASE 3 FAULT MITIGATION',
    title: 'Dual-Panel Decoupled Biofilm Anomaly Detection',
    subtitle: 'Isolating Dielectric Biofilm from Electrochemical Decay (&lt;1.4s Latency, F1 = 0.990)',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 4">
        <p style="font-size: 13px; color: var(--text-muted);">Instantaneous SCADA trip at t=60.02h throttles current to 1.3 A/cm² and boosts DI water to 28 L/min, eliminating toxic chemical flushes.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 12 • TECHNO-ECONOMIC PAYBACK',
    title: '+39.4% Stack Lifespan Extension',
    subtitle: 'Extends Core Life by 8,451 Hours &amp; Reduces LCOH by $0.42/kg H₂',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig5_Lifespan_Extension_Techno_Economic.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 5">
        <p style="font-size: 13px; color: var(--text-muted);">Core stack life extended from 21,428h to 29,879h (3.41 years full load), deferring $420,000 in replacement CAPEX per 5 MW unit.</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 13 • UNIVERSAL MULTI-PLANT MATRIX',
    title: 'Standardized Turnkey Deployment Across 5 Heavy Sectors',
    subtitle: 'ISMB 300 Mezzanine Skid Footprint Enables Plug-and-Play Integration Without Civil Redesign',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 7">
        <p style="font-size: 13px; color: var(--text-muted);">Modular adaptation across Refineries (50-100 TPD), Chlor-Alkali (20-40 TPD), Green Steel (80-150 TPD), Fertilizers (100-250 TPD), and Heavy Mobility (5-15 TPD).</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 14 • COMMERCIAL MONETIZATION &amp; PAYBACK',
    title: 'ASD-STM100 Commercialization &amp; ROI Model',
    subtitle: 'Zero Cash Burn, High-Margin Digital Delivery, Breakeven in Month 3.8',
    body: `
      <div style="text-align: center;">
        <img src="assets/Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png" style="max-height: 420px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 12px; background: #FFFFFF;" alt="Fig 8">
        <p style="font-size: 13px; color: var(--text-muted);">Commercial EPC portfolio (Tier 1: $28.5k audit, Tier 2: $3.5k/mo edge SLA, Tier 3: $85k EPC skid license / $480k full BEDP) delivering 497% 2-year client ROI ($559,500 net cash profit per 5 MW block).</p>
      </div>
    `
  },
  {
    tag: 'SLIDE 15 • READY FOR COMMERCIAL TENDER',
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
