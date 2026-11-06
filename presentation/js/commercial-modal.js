/**
 * HYDROSENSE™ Commercial Scope of Work (SOW), SLA & Licensing Modal
 * Authoritative, institutional contract specification providing deterministic engineering deliverables.
 * Structured under Indian Industrial Statutory Frameworks (The Factories Act 1948,
 * Arbitration and Conciliation Act 1996, PESO SMPV Rules 2016).
 */

let activeCommercialTab = 'audit';

const commercialTiersData = {
  audit: {
    tag: 'TIER 1 • FORENSIC AUDIT & CAE SIMULATION',
    title: '5-Day Plant Health Audit & Multi-Physics Diagnostic Dossier',
    priceINR: '₹6,50,000 INR',
    priceUSD: '$7,800 USD',
    term: 'Fixed-Price Milestone Delivery • 5 Business Days Turnaround',
    target: 'Brownfield PEM & Chlor-Alkali Facilities (IOCL, BPCL, HMEL, Grasim, DCM Shriram, Tata Steel)',
    summary: 'Comprehensive forensic analysis of unmitigated cell voltage climb, parasitic OER/HER degradation, and CFD multi-phase flow distribution executed autonomously via the ADVEKT CAE engine.',
    milestones: [
      {
        day: 'Day 1',
        title: 'SCADA Data Ingestion & Signal Quality Audit',
        desc: 'Ingestion of 100-hour high-frequency SCADA telemetry (OPC-UA / CSV at 10 Hz minimum). Data validation, sensor drift filtering, and thermodynamic mass-energy balance reconciliation.'
      },
      {
        day: 'Day 2',
        title: 'Electrochemical Polarization & Butler-Volmer Deconvolution',
        desc: 'Tafel slope calculation, exchange current density (j_0) decay attribution, and separation of anodic OER kinetic overpotential from membrane ohmic impedance.'
      },
      {
        day: 'Day 3',
        title: 'ADVEKT Multi-Phase CFD & ASME Section VIII Structural Audit',
        desc: 'Two-phase (H2/H2O) flow maldistribution modeling across 50 bipolar channels. ASME BPVC Section VIII Div 1 UG-27(c)(1) vessel shell thickness and minimum corrosion allowance validation.'
      },
      {
        day: 'Day 4',
        title: 'Root-Cause Degradation Attribution',
        desc: 'Deterministic classification isolating catalyst dissolution (Pt/Ir crossover) vs membrane pinhole thinning vs dielectric EPS bacterial biofilm accumulation (<1.4s detection window).'
      },
      {
        day: 'Day 5',
        title: 'Formal Forensic Engineering Dossier & Executive Presentation',
        desc: 'Delivery of 25-page deterministic engineering report (ASME UG-27 & Tafel slope) with actionable operational recommendations, current derating profiles, and closed-loop flush protocols.'
      }
    ],
    inputs: [
      '100 hours of continuous SCADA operational logs (Cell Voltages, Current, Stack Differential-P, Inlet/Outlet Temperatures, DI Water Flow Rate).',
      'Electrolyzer stack technical datasheet & membrane electrode assembly (MEA) specification.',
      'Process & Instrumentation Diagram (P&ID) and General Arrangement drawings of the skid.'
    ],
    deliverables: [
      'Full 25-Page Deterministic Forensic Engineering Dossier (PDF).',
      'ASME Section VIII Div 1 UG-27 vessel thickness & MAWP certification calculation sheets.',
      'Tafel slope & electrochemical active surface area (ECSA) degradation attribution curve.',
      '1-Hour video debrief with Lead Chemical & Systems Engineers.'
    ]
  },

  firmware: {
    tag: 'TIER 2 • EDGE DIGITAL TWIN FIRMWARE',
    title: 'AERO-H2 Real-Time Edge Digital Twin Core Software License',
    priceINR: '₹2,10,000 INR / month',
    priceUSD: '$2,500 USD / month',
    term: 'Annual SaaS Subscription or ₹25,00,000 INR ($30,000 USD) Perpetual License per 5 MW Skid',
    target: 'Industrial Green Hydrogen Skid OEMs & Heavy Chemical Plant Operators',
    summary: 'Quantized Physics-Informed Butler-Volmer Transformer (PIBT) neural inference engine running locally on Siemens S7-1500F PLC, NVIDIA Jetson, or STM32 MCU for 24/7 in-operando health tracking.',
    milestones: [
      {
        day: 'Milestone 1',
        title: 'Hardware Bench Commissioning & Edge Flash',
        desc: 'Deployment of containerized C++/Rust inference runtime onto target edge hardware. Validation of deterministic cycle execution (<25 ms loop latency).'
      },
      {
        day: 'Milestone 2',
        title: 'Industrial SCADA Interconnect Integration',
        desc: 'Configuration of dual-redundant Modbus TCP and OPC-UA communication bridges. Mapping of 48 process tags into the AERO-H2 live state register.'
      },
      {
        day: 'Milestone 3',
        title: 'In-Operando Baseline Calibration',
        desc: 'Execution of automated 48-hour self-calibration routine under plant nominal operating regime to tune local stack membrane hydration constants.'
      },
      {
        day: 'Milestone 4',
        title: 'Autonomous Mitigation Interlock Commissioning',
        desc: 'Dry-run and wet-run verification of the sub-1.4s biofilm anomaly trip interlock, solenoid valve XV-202 fast-pulsing, and automated current derating.'
      }
    ],
    inputs: [
      'Target edge hardware specification (Siemens S7-1500 / NVIDIA Jetson Orin / IPC).',
      'Network architecture diagram & Modbus/OPC-UA register memory map.',
      'Design safety interlock matrix (Cause & Effect diagram).'
    ],
    deliverables: [
      'Containerized AERO-H2 Edge Core runtime binaries (Docker / ARM64 / x86-64 / SIMATIC).',
      'Sub-1.4s Biofilm Anomaly Trip & Butler-Volmer tracking algorithms.',
      '99.9% Software Uptime SLA with 4-hour critical incident emergency engineering support.',
      'Quarterly firmware updates incorporating refined MEA degradation models.'
    ]
  },

  license: {
    tag: 'TIER 3 • TURNKEY DFM CAD & BEDP LICENSE',
    title: 'Turnkey DFM CAD, Engineering Calculations & Manufacturing Package',
    priceINR: '₹42,00,000 INR per 5 MW Skid',
    priceUSD: '$50,000 USD per 5 MW Skid',
    term: 'One-Time Non-Exclusive EPC Fabrication License (Optional ₹2.35 Cr / $280,000 Full 100 TPD BEDP)',
    target: 'EPC Contractors & Heavy Fabricators (Larsen & Toubro, Technip Energies, ThyssenKrupp, BHEL, Adani)',
    summary: 'Complete engineering manufacturing package including native SolidWorks 2025 SP1.2 3D assemblies, AutoCAD 2025 ISO A1 production shop drawings, ASME vessel design calculations, and complete welding BOM.',
    milestones: [
      {
        day: 'Phase 1',
        title: 'Basic Engineering Design Package (BEDP) Handover',
        desc: 'Process Flow Diagrams (PFD), Process & Instrumentation Diagrams (P&ID), and comprehensive heat & mass balance data sheets for all 11 plant subsystems.'
      },
      {
        day: 'Phase 2',
        title: 'SolidWorks 2025 3D B-Rep Assembly & DFM Package',
        desc: 'Full native 3D SolidWorks assembly (.SLDASM, .STEP) including progressive stamping die profiles for ASTM B265 Grade 2 titanium bipolar plates with verified <18% thinning.'
      },
      {
        day: 'Phase 3',
        title: 'AutoCAD 2025 ISO A1 Shop Drawings',
        desc: 'Production fabrication drawings (DWG-PEM-H2-COMMERCIAL-PLANT-002 Rev 5 and DWG-PEM-H2-2026-001 Rev 1) in native .DWG and vector .PDF with zero collisions.'
      },
      {
        day: 'Phase 4',
        title: 'Statutory Verification & Welding Schedule',
        desc: 'ASME Section VIII Div 1 & 3 calculation sheets, The Factories Act 1948 Sec 32/33 structural calculations (IS 800:2007), and ASME Section IX GTAW welding schedules.'
      }
    ],
    inputs: [
      'Target plant site plot plan, soil bearing capacity, and seismic/wind load specifications (IS 875:2015).',
      'Utility interface parameters (Demineralized water supply, 11 kV transformer feed, cooling water loop).',
      'Client enterprise EPC fabrication standards & preferred vendor list.'
    ],
    deliverables: [
      'Complete SolidWorks 2025 SP1.2 3D assembly and Parasolid .X_T / .STEP files.',
      'Plotted AutoCAD ISO A1 fabrication drawings (.DWG, .DXF, .PDF).',
      'Full Bill of Materials (BOM) with 11 subsystems and commercial vendor catalog part numbers.',
      'ASME Section VIII Div 1 & Div 3 Verified Design Calculation Dossier (UG-27 & KD-250).',
      'IEC 61511 Functional Safety LOPA SIL-2 verification calculation ledger.'
    ]
  },

  payback: {
    tag: 'FINANCIAL DEFENSIBILITY & CUSTOMER ROI',
    title: 'Deterministic Techno-Economic Payback & Lifecycle ROI Analysis',
    priceINR: 'Month 4.8 Breakeven',
    priceUSD: '485% 2-Year ROI',
    term: 'Based on 5 MW Industrial PEM Facility Operating at 92% Annual Availability',
    target: 'Chief Technology Officers, Chief Financial Officers & Plant Asset Managers',
    summary: 'Clear quantitative proof explaining why industrial operators recoup the full investment within months by preventing stack degradation, unscheduled downtime, and chemical biocide flushes.',
    milestones: [
      {
        day: 'Capex Risk',
        title: '5 MW Stack Replacement Avoidance: $1,050,000 USD',
        desc: 'PEM stack replacement costs $210/kW. For a 5 MW facility, replacement costs $1.05M USD. Unmitigated biofouling and dynamic OER degradation forces premature stack replacement at 21,428 hours.'
      },
      {
        day: 'Life Extension',
        title: '+39.4% Operational Stack Lifespan: +8,451 Hours',
        desc: 'AERO-H2 PIBT predictive twin and sub-1.4s biofilm isolation extends stack life to 29,879 hours, deferring capital replacement by 1.15 years and saving $412,000 USD net.'
      },
      {
        day: 'Downtime',
        title: 'Prevented Unscheduled Plant Outage: $144,000 USD',
        desc: 'Avoids 21 days of emergency shutdown for membrane cleaning and replacement, preserving 90,720 kg of commercial hydrogen production.'
      },
      {
        day: 'Biocide Cost',
        title: 'Elimination of Toxic Biocide Loops: $38,000 USD / yr',
        desc: 'Closed-loop autonomous high-shear DI water pulsing removes bacterial biofilm physically, eliminating chemical biocide purchases and hazardous waste disposal.'
      }
    ],
    inputs: [
      'Standard industrial electricity tariff: $0.045 / kWh.',
      'Hydrogen off-take price: $4.50 / kg H2.',
      '5 MW Electrolyzer stack nameplate capacity (1000 Nm³/h H2).'
    ],
    deliverables: [
      'Deterministic financial cash-flow model showing positive net breakeven at Month 4.8.',
      'Comprehensive Levelized Cost of Hydrogen (LCOH) reduction of -$0.42 / kg H2.',
      'Full risk-adjusted sensitivity analysis (Monte Carlo N=10,000 iterations).'
    ]
  },

  governance: {
    tag: 'STATUTORY GOVERNANCE & LEGAL TERMS',
    title: 'Statutory Compliance, Indian Arbitration & Master Service Agreement',
    priceINR: 'Governed by Indian Law',
    priceUSD: 'Statutory Safeguards',
    term: 'The Arbitration and Conciliation Act, 1996 • Jurisdiction: New Delhi, India',
    target: 'Corporate Legal Counsel, Procurement Directors & Statutory Safety Inspectors',
    summary: 'Commercial engagements are executed under standardized institutional engineering contracts adhering strictly to Indian industrial legislation and international safety directives.',
    milestones: [
      {
        day: 'Statute 1',
        title: 'Indian Arbitration and Conciliation Act, 1996',
        desc: 'All disputes arising out of commercial deliverables shall be settled through binding arbitration administered in New Delhi under the Indian Arbitration and Conciliation Act, 1996.'
      },
      {
        day: 'Statute 2',
        title: 'The Factories Act, 1948 (Sections 32 & 33)',
        desc: 'All structural mezzanine, stairways, and walkway grating designs are verified compliant under Sections 32 & 33 of The Factories Act, 1948 and IS 800:2007 (Steel General Construction).'
      },
      {
        day: 'Statute 3',
        title: 'PESO SMPV (Unfired) Rules 2016 & Gas Cylinders Rules 2016',
        desc: 'High-pressure hydrogen storage buffer vessels (ST-800) and knockout drums (V-300) are engineered to comply with Petroleum and Explosives Safety Organization (PESO) statutory standards.'
      },
      {
        day: 'Statute 4',
        title: 'Confidentiality & Section 72A Information Technology Act, 2000',
        desc: 'Client SCADA operational data and plant telemetry are strictly safeguarded under bilateral Non-Disclosure Agreements (NDA) and Indian IT Act data protection regulations.'
      }
    ],
    inputs: [
      'Authorized corporate signatory details & Corporate Identification Number (CIN).',
      'Designated statutory safety officer contact.',
      'Facility environmental & hazardous chemicals clearance licenses.'
    ],
    deliverables: [
      'Fully executed Master Services Agreement (MSA) and Statement of Work (SOW).',
      'Mutual Non-Disclosure Agreement (M-NDA) with strict trade secret indemnification.',
      'Statutory compliance certificate for submission to Factory Inspectors and PESO authorities.'
    ]
  }
};

function openCommercialModal(tierKey = 'audit') {
  activeCommercialTab = tierKey;
  const overlay = document.getElementById('commercialModal');
  if (!overlay) return;

  renderCommercialModalContent();
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeCommercialModal() {
  const overlay = document.getElementById('commercialModal');
  if (overlay) overlay.classList.remove('active');
  document.body.style.overflow = 'auto';
}

function switchCommercialTab(tierKey) {
  activeCommercialTab = tierKey;
  renderCommercialModalContent();
}

function renderCommercialModalContent() {
  const cur = commercialTiersData[activeCommercialTab] || commercialTiersData.audit;
  const box = document.getElementById('commercialModalBody');
  if (!box) return;

  // Render navigation tab buttons
  const tabsHtml = `
    <div style="display: flex; gap: 4px; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 16px; border: 1px solid #E2E8F0; overflow-x: auto;">
      <button class="btn btn-secondary ${activeCommercialTab === 'audit' ? 'active' : ''}" style="padding: 6px 12px; font-size: 11.5px; ${activeCommercialTab === 'audit' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchCommercialTab('audit')">
        Tier 1: Forensic Audit (₹6.5L)
      </button>
      <button class="btn btn-secondary ${activeCommercialTab === 'firmware' ? 'active' : ''}" style="padding: 6px 12px; font-size: 11.5px; ${activeCommercialTab === 'firmware' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchCommercialTab('firmware')">
        Tier 2: Edge Firmware SLA
      </button>
      <button class="btn btn-secondary ${activeCommercialTab === 'license' ? 'active' : ''}" style="padding: 6px 12px; font-size: 11.5px; ${activeCommercialTab === 'license' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchCommercialTab('license')">
        Tier 3: Turnkey DFM CAD (₹42L)
      </button>
      <button class="btn btn-secondary ${activeCommercialTab === 'payback' ? 'active' : ''}" style="padding: 6px 12px; font-size: 11.5px; ${activeCommercialTab === 'payback' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchCommercialTab('payback')">
        Customer ROI & Payback (4.8 Mo)
      </button>
      <button class="btn btn-secondary ${activeCommercialTab === 'governance' ? 'active' : ''}" style="padding: 6px 12px; font-size: 11.5px; ${activeCommercialTab === 'governance' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchCommercialTab('governance')">
        Statutory Governance & Law
      </button>
    </div>
  `;

  // Milestone list
  const milestonesHtml = cur.milestones.map(m => `
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #0284C7; background: #EFF6FF; padding: 2px 6px; border-radius: 2px;">${m.day}</span>
        <span style="font-weight: 700; font-size: 13px; color: #0F172A;">${m.title}</span>
      </div>
      <p style="font-size: 12px; color: #475569; margin: 0; line-height: 1.5;">${m.desc}</p>
    </div>
  `).join('');

  // Client Inputs list
  const inputsHtml = cur.inputs.map(i => `
    <li style="font-size: 12px; color: #334155; margin-bottom: 6px; line-height: 1.4;">
      <span style="color: #64748B; font-weight: 700; margin-right: 6px;">&bull;</span>${i}
    </li>
  `).join('');

  // Deliverables list
  const deliverablesHtml = cur.deliverables.map(d => `
    <li style="font-size: 12px; color: #0F172A; font-weight: 500; margin-bottom: 6px; line-height: 1.4;">
      <span style="color: #059669; font-weight: 700; margin-right: 6px;">&#10003;</span>${d}
    </li>
  `).join('');

  box.innerHTML = `
    ${tabsHtml}

    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 12px; margin-bottom: 16px;">
      <div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #0284C7; text-transform: uppercase;">
          ${cur.tag}
        </div>
        <h3 style="font-size: 18px; font-weight: 800; color: #0F172A; margin: 4px 0 2px;">
          ${cur.title}
        </h3>
        <div style="font-size: 12px; color: #64748B;">
          Target Clients: <strong>${cur.target}</strong>
        </div>
      </div>

      <div style="text-align: right;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; color: #0F172A;">
          ${cur.priceINR} <span style="font-size: 14px; font-weight: 600; color: #64748B;">(${cur.priceUSD})</span>
        </div>
        <div style="font-size: 11px; color: #059669; font-weight: 600;">
          ${cur.term}
        </div>
      </div>
    </div>

    <div style="font-size: 13px; line-height: 1.6; color: #334155; margin-bottom: 16px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 14px; border-radius: 4px;">
      <strong>Scope Statement:</strong> ${cur.summary}
    </div>

    <div style="display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px;">
      <div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 8px;">
          Execution Schedule & Milestone Breakdown
        </div>
        ${milestonesHtml}
      </div>

      <div style="display: flex; flex-direction: column; gap: 12px;">
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 4px; padding: 12px 14px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 8px;">
            Client Technical Inputs Required
          </div>
          <ul style="list-style: none; padding: 0; margin: 0;">
            ${inputsHtml}
          </ul>
        </div>

        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 4px; padding: 12px 14px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #166534; text-transform: uppercase; margin-bottom: 8px;">
            Deterministic Deliverables Checklist
          </div>
          <ul style="list-style: none; padding: 0; margin: 0;">
            ${deliverablesHtml}
          </ul>
        </div>

        <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; padding: 12px 14px; text-align: center;">
          <div style="font-size: 11.5px; color: #64748B; margin-bottom: 10px;">
            Governed by The Indian Arbitration & Conciliation Act 1996 • New Delhi Jurisdiction
          </div>
          <div style="display: flex; gap: 8px; justify-content: center;">
            <a href="mailto:sourishs.chem.ug@jadavpuruniversity.in?subject=Formal%20Commercial%20Inquiry%20-%20${encodeURIComponent(cur.title)}" class="btn btn-primary" style="padding: 6px 14px; font-size: 12px;">
              Request Statement of Work (SOW)
            </a>
            <button class="btn btn-secondary" style="padding: 6px 14px; font-size: 12px;" onclick="closeCommercialModal()">
              Close Spec
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Global Exports
window.openCommercialModal = openCommercialModal;
window.closeCommercialModal = closeCommercialModal;
window.switchCommercialTab = switchCommercialTab;
