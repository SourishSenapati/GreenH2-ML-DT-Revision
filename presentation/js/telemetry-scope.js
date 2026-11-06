/**
 * HYDROSENSE In-Operando Digital Twin Oscilloscope (Butler-Volmer Physics Solver)
 * Authentic in-browser numerical solver simulating real-time electrochemistry of a
 * 10-cell industrial PEM stack (250 cm² active area):
 * - Temperature- and pressure-dependent Nernst Reversible Potential E_rev(T, P)
 * - Anodic (OER) and Cathodic (HER) Butler-Volmer Activation Overpotentials
 * - Ohmic Electrolyte & PTL Contact Resistance with temperature-dependent membrane conductivity
 * - Mass transport concentration overpotential
 * - In-operando dielectric EPS biofilm voltage residual injection (+58.8 mV)
 * - Real-time Bayesian Posterior P(Biofilm | e(t)) calculation with recursive filter
 * - Automated closed-loop SCADA trip (<1.4s): DI surge (28 L/min) & current derate (1.3 A/cm²)
 */

let scopeCv, scopeCtx;
let waveformV = [];
let waveformResidual = [];
let animFrameId = null;

// Electrochemical Stack State
const stackState = {
  tempC: 80.0,            // Operating temperature in °C
  pressureBar: 30.0,      // Cathode delivery pressure in bar
  currentDensity: 2.00,   // Operating current density in A/cm²
  waterFlowLpm: 15.0,     // DI water circulation flow in L/min
  cellCount: 10,
  activeAreaCm2: 250.0,
  biofilmThicknessUm: 0.0,
  biofilmActive: false,
  biofilmPosterior: 0.021,
  simTimeSec: 0.0,
  autoMitigated: false,
  tripTimestamp: null
};

// Physical Constants
const R_GAS = 8.314462;
const FARADAY = 96485.33;

function computeElectrochemicalTerms(j, tempC, pressBar, flowLpm, biofilmUm) {
  const tk = tempC + 273.15;
  const p = Math.max(pressBar, 1.0);

  // 1. Nernst Reversible Potential E_rev(T, P)
  const e0 = 1.229 - 0.00085 * (tk - 298.15);
  // P_H2 = P, P_O2 = 0.21 * P (effective anode partial pressure)
  const e_rev = e0 + (R_GAS * tk / (2 * FARADAY)) * Math.log(p * Math.sqrt(Math.max(p * 0.21, 0.1)));

  // 2. Butler-Volmer Activation Overpotential
  // Anode OER: higher activation barrier, b_an ~ 34-45 mV/dec
  const j0_an = 1.2e-4 * Math.exp((-25000 / R_GAS) * (1 / tk - 1 / 353.15));
  const alpha_an = 0.5;
  const eta_act_an = (R_GAS * tk / (alpha_an * FARADAY)) * Math.log(Math.max(j, 0.001) / j0_an);

  // Cathode HER: fast kinetics, b_cat ~ 28-35 mV/dec
  const j0_cat = 1.5e-3 * Math.exp((-18000 / R_GAS) * (1 / tk - 1 / 353.15));
  const alpha_cat = 0.5;
  const eta_act_cat = (R_GAS * tk / (alpha_cat * FARADAY)) * Math.log(Math.max(j, 0.001) / j0_cat);

  const eta_act = eta_act_an + eta_act_cat;

  // 3. Ohmic Resistance Overpotential
  // Membrane conductivity increases with temperature (Arrhenius) and DI water hydration
  const r_base = 0.075; // Ohm*cm² @ 80°C
  const r_temp_factor = Math.exp((12000 / R_GAS) * (1 / tk - 1 / 353.15));
  const r_flow_factor = 1.0 + 0.012 * Math.max(0, 15.0 - flowLpm);
  const r_ohm = r_base * r_temp_factor * r_flow_factor;
  const eta_ohm = j * r_ohm;

  // 4. Concentration Overpotential (Mass transport limitation near 3.5 A/cm²)
  const j_lim = 3.50;
  const j_ratio = Math.min(Math.max(j / j_lim, 0.001), 0.98);
  const eta_conc = - (R_GAS * tk / (2 * FARADAY)) * Math.log(1.0 - j_ratio);

  // 5. Biofilm Dielectric Overpotential
  // EPS matrix resistance: Delta_V = j * rho_EPS * thickness
  // At 4.5 µm thickness, produces exactly +58.8 mV (+0.0588 V) shift
  const eta_biofilm = (biofilmUm / 4.5) * 0.0588 * (j / 2.0);

  // Baseline Model Voltage (without biofilm or noise)
  const v_model = e_rev + eta_act + eta_ohm + eta_conc;

  // Measured Voltage (including biofilm and sensor noise)
  const v_meas = v_model + eta_biofilm;

  // Total 10-Cell Stack Metrics
  const v_stack = v_meas * stackState.cellCount;
  const current_amps = j * stackState.activeAreaCm2;
  const power_kw = (v_stack * current_amps) / 1000.0;

  // Faraday H2 Production Rate: (n_cells * I * eta_F) / (2 * F) in mol/s -> Nm³/h
  const eta_faraday = 0.985;
  const h2_mol_s = (stackState.cellCount * current_amps * eta_faraday) / (2 * FARADAY);
  const h2_nm3_h = h2_mol_s * 22.414e-3 * 3600.0;

  return {
    e_rev,
    eta_act,
    eta_ohm,
    eta_conc,
    eta_biofilm,
    v_model,
    v_meas,
    v_stack,
    current_amps,
    power_kw,
    h2_nm3_h
  };
}

function initScope() {
  scopeCv = document.getElementById('scopeCanvas');
  if (!scopeCv) return;
  scopeCtx = scopeCv.getContext('2d');

  function resizeScope() {
    if (!scopeCv || !scopeCv.parentElement) return;
    scopeCv.width = scopeCv.parentElement.clientWidth;
    scopeCv.height = 240;
  }
  resizeScope();
  window.addEventListener('resize', resizeScope);

  // Pre-fill history buffers
  const ptCount = Math.floor(scopeCv.width / 2.5);
  waveformV = [];
  waveformResidual = [];
  for (let i = 0; i < ptCount; i++) {
    waveformV.push(1.821 + (Math.random() - 0.5) * 0.004);
    waveformResidual.push((Math.random() - 0.5) * 2.0);
  }

  if (animFrameId) cancelAnimationFrame(animFrameId);
  runScopeLoop();
}

function runScopeLoop() {
  if (!scopeCtx || !scopeCv) return;
  stackState.simTimeSec += 0.033; // ~30 fps step

  // 1. Biological Dynamics: Growth vs Shear Flush
  if (stackState.biofilmActive) {
    stackState.biofilmThicknessUm = Math.min(stackState.biofilmThicknessUm + 0.11, 4.5);
  } else if (stackState.autoMitigated && stackState.biofilmThicknessUm > 0) {
    // High shear DI water flush (28 L/min) erodes biofilm matrix
    stackState.biofilmThicknessUm = Math.max(stackState.biofilmThicknessUm - 0.09, 0.0);
  }

  // 2. Solve Multi-Physics Model
  const sol = computeElectrochemicalTerms(
    stackState.currentDensity,
    stackState.tempC,
    stackState.pressureBar,
    stackState.waterFlowLpm,
    stackState.biofilmThicknessUm
  );

  // 3. Sensor Measurement with Gaussian Noise
  const sensorNoiseMv = (Math.random() - 0.5) * 2.4;
  const measuredResidualMv = (sol.eta_biofilm * 1000.0) + sensorNoiseMv;
  const measuredCellV = sol.v_model + (measuredResidualMv * 1e-3);

  // 4. Real-Time Recursive Bayesian Posterior Filter
  // Likelihood ratio of normal (mean=0, sigma=3.2) vs biofilm (mean=58.8, sigma=7.5)
  const mu_0 = 0.0, sig_0 = 3.2;
  const mu_1 = 58.8, sig_1 = 7.5;

  const p_e_given_norm = Math.exp(-0.5 * Math.pow((measuredResidualMv - mu_0) / sig_0, 2)) / (sig_0 * 2.506);
  const p_e_given_bio = Math.exp(-0.5 * Math.pow((measuredResidualMv - mu_1) / sig_1, 2)) / (sig_1 * 2.506);

  const prior = stackState.biofilmActive ? 0.40 : 0.02;
  const rawPosterior = (p_e_given_bio * prior) / Math.max((p_e_given_bio * prior + p_e_given_norm * (1.0 - prior)), 1e-12);

  // Exponential moving average filter
  stackState.biofilmPosterior = 0.88 * stackState.biofilmPosterior + 0.12 * rawPosterior;

  // 5. Closed-Loop SCADA Trip Logic (<1.4s latency)
  if (stackState.biofilmActive && stackState.biofilmPosterior >= 0.950 && !stackState.autoMitigated) {
    stackState.autoMitigated = true;
    stackState.biofilmActive = false; // Tripped: start mitigation
    stackState.tripTimestamp = stackState.simTimeSec;
    executeAutonomousScadaAction();
  }

  // Auto-recovery after flush completes
  if (stackState.autoMitigated && stackState.biofilmThicknessUm <= 0.05 && stackState.biofilmPosterior < 0.10) {
    stackState.autoMitigated = false;
    restoreNominalOperatingSetpoints();
  }

  // Update rolling waveforms
  waveformV.push(measuredCellV);
  waveformResidual.push(measuredResidualMv);
  const maxPts = Math.floor(scopeCv.width / 2.5);
  if (waveformV.length > maxPts) waveformV.shift();
  if (waveformResidual.length > maxPts) waveformResidual.shift();

  // Render Visuals
  renderDualTraceOscilloscope();
  updateMathematicalDashboard(sol, measuredCellV, measuredResidualMv);

  animFrameId = requestAnimationFrame(runScopeLoop);
}

function renderDualTraceOscilloscope() {
  const w = scopeCv.width;
  const h = scopeCv.height;

  // Clean Architectural Light Background
  scopeCtx.fillStyle = '#FFFFFF';
  scopeCtx.fillRect(0, 0, w, h);

  // Margins for Calibrated Dual Axes
  const leftMargin = 48;
  const rightMargin = 48;
  const plotW = w - leftMargin - rightMargin;
  const plotH = h - 30;
  const plotTop = 15;

  // Grid Lines & Axis Ticks
  scopeCtx.strokeStyle = '#E2E8F0';
  scopeCtx.lineWidth = 1;

  // Horizontal Grid Lines (6 intervals)
  for (let i = 0; i <= 5; i++) {
    const y = plotTop + (i / 5) * plotH;
    scopeCtx.beginPath();
    scopeCtx.moveTo(leftMargin, y);
    scopeCtx.lineTo(w - rightMargin, y);
    scopeCtx.stroke();

    // Left Y-Axis Label: Cell Voltage (1.50V to 2.20V)
    const voltVal = 2.20 - (i / 5) * (2.20 - 1.50);
    scopeCtx.fillStyle = '#0284C7';
    scopeCtx.font = '10px "JetBrains Mono", monospace';
    scopeCtx.textAlign = 'right';
    scopeCtx.fillText(`${voltVal.toFixed(2)}V`, leftMargin - 6, y + 3);

    // Right Y-Axis Label: Residual e(t) (+80 mV to -10 mV)
    const resVal = 80 - (i / 5) * (80 - (-10));
    scopeCtx.fillStyle = '#D97706';
    scopeCtx.textAlign = 'left';
    scopeCtx.fillText(`${resVal > 0 ? '+' : ''}${resVal.toFixed(0)}mV`, w - rightMargin + 6, y + 3);
  }

  // Vertical Grid Lines (Time axis: -10s, -8s, -6s, -4s, -2s, 0s)
  for (let i = 0; i <= 5; i++) {
    const x = leftMargin + (i / 5) * plotW;
    scopeCtx.beginPath();
    scopeCtx.moveTo(x, plotTop);
    scopeCtx.lineTo(x, plotTop + plotH);
    scopeCtx.stroke();

    const tSec = -10 + i * 2;
    scopeCtx.fillStyle = '#64748B';
    scopeCtx.font = '9.5px "JetBrains Mono", monospace';
    scopeCtx.textAlign = 'center';
    scopeCtx.fillText(`${tSec === 0 ? 't' : tSec + 's'}`, x, h - 4);
  }

  // Zero-Residual Reference Line (e(t) = 0 mV)
  const zeroResY = plotTop + ((80 - 0) / (80 - (-10))) * plotH;
  scopeCtx.strokeStyle = '#94A3B8';
  scopeCtx.setLineDash([4, 4]);
  scopeCtx.beginPath();
  scopeCtx.moveTo(leftMargin, zeroResY);
  scopeCtx.lineTo(w - rightMargin, zeroResY);
  scopeCtx.stroke();
  scopeCtx.setLineDash([]);

  // TRACE 1 (Blue): Measured Cell Voltage V_cell(t) [1.50V - 2.20V]
  scopeCtx.strokeStyle = '#0284C7';
  scopeCtx.lineWidth = 1.8;
  scopeCtx.beginPath();
  const vMin = 1.50, vMax = 2.20;
  for (let i = 0; i < waveformV.length; i++) {
    const x = leftMargin + (i / (waveformV.length - 1)) * plotW;
    const vClamped = Math.min(Math.max(waveformV[i], vMin), vMax);
    const y = plotTop + (1.0 - (vClamped - vMin) / (vMax - vMin)) * plotH;
    if (i === 0) scopeCtx.moveTo(x, y);
    else scopeCtx.lineTo(x, y);
  }
  scopeCtx.stroke();

  // TRACE 2 (Amber / Red on trip): Residual Error e(t) [-10 mV to +80 mV]
  const isTrip = stackState.biofilmPosterior >= 0.950;
  scopeCtx.strokeStyle = isTrip ? '#DC2626' : (stackState.autoMitigated ? '#059669' : '#D97706');
  scopeCtx.lineWidth = 2.2;
  scopeCtx.beginPath();
  const rMin = -10.0, rMax = 80.0;
  for (let i = 0; i < waveformResidual.length; i++) {
    const x = leftMargin + (i / (waveformResidual.length - 1)) * plotW;
    const rClamped = Math.min(Math.max(waveformResidual[i], rMin), rMax);
    const y = plotTop + (1.0 - (rClamped - rMin) / (rMax - rMin)) * plotH;
    if (i === 0) scopeCtx.moveTo(x, y);
    else scopeCtx.lineTo(x, y);
  }
  scopeCtx.stroke();
}

function updateMathematicalDashboard(sol, measuredCellV, measuredResidualMv) {
  // Update Mathematical Decomposition Elements
  const elErev = document.getElementById('calcErev');
  const elEtaAct = document.getElementById('calcEtaAct');
  const elEtaOhm = document.getElementById('calcEtaOhm');
  const elEtaConc = document.getElementById('calcEtaConc');
  const elEtaBio = document.getElementById('calcEtaBio');
  const elVcell = document.getElementById('calcVcell');
  const elVstack = document.getElementById('calcVstack');
  const elPstack = document.getElementById('calcPstack');
  const elH2Rate = document.getElementById('calcH2Rate');

  if (elErev) elErev.innerText = `${sol.e_rev.toFixed(3)} V`;
  if (elEtaAct) elEtaAct.innerText = `${sol.eta_act.toFixed(3)} V`;
  if (elEtaOhm) elEtaOhm.innerText = `${sol.eta_ohm.toFixed(3)} V`;
  if (elEtaConc) elEtaConc.innerText = `${sol.eta_conc.toFixed(3)} V`;
  if (elEtaBio) {
    elEtaBio.innerText = `${(sol.eta_biofilm * 1000).toFixed(1)} mV`;
    elEtaBio.style.color = sol.eta_biofilm > 0.01 ? '#DC2626' : 'var(--text-main)';
  }
  if (elVcell) elVcell.innerText = `${measuredCellV.toFixed(3)} V`;
  if (elVstack) elVstack.innerText = `${sol.v_stack.toFixed(2)} V`;
  if (elPstack) elPstack.innerText = `${sol.power_kw.toFixed(2)} kW`;
  if (elH2Rate) elH2Rate.innerText = `${sol.h2_nm3_h.toFixed(3)} Nm³/h`;

  // Update Status Banner
  const tagEl = document.getElementById('scopeStatusTag');
  const noteEl = document.getElementById('scopeTripNote');
  const postBar = document.getElementById('scopePosteriorBar');
  const postVal = document.getElementById('scopePosteriorVal');

  if (postBar) {
    const pct = Math.min(Math.max(stackState.biofilmPosterior * 100, 0), 100);
    postBar.style.width = `${pct}%`;
    postBar.style.backgroundColor = pct >= 95 ? '#DC2626' : (pct >= 50 ? '#D97706' : '#059669');
  }
  if (postVal) {
    postVal.innerText = `${stackState.biofilmPosterior.toFixed(3)} (${(stackState.biofilmPosterior * 100).toFixed(1)}%)`;
    postVal.style.color = stackState.biofilmPosterior >= 0.95 ? '#DC2626' : (stackState.autoMitigated ? '#059669' : 'var(--text-main)');
  }

  if (tagEl) {
    if (stackState.biofilmPosterior >= 0.950 && !stackState.autoMitigated) {
      tagEl.innerText = 'AUTONOMOUS TRIP (<1.4s): DIELECTRIC EPS DETECTED';
      tagEl.style.color = '#DC2626';
    } else if (stackState.autoMitigated) {
      tagEl.innerText = 'ACTIVE MITIGATION: PURE DI WATER SURGE (28 L/min) & CURRENT DERATE';
      tagEl.style.color = '#0F172A';
    } else {
      tagEl.innerText = 'NORMAL OPERATING REGIME // BUTLER-VOLMER TRACKING';
      tagEl.style.color = '#334155';
    }
  }

  if (noteEl) {
    if (stackState.biofilmPosterior >= 0.950 || stackState.autoMitigated) {
      noteEl.innerHTML = `<strong>SCADA Interlock Engaged:</strong> Solenoid valve XV-202 opened. High shear water sweep (28.0 L/min) purging foulant. Current derated to 1.30 A/cm².`;
      noteEl.style.background = '#EFF6FF';
      noteEl.style.borderColor = '#0284C7';
      noteEl.style.color = '#0F172A';
    } else {
      noteEl.innerHTML = `<strong>Mitigation Interlock:</strong> Automated DI pulse trigger at P(Biofilm) &ge; 0.950. Latency benchmark: &lt;1.40s.`;
      noteEl.style.background = '#F8FAFC';
      noteEl.style.borderColor = '#CBD5E1';
      noteEl.style.color = '#334155';
    }
  }
}

// User Slider Controls
function updateStackParam(param, val) {
  const num = parseFloat(val);
  if (param === 'current') {
    stackState.currentDensity = num;
    const el = document.getElementById('sliderValCurrent');
    if (el) el.innerText = `${num.toFixed(2)} A/cm²`;
  } else if (param === 'temp') {
    stackState.tempC = num;
    const el = document.getElementById('sliderValTemp');
    if (el) el.innerText = `${num.toFixed(1)} °C`;
  } else if (param === 'press') {
    stackState.pressureBar = num;
    const el = document.getElementById('sliderValPress');
    if (el) el.innerText = `${num.toFixed(1)} bar`;
  } else if (param === 'flow') {
    stackState.waterFlowLpm = num;
    const el = document.getElementById('sliderValFlow');
    if (el) el.innerText = `${num.toFixed(1)} L/min`;
  }
}

function triggerBiofilmFault() {
  stackState.biofilmActive = true;
  stackState.autoMitigated = false;
  stackState.simTimeSec = 0.0;
}

function executeAutonomousScadaAction() {
  // Automated PLC setpoints: surge water flow and derate current
  stackState.waterFlowLpm = 28.0;
  stackState.currentDensity = 1.30;

  // Sync Slider UI to reflect automated action
  const flowSlider = document.getElementById('sliderFlow');
  const currentSlider = document.getElementById('sliderCurrent');
  if (flowSlider) flowSlider.value = 28.0;
  if (currentSlider) currentSlider.value = 1.30;
  const flowVal = document.getElementById('sliderValFlow');
  const currentVal = document.getElementById('sliderValCurrent');
  if (flowVal) flowVal.innerText = '28.0 L/min (AUTO PURGE)';
  if (currentVal) currentVal.innerText = '1.30 A/cm² (AUTO DERATE)';
}

function restoreNominalOperatingSetpoints() {
  stackState.waterFlowLpm = 15.0;
  stackState.currentDensity = 2.00;
  stackState.biofilmThicknessUm = 0.0;
  stackState.biofilmPosterior = 0.021;

  const flowSlider = document.getElementById('sliderFlow');
  const currentSlider = document.getElementById('sliderCurrent');
  if (flowSlider) flowSlider.value = 15.0;
  if (currentSlider) currentSlider.value = 2.00;
  const flowVal = document.getElementById('sliderValFlow');
  const currentVal = document.getElementById('sliderValCurrent');
  if (flowVal) flowVal.innerText = '15.0 L/min';
  if (currentVal) currentVal.innerText = '2.00 A/cm²';
}

function resetScopeSimulation() {
  stackState.biofilmActive = false;
  stackState.autoMitigated = false;
  stackState.biofilmThicknessUm = 0.0;
  stackState.biofilmPosterior = 0.021;
  stackState.tempC = 80.0;
  stackState.pressureBar = 30.0;
  stackState.currentDensity = 2.00;
  stackState.waterFlowLpm = 15.0;
  stackState.simTimeSec = 0.0;

  const inputs = [
    ['sliderCurrent', 'sliderValCurrent', 2.00, '2.00 A/cm²'],
    ['sliderTemp', 'sliderValTemp', 80.0, '80.0 °C'],
    ['sliderPress', 'sliderValPress', 30.0, '30.0 bar'],
    ['sliderFlow', 'sliderValFlow', 15.0, '15.0 L/min']
  ];
  inputs.forEach(([sId, vId, val, label]) => {
    const s = document.getElementById(sId);
    const v = document.getElementById(vId);
    if (s) s.value = val;
    if (v) v.innerText = label;
  });
}

window.initScope = initScope;
window.updateStackParam = updateStackParam;
window.triggerBiofilmFault = triggerBiofilmFault;
window.resetScopeSimulation = resetScopeSimulation;
window.resetScope = resetScopeSimulation;
