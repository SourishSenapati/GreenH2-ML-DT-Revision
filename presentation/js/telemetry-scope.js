/**
 * HYDROSENSE™ In-Operando Digital Twin Oscilloscope (Butler-Volmer Physics Solver)
 * Simulates real-time electrochemistry of a 10-cell industrial stack (250 cm² active area):
 * - Nernst Reversible Potential E_rev(T, P)
 * - Anode & Cathode Butler-Volmer Activation Overpotentials
 * - Ohmic Electrolyte & PTL Resistance Overpotential
 * - In-operando dielectric EPS biofilm voltage residual injection
 * - Real-time Bayesian Posterior P(Biofilm | e(t)) calculation
 * - Closed-loop autonomous DI surge & SCADA current derating (<1.4s trip)
 */

let scopeCv, scopeCtx;
let waveformV = [];
let waveformResidual = [];
let animFrameId = null;

// Electrochemical Stack State
const stackState = {
  tempK: 353.15,          // 80°C
  pressureBar: 30.0,      // 30 bar cathode
  currentDensity: 2.00,   // A/cm²
  waterFlowLpm: 15.0,     // L/min
  cellCount: 10,
  activeAreaCm2: 250.0,
  biofilmThicknessUm: 0.0,
  biofilmActive: false,
  biofilmPosterior: 0.021,
  simTimeSec: 0.0,
  autoMitigated: false
};

// Physical Constants
const R_GAS = 8.314462;
const FARADAY = 96485.33;

function computeButlerVolmerCellVoltage(j, T, P) {
  // 1. Nernst Reversible Potential
  const e0 = 1.229 - 0.00085 * (T - 298.15);
  const e_rev = e0 + (R_GAS * T / (2 * FARADAY)) * Math.log(P * Math.sqrt(P * 0.21));

  // 2. Butler-Volmer Activation Overpotential (Tafel slope b_an = 34.2 mV/dec, j0 = 1.2e-4 A/cm2)
  const j0_an = 1.2e-4;
  const eta_act_an = 0.0342 * Math.log(Math.max(j, 0.001) / j0_an);
  const j0_cat = 1.5e-3;
  const eta_act_cat = 0.0285 * Math.log(Math.max(j, 0.001) / j0_cat);

  // 3. Ohmic Resistance Overpotential (Membrane conductivity sigma_m = 0.14 S/cm @ 80C)
  const r_ohm = 0.075; // Ohm*cm2
  const eta_ohm = j * r_ohm;

  // 4. Concentration Overpotential
  const j_lim = 3.5;
  const eta_conc = - (R_GAS * T / (2 * FARADAY)) * Math.log(Math.max(1.0 - j / j_lim, 0.01));

  return e_rev + eta_act_an + eta_act_cat + eta_ohm + eta_conc;
}

function initScope() {
  scopeCv = document.getElementById('scopeCanvas');
  if (!scopeCv) return;
  scopeCtx = scopeCv.getContext('2d');

  function resizeScope() {
    if (!scopeCv || !scopeCv.parentElement) return;
    scopeCv.width = scopeCv.parentElement.clientWidth - 40;
    scopeCv.height = 200;
  }
  resizeScope();
  window.addEventListener('resize', resizeScope);

  // Pre-fill buffer
  const ptCount = Math.floor(scopeCv.width / 3);
  waveformV = [];
  waveformResidual = [];
  for (let i = 0; i < ptCount; i++) {
    waveformV.push(1.82 + (Math.random() - 0.5) * 0.008);
    waveformResidual.push((Math.random() - 0.5) * 2.5);
  }

  if (animFrameId) cancelAnimationFrame(animFrameId);
  runScopeLoop();
}

function runScopeLoop() {
  if (!scopeCtx || !scopeCv) return;
  stackState.simTimeSec += 0.033; // ~30 fps step

  // 1. Calculate Baseline Voltage
  const v_phys = computeButlerVolmerCellVoltage(stackState.currentDensity, stackState.tempK, stackState.pressureBar);

  // 2. Dynamics of Biofilm Injection
  let delta_v_biofilm = 0.0;
  if (stackState.biofilmActive) {
    stackState.biofilmThicknessUm = Math.min(stackState.biofilmThicknessUm + 0.12, 4.5);
    // Dielectric resistance rise: Delta_V = j * (rho_biofilm * thickness)
    delta_v_biofilm = 0.0588 * (1.0 - Math.exp(-stackState.simTimeSec * 1.8)); // climbs to +58.8 mV
  } else if (stackState.autoMitigated && stackState.biofilmThicknessUm > 0) {
    stackState.biofilmThicknessUm = Math.max(stackState.biofilmThicknessUm - 0.25, 0.0);
    delta_v_biofilm = 0.0588 * (stackState.biofilmThicknessUm / 4.5);
  }

  // 3. Sensor Noise & Stack Total
  const sensorNoiseMv = (Math.random() - 0.5) * 2.8; // mV
  const measuredResidualMv = delta_v_biofilm * 1000.0 + sensorNoiseMv;
  const measuredCellV = v_phys + (measuredResidualMv * 1e-3);

  // 4. Bayesian Posterior Computation
  // Gaussian likelihood: p(e(t) | normal) vs p(e(t) | biofilm)
  const mu_normal = 0.0, sigma_normal = 3.5;
  const mu_biofilm = 58.8, sigma_biofilm = 8.0;

  const p_norm = Math.exp(-0.5 * Math.pow((measuredResidualMv - mu_normal) / sigma_normal, 2)) / (sigma_normal * 2.5);
  const p_bio = Math.exp(-0.5 * Math.pow((measuredResidualMv - mu_biofilm) / sigma_biofilm, 2)) / (sigma_biofilm * 2.5);
  const p_prior = stackState.biofilmActive ? 0.35 : 0.01;
  const posterior = (p_bio * p_prior) / ((p_bio * p_prior) + (p_norm * (1.0 - p_prior)) + 1e-9);
  
  // Smooth posterior filter
  stackState.biofilmPosterior = 0.85 * stackState.biofilmPosterior + 0.15 * posterior;

  // 5. Autonomous Mitigation Trigger (<1.4s trip)
  if (stackState.biofilmActive && stackState.biofilmPosterior >= 0.95 && !stackState.autoMitigated) {
    stackState.autoMitigated = true;
    stackState.biofilmActive = false; // Engages flush
    executeAutonomousScadaAction();
  }

  // Update buffers
  waveformV.push(measuredCellV);
  waveformResidual.push(measuredResidualMv);
  const maxPts = Math.floor(scopeCv.width / 3);
  if (waveformV.length > maxPts) waveformV.shift();
  if (waveformResidual.length > maxPts) waveformResidual.shift();

  // Draw Canvas
  renderScopeCanvas();

  // Update Sensor Readouts
  updateSensorHud(measuredCellV, measuredResidualMv);

  animFrameId = requestAnimationFrame(runScopeLoop);
}

function renderScopeCanvas() {
  // Pure White Drafting Background
  scopeCtx.fillStyle = '#FFFFFF';
  scopeCtx.fillRect(0, 0, scopeCv.width, scopeCv.height);

  // Slate 200 Hairline Grid
  scopeCtx.strokeStyle = '#E2E8F0';
  scopeCtx.lineWidth = 1;
  for (let y = 25; y < scopeCv.height; y += 30) {
    scopeCtx.beginPath();
    scopeCtx.moveTo(0, y);
    scopeCtx.lineTo(scopeCv.width, y);
    scopeCtx.stroke();
  }
  for (let x = 40; x < scopeCv.width; x += 50) {
    scopeCtx.beginPath();
    scopeCtx.moveTo(x, 0);
    scopeCtx.lineTo(x, scopeCv.height);
    scopeCtx.stroke();
  }

  const zeroY = scopeCv.height * 0.72;

  // Draw Baseline Zero Line
  scopeCtx.strokeStyle = '#CBD5E1';
  scopeCtx.setLineDash([4, 4]);
  scopeCtx.beginPath();
  scopeCtx.moveTo(0, zeroY);
  scopeCtx.lineTo(scopeCv.width, zeroY);
  scopeCtx.stroke();
  scopeCtx.setLineDash([]);

  // Draw Voltage Residual Trace
  const isTrip = stackState.biofilmPosterior >= 0.95;
  scopeCtx.strokeStyle = isTrip ? '#DC2626' : (stackState.autoMitigated ? '#059669' : '#0284C7');
  scopeCtx.lineWidth = 2.0;
  scopeCtx.beginPath();
  for (let i = 0; i < waveformResidual.length; i++) {
    const x = i * 3;
    const y = zeroY - waveformResidual[i] * 1.8;
    if (i === 0) scopeCtx.moveTo(x, y);
    else scopeCtx.lineTo(x, y);
  }
  scopeCtx.stroke();
}

function updateSensorHud(cellV, resMv) {
  const cdEl = document.getElementById('hudCurrentDensity') || document.getElementById('scopeCurrent');
  const flowEl = document.getElementById('hudDiFlow') || document.getElementById('scopeFlow');
  const pressEl = document.getElementById('hudStackPress') || document.getElementById('scopePress');
  const postEl = document.getElementById('hudBioPosterior') || document.getElementById('scopePosterior');
  const tagEl = document.getElementById('scopeStatusTag');

  if (cdEl) cdEl.innerText = `${stackState.currentDensity.toFixed(2)} A/cm²`;
  if (flowEl) flowEl.innerText = `${stackState.waterFlowLpm.toFixed(1)} L/min`;
  if (pressEl) pressEl.innerText = `${stackState.pressureBar.toFixed(1)} bar`;
  if (postEl) {
    postEl.innerText = stackState.biofilmPosterior.toFixed(3);
    postEl.style.color = stackState.biofilmPosterior >= 0.95 ? '#DC2626' : (stackState.autoMitigated ? '#059669' : 'var(--text-main)');
  }
  if (tagEl) {
    if (stackState.biofilmPosterior >= 0.95 && !stackState.autoMitigated) {
      tagEl.innerText = 'AUTONOMOUS TRIP: DIELECTRIC BIOFILM DETECTED (<1.4s)';
      tagEl.style.color = '#DC2626';
    } else if (stackState.autoMitigated) {
      tagEl.innerText = 'PROACTIVE MITIGATION ENGAGED: PURE DI WATER SURGE (28 L/min)';
      tagEl.style.color = '#0F172A';
    } else {
      tagEl.innerText = 'NORMAL OPERATING REGIME // BUTLER-VOLMER TRACKING';
      tagEl.style.color = '#334155';
    }
  }
}

function triggerBiofilmFault() {
  stackState.biofilmActive = true;
  stackState.autoMitigated = false;
  stackState.simTimeSec = 0.0;
}

function executeAutonomousScadaAction() {
  // SCADA trips DI purge solenoid and derates power
  stackState.waterFlowLpm = 28.0;       // 15 -> 28 L/min purge
  stackState.currentDensity = 1.30;     // 2.0 -> 1.3 A/cm² derate

  setTimeout(() => {
    // Return to nominal after flush
    stackState.waterFlowLpm = 15.0;
    stackState.currentDensity = 2.00;
    stackState.autoMitigated = false;
    stackState.biofilmThicknessUm = 0.0;
  }, 4500);
}

function resetScopeSimulation() {
  stackState.biofilmActive = false;
  stackState.autoMitigated = false;
  stackState.biofilmThicknessUm = 0.0;
  stackState.biofilmPosterior = 0.021;
  stackState.currentDensity = 2.00;
  stackState.waterFlowLpm = 15.0;
  stackState.pressureBar = 30.0;
  stackState.simTimeSec = 0.0;
}

window.initScope = initScope;
window.triggerBiofilmFault = triggerBiofilmFault;
window.resetScopeSimulation = resetScopeSimulation;
window.resetScope = resetScopeSimulation;
