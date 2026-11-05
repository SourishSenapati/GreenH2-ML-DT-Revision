/**
 * HYDROSENSE™ Live In-Operando Oscilloscope Simulator
 * High-frequency digital twin tracking stack voltage residuals in Pure Architectural Light Mode.
 * Demonstrates Bayesian biofilm detection and autonomous closed-loop DI water flush mitigation.
 */

let scopeCv, scopeCtx;
let waveform = [];
let faultActive = false;
let animFrameId = null;

function initScope() {
  scopeCv = document.getElementById('scopeCanvas');
  if (!scopeCv) return;
  scopeCtx = scopeCv.getContext('2d');

  function resizeScope() {
    if (!scopeCv) return;
    scopeCv.width = scopeCv.parentElement.clientWidth - 40;
    scopeCv.height = 200;
  }
  resizeScope();
  window.addEventListener('resize', resizeScope);

  // Initialize waveform buffer
  waveform = [];
  const ptCount = Math.floor(scopeCv.width / 4);
  for (let i = 0; i < ptCount; i++) {
    waveform.push((Math.random() - 0.5) * 6.0);
  }

  if (animFrameId) cancelAnimationFrame(animFrameId);
  runScope();
}

function runScope() {
  if (!scopeCtx || !scopeCv) return;

  // Pure White Background
  scopeCtx.fillStyle = '#FFFFFF';
  scopeCtx.fillRect(0, 0, scopeCv.width, scopeCv.height);

  // Architectural Slate Drafting Grid
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

  // Generate real-time voltage residual sample
  let res = (Math.random() - 0.5) * 6.0;
  if (faultActive) {
    res += 58.8 + (Math.random() - 0.5) * 10.0; // Injected EPS biofilm dielectric shift
  }
  waveform.push(res);
  if (waveform.length > scopeCv.width / 4) waveform.shift();

  // Oscilloscope Trace Line (Cobalt Blue in normal, Red on trip)
  scopeCtx.strokeStyle = faultActive ? '#DC2626' : '#0284C7';
  scopeCtx.lineWidth = 2.0;
  scopeCtx.beginPath();
  const zeroY = scopeCv.height * 0.70;
  for (let i = 0; i < waveform.length; i++) {
    const x = i * 4;
    const y = zeroY - waveform[i] * 1.6;
    if (i === 0) scopeCtx.moveTo(x, y);
    else scopeCtx.lineTo(x, y);
  }
  scopeCtx.stroke();

  // Zero Reference Line
  scopeCtx.strokeStyle = '#CBD5E1';
  scopeCtx.setLineDash([4, 4]);
  scopeCtx.beginPath();
  scopeCtx.moveTo(0, zeroY);
  scopeCtx.lineTo(scopeCv.width, zeroY);
  scopeCtx.stroke();
  scopeCtx.setLineDash([]);

  animFrameId = requestAnimationFrame(runScope);
}

function triggerBiofilmFault() {
  faultActive = true;
  const statusTag = document.getElementById('scopeStatusTag');
  if (statusTag) {
    statusTag.innerText = 'CRITICAL TRIP: BIOFILM DETECTED (<1.4s)';
    statusTag.style.color = '#DC2626';
  }
  const cur = document.getElementById('scopeCurrent');
  if (cur) {
    cur.innerText = '1.30 A/cm² (Throttled)';
    cur.style.color = '#D97706';
  }
  const flow = document.getElementById('scopeFlow');
  if (flow) {
    flow.innerText = '28.0 L/min (Pulse)';
    flow.style.color = '#059669';
  }
  const post = document.getElementById('scopePosterior');
  if (post) {
    post.innerText = '0.988 (Trip Engaged)';
    post.style.color = '#DC2626';
  }
  const note = document.getElementById('scopeTripNote');
  if (note) {
    note.innerHTML = '<strong>Closed-Loop Safety Engaged:</strong> Current throttled to 1.3 A/cm²; pure DI flush pulsed at 28 L/min to dislodge biofilm without chemical biocides.';
    note.style.borderColor = 'var(--status-alert-border)';
    note.style.background = 'var(--status-alert-bg)';
    note.style.color = 'var(--status-alert)';
  }
}

function resetScope() {
  faultActive = false;
  const statusTag = document.getElementById('scopeStatusTag');
  if (statusTag) {
    statusTag.innerText = 'NORMAL OPERATING REGIME';
    statusTag.style.color = 'var(--status-pass)';
  }
  const cur = document.getElementById('scopeCurrent');
  if (cur) {
    cur.innerText = '2.00 A/cm²';
    cur.style.color = 'var(--accent)';
  }
  const flow = document.getElementById('scopeFlow');
  if (flow) {
    flow.innerText = '15.0 L/min';
    flow.style.color = 'var(--accent)';
  }
  const post = document.getElementById('scopePosterior');
  if (post) {
    post.innerText = '0.021';
    post.style.color = 'var(--status-pass)';
  }
  const note = document.getElementById('scopeTripNote');
  if (note) {
    note.innerHTML = '<strong>Autonomous Mitigation Ready:</strong> Proactive DI pulse engages at P(F) &ge; 0.95.';
    note.style.borderColor = 'var(--status-pass-border)';
    note.style.background = 'var(--status-pass-bg)';
    note.style.color = 'var(--status-pass)';
  }
}

window.initScope = initScope;
window.triggerBiofilmFault = triggerBiofilmFault;
window.resetScope = resetScope;

