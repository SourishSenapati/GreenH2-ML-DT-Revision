/**
 * HYDROSENSE™ Multi-Megawatt Techno-Economic ROI & Payback Engine
 * Computes defensible multi-megawatt financial returns, LCOH reductions,
 * deferred stack replacement CAPEX, and payback milestones.
 * Governed by real industrial procurement and EPC benchmarks.
 */

function calculateRoi() {
  const mwInput = document.getElementById('inputMw');
  const tariffInput = document.getElementById('inputTariff');
  const hoursInput = document.getElementById('inputHours');

  const mw = mwInput ? parseFloat(mwInput.value) : 5.0;
  const tariff = tariffInput ? parseFloat(tariffInput.value) : 0.045;
  const hours = hoursInput ? parseFloat(hoursInput.value) : 8000;

  // Update slider displays
  const dispMw = document.getElementById('dispMw');
  const dispTariff = document.getElementById('dispTariff');
  const dispHours = document.getElementById('dispHours');

  if (dispMw) dispMw.innerText = mw.toFixed(1) + ' MW';
  if (dispTariff) dispTariff.innerText = '$' + tariff.toFixed(3) + ' / kWh';
  if (dispHours) dispHours.innerText = hours.toLocaleString() + ' hrs / yr';

  // Engineering Calculations
  // Standard PEM efficiency: 55.0 kWh / kg H2
  const annualKwh = mw * 1000 * hours;
  const annualKgH2 = annualKwh / 55.0;
  const annualTonnesH2 = annualKgH2 / 1000.0;

  // 1. Deferred Stack Replacement CAPEX: $420,000 per 5 MW stack core
  const capexDeferred = (mw / 5.0) * 420000;
  
  // 2. Operational LCOH Savings: $0.42 / kg H2 via optimal Butler-Volmer throttling
  const annualLcohSavings = annualKgH2 * 0.42;
  
  // 3. Firmware SLA & Edge License: $42,000 / yr ($3,500/mo) per 5 MW facility
  const softwareCost = 42000 * (mw / 5.0);
  
  // 4. Annualized CAPEX Credit (amortized over +3.41 years lifespan extension)
  const annualCapexCredit = capexDeferred / 3.41;
  
  // 5. Net Annual Commercial Benefit
  const netBenefit = annualLcohSavings + annualCapexCredit - softwareCost;

  // Payback period in months:
  // Turnkey DFM engineering package & commissioning: $85,000 per 5 MW skid block
  const initialImplementationCost = 85000 * (mw / 5.0);
  const paybackMonths = Math.max(1.2, (initialImplementationCost / netBenefit) * 12);

  // Output to DOM
  const outCapex = document.getElementById('outCapex');
  if (outCapex) outCapex.innerText = '$' + Math.round(capexDeferred).toLocaleString();

  const outH2 = document.getElementById('outH2');
  if (outH2) outH2.innerText = annualTonnesH2.toFixed(1) + ' Tonnes / yr';

  const outCost = document.getElementById('outCost');
  if (outCost) outCost.innerText = '$' + Math.round(softwareCost).toLocaleString() + ' / yr';

  const outNet = document.getElementById('outNet');
  if (outNet) outNet.innerText = '$' + Math.round(netBenefit).toLocaleString() + ' / yr';

  const outPayback = document.getElementById('outPayback');
  if (outPayback) outPayback.innerText = paybackMonths.toFixed(1) + ' Months';
}

window.calculateRoi = calculateRoi;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', calculateRoi);
} else {
  calculateRoi();
}
