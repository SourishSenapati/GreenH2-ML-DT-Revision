/**
 * HYDROSENSE™ Machine Learning Digital Twin Workbench
 * Handles 5-phase interactive neural model exploration, dynamic KaTeX mathematical proof rendering,
 * and high-resolution figure inspection.
 */

const mlCatalog = {
  'fig1': {
    src: 'assets/Fig1_XGBoost_Feature_Importance.png',
    phase: 'PHASE 1: VIRTUAL SENSING FEATURE ATTRIBUTION',
    title: 'XGBoost Feature Importance Attribution',
    desc: 'Gradient-boosted decision trees quantify catalyst kinetic degradation. Identifies Tafel slope ($b_{\\mathrm{an}}$: 34.2%), electrochemical active surface area (ECSA: 23.1%), and PTL porosity ($\\varepsilon$: 17.1%) as the top 3 physical drivers, capturing >74.4% of total variance.',
    math: '$$\\hat{y}_{\\mathrm{decay}} = \\sum_{k=1}^K f_k(\\mathbf{x}_i), \\quad f_k \\in \\mathcal{F}, \\quad R^2 = 0.9630$$',
    badges: '<span class="cred-tag highlight">R² = 0.963</span><span class="cred-tag">NRMSE = 0.046</span><span class="cred-tag">140 Experimental Datasets</span>'
  },
  'fig2': {
    src: 'assets/Fig2_Parity_Plot_Predicted_vs_Empirical.png',
    phase: 'PHASE 1: PARITY ACCURACY BENCHMARK',
    title: 'Catalyst Morphology & Kinetic Decay Parity Plot',
    desc: '140 galvanostatic experimental samples bounded within ±5% engineering and ±10% tolerance error bands. Achieves sub-microvolt precision with an RMSE of 1.9110 µV/h and maximum residual < 1.4 µV/h.',
    math: '$$\\mathrm{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (\\dot{V}_{\\mathrm{emp},i} - \\dot{V}_{\\mathrm{pred},i})^2} = 1.9110\\ \\mu\\mathrm{V/h}$$',
    badges: '<span class="cred-tag highlight">R² = 0.9630</span><span class="cred-tag">RMSE = 1.911 µV/h</span><span class="cred-tag">Max Residual &lt; 1.4 µV/h</span>'
  },
  'fig3': {
    src: 'assets/Fig3_PIBT_Time_Series_Degradation_Forecast.png',
    phase: 'PHASE 2: PROGNOSTIC TIME-SERIES TRACKING',
    title: 'Physics-Informed Bayesian Transformer (PIBT)',
    desc: 'Butler-Volmer thermodynamic loss constraint prevents catastrophic forecast divergence during dynamic renewable load cycling (0 to 100 hours). Achieves 31.4% error reduction over unconstrained LSTMs (NRMSE = 0.054).',
    math: '$$\\mathcal{L}_{\\mathrm{total}} = \\mathcal{L}_{\\mathrm{MSE}} + \\lambda_{\\mathrm{BV}} \\| V_{\\mathrm{pred}} - V_{\\mathrm{Butler-Volmer}} \\|_2^2$$',
    badges: '<span class="cred-tag highlight">31.4% Error Reduction</span><span class="cred-tag">NRMSE = 0.054</span><span class="cred-tag">95% Bayesian Confidence Band</span>'
  },
  'fig4': {
    src: 'assets/Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png',
    phase: 'PHASE 3: DECOUPLED FAULT DETECTION',
    title: 'Decoupled Bayesian Biofilm Anomaly Detection',
    desc: 'Decouples voltage residuals $e(t)$ from electrochemical decay. Detects dielectric bacterial biofilm accumulation in &lt;1.4s, triggering immediate SCADA throttling (2.0 &rarr; 1.3 A/cm²) and pure DI water pulse (15 &rarr; 28 L/min), eliminating toxic biocides.',
    math: '$$P(\\mathcal{F}_{\\mathrm{biofilm}} \\mid e(t)) = \\frac{p(e(t) \\mid \\mathcal{F}_{\\mathrm{biofilm}}) P(\\mathcal{F}_{\\mathrm{biofilm}})}{p(e(t))} > 0.95$$',
    badges: '<span class="cred-tag highlight">F1-Score = 0.990</span><span class="cred-tag">Trip Latency &lt; 1.4s</span><span class="cred-tag">&Delta;V = +58.8 mV Detection</span>'
  },
  'fig6': {
    src: 'assets/Fig6_Cross_Sectoral_Validation.png',
    phase: 'PHASE 4: MULTI-SCALE VALIDATION',
    title: 'Cross-Sectoral Generalization & Multi-Scale Validation',
    desc: 'Maintains consistent >93% accuracy across diverse operational scales: 25 cm² single cells (R² = 0.963), 250 cm² 10-cell industrial stacks (R² = 0.948), commercial chlor-alkali brine cells (R² = 0.932), and multi-MW 100 TPD pilot arrays (R² = 0.956).',
    math: '$$\\mathrm{Accuracy} = 1 - \\mathrm{NRMSE} > 92.9\\% \\quad \\forall \\text{ Industrial Domains}$$',
    badges: '<span class="cred-tag highlight">&gt;93% Generalization</span><span class="cred-tag">4 Heavy Industrial Scales</span><span class="cred-tag">Zero Overfitting</span>'
  }
};

let activeMl = 'fig1';

function setMlFig(key, el) {
  if (!mlCatalog[key]) return;
  activeMl = key;
  document.querySelectorAll('#proprietary-ml .tab-item').forEach(b => b.classList.remove('active'));
  if (el) el.classList.add('active');

  const item = mlCatalog[key];
  const img = document.getElementById('mlDisplayImg');
  if (img) img.src = item.src;
  
  const phaseLabel = document.getElementById('mlPhaseLabel');
  if (phaseLabel) phaseLabel.innerText = item.phase;
  
  const mainTitle = document.getElementById('mlMainTitle');
  if (mainTitle) mainTitle.innerText = item.title;
  
  const desc = document.getElementById('mlDescription');
  if (desc) desc.innerHTML = item.desc;
  
  const mathEq = document.getElementById('mlMathEq');
  if (mathEq) mathEq.innerHTML = item.math;
  
  const badges = document.getElementById('mlBadges');
  if (badges) badges.innerHTML = item.badges;

  if (window.renderMathInElement) {
    const card = document.getElementById('mlDetailCard');
    if (card) {
      window.renderMathInElement(card, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false}
        ]
      });
    }
  }
}

function zoomCurrentMl() {
  if (window.showModal && mlCatalog[activeMl]) {
    window.showModal(mlCatalog[activeMl].src, mlCatalog[activeMl].title);
  }
}

window.setMlFig = setMlFig;
window.zoomCurrentMl = zoomCurrentMl;
