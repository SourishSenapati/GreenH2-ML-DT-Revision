/**
 * HYDROSENSE™ Client Application Bootstrap
 * Connects modular CAD, ML Workbench, ROI Calculator, Telemetry Oscilloscope,
 * and Lightbox Modals.
 */

// --- Modal Lightbox (Pure Light Architectural Frame) ---
function showModal(src, title = 'High-Resolution Asset Inspection') {
  const modal = document.getElementById('imageModal');
  const img = document.getElementById('modalImg');
  const titleEl = document.getElementById('modalTitle');
  if (!modal || !img || !titleEl) return;

  img.src = src;
  titleEl.innerText = title;
  modal.classList.add('active');
}

function hideModal(e) {
  if (e && e.target && e.target.id !== 'imageModal' && !e.target.classList.contains('close-btn')) {
    return;
  }
  const modal = document.getElementById('imageModal');
  if (modal) modal.classList.remove('active');
}

// Global Escape listener for modals
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('imageModal');
    if (modal && modal.classList.contains('active')) {
      modal.classList.remove('active');
    }
    const stdModal = document.getElementById('standardsModal');
    if (stdModal && stdModal.classList.contains('active')) {
      stdModal.classList.remove('active');
    }
  }
});

function openStandardFromCurrentSubsystem() {
  const mapping = {
    'STR-100': 'factories_act',
    'EL-200': 'diff_p',
    'V-300': 'peso_smpv',
    'V-400': 'iec',
    'DR-500': 'iso_purity',
    'WTR-600': 'factories_act',
    'CMP-700': 'asme',
    'ST-800': 'peso_smpv',
    'ELE-900': 'iec',
    'PLC-1000': 'iec',
    'PIP-1100': 'asme'
  };
  const activeId = window.selectedSubsystem || 'EL-200';
  const targetKey = mapping[activeId] || 'asme';
  if (typeof openStandardModal === 'function') {
    openStandardModal(targetKey);
  }
}

// Bootstrap application on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  console.log("HYDROSENSE™ Industrial Operating System initialized.");
  
  if (typeof initPlant3DScene === 'function') {
    initPlant3DScene();
  }

  if (typeof calculateRoi === 'function') {
    calculateRoi();
  }

  if (typeof initScope === 'function') {
    initScope();
  }

  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      throwOnError: false,
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
      ignoredClasses: ['btn', 'tier-price-tag', 'slider-num', 'data-val', 'metric-num', 'no-math', 'metric-sub', 'hero-lead']
    });
  }
});

window.showModal = showModal;
window.hideModal = hideModal;
window.openStandardFromCurrentSubsystem = openStandardFromCurrentSubsystem;


