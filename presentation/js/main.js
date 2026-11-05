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
  }
});

// Bootstrap application on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  console.log("HYDROSENSE™ Industrial Operating System initialized.");
  
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
        {left: '$', right: '$', display: false}
      ]
    });
  }
});

window.showModal = showModal;
window.hideModal = hideModal;
