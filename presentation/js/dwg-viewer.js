/**
 * HYDROSENSE AutoCAD 2025 ISO A1 Engineering Drawing Viewer
 * Renders authentic monochrome CAD plots plotted directly via AutoCAD 2025 accoreconsole:
 * - Sheet 1: DWG-PEM-H2-COMMERCIAL-PLANT-002 Rev 5 (General Arrangement, Elevation & Plot Plan)
 * - Sheet 2: DWG-PEM-H2-2026-001 Rev 1 (10-Cell Industrial Stack & Flow Field DFM)
 * Features smooth high-performance pan/zoom, sheet switching, and direct vector downloads.
 */

let activeDwgSheet = 'plant';
let dwgZoom = 1.0;
let dwgPanX = 0;
let dwgPanY = 0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

const dwgSheets = {
  plant: {
    id: 'DWG-PEM-H2-COMMERCIAL-PLANT-002',
    rev: 'Rev 5',
    title: '100 TPD Turnkey Facility: General Arrangement, Elevation & Process Plot Plan',
    img: 'assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png',
    pdf: 'assets/DWG-PEM-H2-COMMERCIAL-PLANT-002.pdf',
    dxf: 'assets/DWG-PEM-H2-COMMERCIAL-PLANT-002.dxf',
    standards: 'Domestic: The Factories Act, 1948 (Sec 32/33) | PESO SMPV Rules 2016 | IS 800:2007 | Global Export: ASME BPVC Sec VIII Div 1 & 3 | ASME B31.12 | API 618 | ISO 22734',
    scale: '1:50 / 1:25 @ ISO A1 (841 × 594 mm)'
  },
  stack: {
    id: 'DWG-PEM-H2-2026-001',
    rev: 'Rev 1',
    title: '10-Cell Industrial PEM Electrolyzer Stack: General Arrangement & Bipolar Plate Details',
    img: 'assets/DWG-PEM-H2-2026-001_Preview.png',
    pdf: 'assets/DWG-PEM-H2-2026-001.pdf',
    dxf: 'assets/DWG-PEM-H2-2026-001.dxf',
    standards: 'Domestic: National Green H2 Mission BIS Draft | IS 16087 | Global Export: ISO 22734 | ASTM B265 Gr. 2 Ti | ASME B16.5 | ASME Sec. IX',
    scale: '1:1 / 15:1 @ ISO A1 (841 × 594 mm)'
  }
};

function initAutoCADViewer() {
  const container = document.getElementById('autocadSvgContainer');
  if (!container) return;

  renderDwgViewer();
}

function renderDwgViewer() {
  const container = document.getElementById('autocadSvgContainer');
  if (!container) return;

  const cur = dwgSheets[activeDwgSheet];

  container.innerHTML = `
    <div class="dwg-workbench-frame" style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
      <!-- Sub-Header / Sheet Switcher Ribbon -->
      <div style="background: #F8FAFC; border-bottom: 1px solid #E2E8F0; padding: 8px 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; gap: 6px; align-items: center;">
          <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase;">CAD Sheet:</span>
          <button class="btn btn-secondary ${activeDwgSheet === 'plant' ? 'active' : ''}" style="padding: 4px 10px; font-size: 11.5px; ${activeDwgSheet === 'plant' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchDwgSheet('plant')">
            Sheet 1: 100 TPD Plant GA (DWG-002)
          </button>
          <button class="btn btn-secondary ${activeDwgSheet === 'stack' ? 'active' : ''}" style="padding: 4px 10px; font-size: 11.5px; ${activeDwgSheet === 'stack' ? 'background: #0284C7; color: #FFFFFF; border-color: #0284C7;' : 'background: #FFFFFF;'}" onclick="switchDwgSheet('stack')">
            Sheet 2: 10-Cell Stack DFM (DWG-001)
          </button>
        </div>

        <div style="display: flex; gap: 8px; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
          <a href="${cur.pdf}" download class="btn btn-secondary" style="background: #FFFFFF; padding: 4px 10px; font-size: 11px; color: #0F172A;" title="Download Vector PDF plotted from AutoCAD 2025">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Plotted PDF
          </a>
          <a href="${cur.dxf}" download class="btn btn-secondary" style="background: #FFFFFF; padding: 4px 10px; font-size: 11px; color: #0F172A;" title="Download Native AutoCAD R2018 DXF">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            AutoCAD DXF
          </a>
          <button class="btn btn-secondary" style="background: #FFFFFF; padding: 4px 10px; font-size: 11px;" onclick="window.showModal('${cur.img}', '${cur.id} ${cur.rev} (300 DPI)')">
            Inspect 300 DPI
          </button>
        </div>
      </div>

      <!-- Interactive Canvas Viewport (Pure Light Architectural Plot Background) -->
      <div id="dwgViewport" style="position: relative; width: 100%; height: 600px; background: #F1F5F9; overflow: hidden; cursor: grab; display: flex; justify-content: center; align-items: center;">
        <div id="dwgSheetWrapper" style="transform-origin: center center; transition: transform 0.08s ease-out; user-select: none;">
          <img id="dwgSheetImg" src="${cur.img}" alt="${cur.title}" style="max-width: 100%; max-height: 580px; display: block; box-shadow: 0 4px 12px rgba(0,0,0,0.12); border: 1px solid #94A3B8; background: #FFFFFF; pointer-events: none;">
        </div>

        <!-- Floating Pan/Zoom HUD Overlay -->
        <div style="position: absolute; bottom: 12px; right: 12px; background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(4px); border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; display: flex; align-items: center; gap: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
          <span id="dwgZoomDisplay" style="font-weight: 700; color: #334155; min-width: 44px; text-align: right;">100%</span>
          <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="zoomAutoCAD(1.25)">+</button>
          <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="zoomAutoCAD(0.8)">&minus;</button>
          <button class="btn btn-secondary" style="padding: 2px 7px; font-size: 11px;" onclick="resetAutoCADZoom()">1:1</button>
        </div>

        <!-- Floating Drawing Standards Pill -->
        <div style="position: absolute; top: 12px; left: 12px; background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(4px); border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 10px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #475569; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
          <strong>${cur.id} ${cur.rev}</strong> | ${cur.scale}
        </div>
      </div>

      <!-- Statutory Notes & Schedule Strip -->
      <div style="background: #F8FAFC; border-top: 1px solid #E2E8F0; padding: 6px 14px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #64748B; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
        <span><strong>Statutory Standards:</strong> ${cur.standards}</span>
        <span>Jadavpur University | Chemical Engineering</span>
      </div>
    </div>
  `;

  attachDwgPanZoomListeners();
}

function switchDwgSheet(sheetKey) {
  if (dwgSheets[sheetKey]) {
    activeDwgSheet = sheetKey;
    dwgZoom = 1.0;
    dwgPanX = 0;
    dwgPanY = 0;
    renderDwgViewer();
  }
}

function updateDwgTransform() {
  const wrapper = document.getElementById('dwgSheetWrapper');
  const disp = document.getElementById('dwgZoomDisplay');
  if (wrapper) {
    wrapper.style.transform = `translate(${dwgPanX}px, ${dwgPanY}px) scale(${dwgZoom})`;
  }
  if (disp) {
    disp.textContent = Math.round(dwgZoom * 100) + '%';
  }
}

function zoomAutoCAD(factor) {
  dwgZoom = Math.min(Math.max(0.5, dwgZoom * factor), 4.0);
  updateDwgTransform();
}

function resetAutoCADZoom() {
  dwgZoom = 1.0;
  dwgPanX = 0;
  dwgPanY = 0;
  updateDwgTransform();
}

function attachDwgPanZoomListeners() {
  const vp = document.getElementById('dwgViewport');
  if (!vp) return;

  vp.addEventListener('mousedown', e => {
    isPanning = true;
    startPanX = e.clientX - dwgPanX;
    startPanY = e.clientY - dwgPanY;
    vp.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', e => {
    if (!isPanning) return;
    dwgPanX = e.clientX - startPanX;
    dwgPanY = e.clientY - startPanY;
    updateDwgTransform();
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      if (vp) vp.style.cursor = 'grab';
    }
  });

  vp.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    zoomAutoCAD(factor);
  }, { passive: false });
}

window.initAutoCADViewer = initAutoCADViewer;
window.switchDwgSheet = switchDwgSheet;
window.zoomAutoCAD = zoomAutoCAD;
window.resetAutoCADZoom = resetAutoCADZoom;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAutoCADViewer);
} else {
  setTimeout(initAutoCADViewer, 200);
}
