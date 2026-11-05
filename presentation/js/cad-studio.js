/**
 * HYDROSENSE™ SolidWorks CAD Studio Engine
 * Manages 7 interactive engineering viewports, metadata switching, and high-res modal zoom.
 */

const cadGallery = {
  'master': { 
    src: 'assets/PEM_Commercial_Plant_Catalog_Render.png', 
    title: 'SolidWorks Master Assembly (11 Subsystems • 100 TPD Turnkey Facility)' 
  },
  'iso': { 
    src: 'assets/PEM_Commercial_Plant_Isometric_HD.png', 
    title: 'SolidWorks 3D Isometric View (300 DPI Studio Photography)' 
  },
  'plotplan': { 
    src: 'assets/PEM_Commercial_Plant_PlotPlan_HD.png', 
    title: 'AutoCAD Commercial Plant Plot Plan & Equipment Schedules' 
  },
  'front': { 
    src: 'assets/PEM_Commercial_Plant_FrontElevation_HD.png', 
    title: 'Front Elevation & Mezzanine Piping Layout (ISMB 300)' 
  },
  'stack': { 
    src: 'assets/PEM_10Cell_Industrial_Stack_Catalog_Render.png', 
    title: '10-Cell Industrial B-Rep Stack Assembly (ASTM B265 Gr. 2 Titanium)' 
  },
  'cell': { 
    src: 'assets/PEM_Cell_Exploded_Architecture_Diagram.png', 
    title: 'Exploded Single-Cell Repeat Unit Architecture & Transport Phenomena' 
  },
  'dwg': { 
    src: 'assets/DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png', 
    title: 'AutoCAD 2025 ISO A1 Engineering Shop Drawing (ASME Sec VIII Div 1 & B31.3)' 
  }
};

let activeCad = 'master';

function setCadView(key, el) {
  if (!cadGallery[key]) return;
  activeCad = key;
  document.querySelectorAll('#cad-studio .tab-item').forEach(b => b.classList.remove('active'));
  if (el) el.classList.add('active');
  const img = document.getElementById('cadDisplayImg');
  if (img) img.src = cadGallery[key].src;
}

function zoomCurrentCad() {
  if (window.showModal && cadGallery[activeCad]) {
    window.showModal(cadGallery[activeCad].src, cadGallery[activeCad].title);
  }
}

window.setCadView = setCadView;
window.zoomCurrentCad = zoomCurrentCad;

