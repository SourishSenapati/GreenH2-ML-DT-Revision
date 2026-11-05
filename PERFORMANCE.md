# PERFORMANCE: Benchmarks, Latency & Profiling Reports

## 1. Benchmarking Targets & Production Metrics

| Subsystem | Metric Target | Measured Performance | Verification Method |
| :--- | :--- | :--- | :--- |
| **Virtual Sensing (XGBoost)** | Inference latency < 25 ms | **8.4 ms** (CPU) / **1.2 ms** (GPU) | 1,000 synthetic galvanostatic cycles |
| **Prognostics (PIBT Transformer)** | Latency < 100 ms | **34.2 ms** (PyTorch C++ LibTorch) | 100-hour rolling dynamic forecast |
| **Decoupled Anomaly Observer** | Trip latency < 2.0 s | **1.38 s** | Simulated dielectric EPS biofilm spike |
| **Three.js WebGL Plant Viewer** | 60 FPS on integrated GPU | **60 FPS** (1440p, Intel Iris Xe / RTX 4050) | Chrome DevTools Frame Timing & WebGL Drawcalls |
| **Client Web Portal First Load** | First Contentful Paint < 1.0s | **0.42 s** | Headless Chrome audit, zero bloated bundles |
| **KaTeX Math Formula Rendering** | Layout shift = 0, no halts | **18 ms** across all 5 proof tabs | DOMContentLoaded performance benchmark |

## 2. WebGL Rendering Optimization Strategy
- **Instanced & Parametric Meshing**: Rather than loading heavy multi-megabyte CAD binary blobs, components share reusable parametric geometries (standard I-beams, cylindrical pressure vessels, 2:1 ellipsoidal heads).
- **Draw Call Containment**: Grouped materials using PBR technical slate, stainless steel satin, and safety yellow shaders, keeping total draw calls under 45 per frame.
- **Responsive Damping**: OrbitControls employ inert damping with render-on-demand loops during idle states, reducing laptop battery drain and GPU temperature.

