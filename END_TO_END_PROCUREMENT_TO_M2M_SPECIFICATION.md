# End-to-End Procurement-to-M2M Industrial Execution Specification
## From Sourcing & Raw Materials to Factory Acceptance Testing (FAT) & Machine-to-Machine (M2M) Commissioning
**Document Ref:** PEM-H2-M2M-2026-001  
**Project Ref:** 1:1 Commercial PEM Green Hydrogen Facility & AERO-PURE Water Systems  
**Author:** Sourish Senapati (Department of Chemical Engineering, Jadavpur University)  
**Standard Mission:** ASD-STM100 Industrial Commercial Deployment

---

## 1. 6-Stage Industrial Execution Gate Architecture

To transition this technology from CAD models and AI algorithms into operational plant machinery ready for commercial sale, every stage follows strict engineering gate reviews:

```
[STAGE 1: PROCUREMENT] ──► [STAGE 2: FABRICATION] ──► [STAGE 3: STACK ASSEMBLY]
       │                          │                            │
       ▼                          ▼                            ▼
• Mill Test Certs (3.1)    • Progressive Stamping       • Robotic Pick & Place
• SA-240 316L / Ti Gr.2    • CNC Milling (Ra ≤ 0.8µm)   • Synchronized Multi-Spindle
• Traceability Tracking    • ASME Sec IX Weld Inspection• 15 kN Belleville Torque
       │                          │                            │
       └──────────────────────────┴────────────────────────────┘
                                  │
                                  ▼
[STAGE 4: FACTORY ACCEPTANCE TESTING (FAT)]
  • He Mass Spectrometer Leak Test (< 1x10^-6 mbar·L/s)
  • ASME UG-99 Hydrostatic Proof Test (39.0 bar)
  • Hi-Pot Insulation Test (2.5 kV AC)
                                  │
                                  ▼
[STAGE 5: SITE RIGGING & FOUNDATION TIE-IN]
  • Modular Skid Anchor Grouting (IS 800)
  • ASME B31.3 Piping Tie-In & Nitrogen Flushing
  • 5000A DC Busway Bolted Joint Resistance Check
                                  │
                                  ▼
[STAGE 6: M2M INTEGRATION & DIGITAL TWIN COMMISIONING]
  • OPC UA / MQTT Telemetry Stream to Edge Box
  • 24/7 PIBT Virtual Sensing & Biofilm Baseline Calibration
  • Closed-Loop SCADA Throttling Handshake
```

---

## 2. Stage-by-Stage Detailed Engineering Execution

### Stage 1: Procurement & Raw Material Traceability
- **Titanium Sheets:** ASTM B265 Grade 2 cold-rolled titanium sheets ($1.00 \pm 0.02\ \text{mm}$) sourced with EN 10204 Type 3.1 Mill Test Certificates (MTC) verifying interstitial elements ($\text{O} \le 0.25\%$, $\text{Fe} \le 0.30\%$, $\text{C} \le 0.08\%$).
- **End Plate Steel:** ASME SA-240 Type 316L hot-rolled, annealed and pickled (HRAP) plate, $35.0\ \text{mm}$ thickness, with ultrasonic flaw examination per ASTM A578 Level B.
- **Copper Busways:** ASTM B187 C11000 ETP Copper ($100 \times 12\ \text{mm}$ bars), $99.9\%$ purity, minimum electrical conductivity $101.0\%\ \text{IACS}$.
- **Fasteners:** ISO 4014 Grade 8.8 high-tensile studs with zinc-flake / PTFE coating to prevent galling against stainless steel tapped holes.
- **Membranes:** Nafion 212 perfluorosulfonic acid (PFSA) rolls with certified dry thickness $50.8 \pm 2.0\ \mu\text{m}$ and equivalent weight (EW) $1100\ \text{g/mol}$.

### Stage 2: Component Fabrication & Quality Control (QC)
- **Progressive Stamping of Bipolar Plates:**
  - 400-ton hydraulic press with tungsten carbide progressive die set.
  - Channel geometry inspected on Hexagon Coordinate Measuring Machine (CMM): depth $0.80 \pm 0.02\ \text{mm}$, pitch $2.00 \pm 0.01\ \text{mm}$, draft angle $7.0^\circ \pm 0.5^\circ$.
  - 100% optical surface inspection for micro-cracks or burrs along sealing lands.
- **ASME Pressure Vessel Fabrication (V-300 & ST-800):**
  - Shell plate pre-rolled on 4-roll CNC bending machine; longitudinal and circumferential seams welded via GTAW root + GMAW fill.
  - 100% Radiographic Examination (RT-1) per ASME Section VIII UW-11(a).
  - Post-Weld Heat Treatment (PWHT) applied if required by wall thickness / service.

### Stage 3: Cleanroom Stack Assembly (ISO Class 7)
- **Environment:** Positive-pressure cleanroom (ISO 14644-1 Class 7) with continuous deionized air ionization to prevent particulate deposition.
- **Assembly Sequence:**
  1. Base plate positioning on alignment fixture.
  2. Anode C11000 current collector with gold-plated contact face placed.
  3. Cell repeating units (Bipolar Plate $\to$ EPDM Gasket $\to$ Titanium PTL $\to$ CCM $\to$ Titanium PTL $\to$ Gasket) assembled using optical dual-camera centering pins ($\pm 0.05\ \text{mm}$ alignment).
  4. Top clamping plate lowered and 8x M12 tie-rods hand-threaded with Belleville spring disc stacks.
- **Synchronized Torque Tightening:** Automated 8-spindle multi-nut runner tightens perimeter bolts in diagonal cross-pattern in 3 stages: $5.0\ \text{kN} \to 10.0\ \text{kN} \to \mathbf{15.0\ \text{kN}}$ per bolt. Total stack compression: **$120.0\ \text{kN}$** ($1.50\ \text{MPa}$ uniform seal face pressure).

### Stage 4: Factory Acceptance Testing (FAT) Protocols
Every commercial machine must pass three non-negotiable FAT trials prior to factory dispatch:
1. **Helium Mass Spectrometer Leak Test:**
   - Stack pressurized with 99.999% Helium to $32.0\ \text{bar}$. Enclosed in vacuum chamber.
   - Acceptable external leak rate: **$< 1.0 \times 10^{-6}\ \text{mbar}\cdot\text{L/s}$**.
   - Internal cross-over leak rate between anode and cathode: **$< 5.0 \times 10^{-5}\ \text{mbar}\cdot\text{L/s}$**.
2. **ASME UG-99 Hydrostatic Proof Test:**
   - V-300 Knockout Drum, piping headers, and stack water chambers filled with deaerated DI water and pressurized to **$39.00\ \text{bar(g)}$** for 30 minutes with zero observable pressure drop.
3. **Dielectric Hi-Pot Insulation Test:**
   - 2,500 V AC applied between current collectors and structural grounding frame for 60 seconds. Leakage current must remain **$< 1.5\ \text{mA}$** (verifying complete isolation).

### Stage 5: Site Installation & Rigging (Civil / Mechanical Tie-In)
- **Skid Rigging:** Single-crane lift utilizing the 4 rated lifting lugs on the ISMB 300 mezzanine frame (total skid dry weight: $6,450\ \text{kg}$).
- **Foundation Grouting:** Skid lowered onto reinforced concrete foundation pad ($13.0 \times 9.0 \times 0.35\ \text{m}$); baseplates secured with 16x M24 anchor bolts and packed with non-shrink epoxy grout.
- **Piping Tie-Ins:** Flanged connections to plant battery limits (DN50 DI water supply, DN80 O2 vent, DN15 350-bar hydrogen product header).
- **Power Tie-In:** 11 kV grid cable landed on TR-901 medium-voltage cable box; 400V auxiliary panel powered.

### Stage 6: M2M Integration & Digital Twin Calibration
- **Edge Box Deployment:** Industrial IPC (Siemens Microbox / NVIDIA Jetson) installed in PLC-1001 cabinet.
- **M2M Protocol:** Native **OPC UA (IEC 62541)** server configured on Siemens S7-1500F PLC, streaming telemetry packets to edge digital twin at 100 Hz:
  ```json
  {
    "timestamp": "2026-11-05T22:00:00.000Z",
    "stack_id": "EL-201",
    "current_A": 500.2,
    "voltage_total_V": 19.34,
    "cell_voltages_V": [1.934, 1.932, 1.935, 1.933, 1.936, 1.931, 1.934, 1.935, 1.933, 1.937],
    "inlet_temp_C": 60.1,
    "outlet_temp_C": 68.4,
    "pressure_cathode_barg": 30.05,
    "pressure_anode_barg": 2.12,
    "flow_di_water_Lpm": 15.02,
    "conductivity_uS_cm": 0.048,
    "pibt_predicted_V": 19.338,
    "voltage_residual_mV": 2.1,
    "bayesian_biofilm_prob": 0.012,
    "status": "NOMINAL_OPTIMAL"
  }
  ```
- **Closed-Loop Safety Handshake:** If the edge PIBT model flags $e(t) > +18.5\ \text{mV}$ with Bayesian confidence $> 0.95$, an immediate discrete Modbus / Profinet telegram commands the PLC to reduce current by 35% and boost water flush, preventing stack damage and logging the mitigation event.

---

## 3. Commercial Machinery Capitalization & Client Warranty
- **Standard Manufacturer Warranty:** 30,000 Operating Hours or 4 Years full performance warranty when paired with the continuous AERO-H2 Digital Twin monitoring subscription.
- **Turnkey Machine Price (5 MW Commercial Facility):** \$3,850,000 USD (Includes 4x 50-Cell Stacks, Mezzanine Skid, Knockout Drum, TSA Dryers, Diaphragm Compressor, Power Center, and SCADA Suite).
- **AERO-PURE Water Polishing Skid Standalone Price:** \$85,000 USD (Delivers 15 L/min ASTM D1193 Type 1 ultrapure water).
