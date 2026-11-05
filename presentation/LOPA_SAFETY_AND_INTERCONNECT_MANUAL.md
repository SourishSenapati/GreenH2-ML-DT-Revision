# Industrial LOPA Safety, Piping & Electrical Interconnection Manual
**Commercial Turnkey Green Hydrogen Production Plant (100 TPD Scale)**
**Document Ref:** PEM-H2-LOPA-ELEC-2026-001  
**Lead Process Safety & Electrical Engineer:** Sourish Senapati (Jadavpur University)  
**Safety Integrity Standard:** IEC 61508 / IEC 61511 (SIL-2) • ASME B31.3 • NFPA 2 Hydrogen Technologies Code • IEEE 519

---

## 1. Layers of Protection Analysis (LOPA) & Risk Reduction Architecture

In industrial green hydrogen facilities, catastrophic hazards (overpressure rupture, hydrogen deflagration, membrane cross-over, and electrical arc flash) are prevented through five independent, non-interfering **Independent Protection Layers (IPLs)**.

```
+-------------------------------------------------------------------------------------------------------+
|                                    5-TIER INDUSTRIAL LOPA ARCHITECTURE                                |
+-------------------------------------------------------------------------------------------------------+
|  IPL 5: PASSIVE MITIGATION (Physical Blast Walls, Flame Arrestors & 6.2m Elevated O2/H2 Vent Stacks) |
|  IPL 4: MECHANICAL RELIEF (ASME Section VIII Certified Angle PSVs & Dual Reverse-Buckling Rupture Discs)|
|  IPL 3: SAFETY INSTRUMENTED SYSTEM (SIL-2 Fail-Safe Siemens S7-1500F PLC & 2oo3 Voting Interlocks)    |
|  IPL 2: OPERATOR INTERVENTION & CRITICAL SCADA ALARMS (Pre-Alarm Annunciation & Visual Andon Tower)   |
|  IPL 1: BASIC PROCESS CONTROL SYSTEM (BPCS Closed-Loop PID Throttling & VFD Inverter Speed Control)   |
+-------------------------------------------------------------------------------------------------------+
                                        ▲
                         INITIATING CAUSE / PROCESS UPSET
```

### 1.1 Detailed LOPA Risk Reduction Matrix

| Hazard Node & Scenario | Unmitigated Frequency ($f_i$) | Cause & Mechanism | IPL-1: BPCS Control | IPL-2: Critical Alarms | IPL-3: SIS / SIL-2 Loop (ESD) | IPL-4: Mechanical Relief (PSV) | IPL-5: Passive Venting | Mitigated Event Frequency ($f_m$) | Risk Target Met? |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Node 1: V-300 Knockout Drum Overpressure** | $1.0 \times 10^{-1}$ /yr | Downstream control valve failure or DeOxo blockage | PIC-301 throttles stack current via 5000A rectifier ($\text{PFD} = 0.1$) | PAH-301 alarm triggers operator ramp-down ($\text{PFD} = 0.1$) | PSHH-301 trips ESD-001 in $<50\ \text{ms}$, opening blowdown valve XV-301 ($\text{PFD} = 0.01$) | PSV-301 (ASME Sec VIII, $34.5\ \text{bar}$ set, $100\%$ discharge) ($\text{PFD} = 0.01$) | Flame arrestor FA-301 & high-velocity discharge stack | $< 1.0 \times 10^{-6}$ /yr | **YES (SIL-2 Compliant)** |
| **Node 2: Membrane Hydrogen Cross-Over (Explosion)** | $5.0 \times 10^{-2}$ /yr | Membrane thinning or localized pinhole perforation | Dual thermal mass flow balancing $\Delta P \le 0.2\ \text{bar}$ | AIT-401 flags $\text{H}_2 \text{ in } \text{O}_2 > 1.0\ \text{vol}\%$ | ASHH-401 trips nitrogen purge XV-402 and opens vent XV-401 ($\text{PFD} = 0.005$) | Rupture Disc RD-401 ($5.0\ \text{bar}$) bursts to vent stack | Outdoor elevated vent stack ($6.2\ \text{m}$) discharging above roofline | $< 2.5 \times 10^{-7}$ /yr | **YES (Broadly Acceptable)** |
| **Node 3: 350-bar Diaphragm Compressor Rupture** | $2.0 \times 10^{-1}$ /yr | 3rd stage check valve failure or line blockage | Discharge pressure transmitter PIT-703 modulates VFD ($\text{PFD} = 0.1$) | PAH-703 alarm at $365\ \text{bar}$ | PSHH-703 trips 65 kW drive contactor in $<30\ \text{ms}$ ($\text{PFD} = 0.01$) | PSV-701 ($385\ \text{bar}$ set pressure, angle body) ($\text{PFD} = 0.01$) | Inert nitrogen blanketed secondary containment hood | $< 2.0 \times 10^{-6}$ /yr | **YES (SIL-2 Compliant)** |
| **Node 4: Dielectric Biofilm Ohmic Runaway** | $4.0 \times 10^{-1}$ /yr | Bacterial colonization in PTL causing voltage spikes | PIBT model detects residual $e(t) > +18.5\ \text{mV}$ | Automated SCADA current throttling: $2.0 \to 1.3\ \text{A/cm}^2$ | High-voltage interlock trips cell bank if $V_{\text{cell}} > 2.10\ \text{V}$ ($\text{PFD} = 0.01$) | None (Non-overpressure failure mode) | Automated DI water high-velocity flush ($28\ \text{L/min}$) | $< 4.0 \times 10^{-5}$ /yr | **YES (Mitigated)** |

---

## 2. Complete Physical Process Piping & Valved Headers (ASME B31.3)

Every pipe run in the commercial plant assembly is physically routed between equipment nozzles, featuring weld-neck flanges, forged elbow union hubs, and isolation valves:

```
                                  [ATMOSPHERIC DISCHARGE EL +6.20 m]
                                                 ▲
                                                 │ 3" SS316L Vent
                                          [FLAME ARRESTOR]
                                                 │
                                     ┌───────────┴───────────┐
                                     │   V-400 O2 CYCLONE    │
                                     │  DEGASSER (Dia 800)   │
                                     └───────────┬───────────┘
                                                 ▲ Anode 2-Phase Line
                                                 │ (DN50 Schedule 40)
                                                 │
+-------------------+                      +─────┴────────────────────────+
| TR-901 / REC-901  |===[5000A DC BUSWAYS] |   EL-200 ELECTROLYZER BAY    |
| 11kV TRANSFORMER  |                      |   (4x 50-Cell Multi-Stacks)  |
| & POWER CENTER    |                      +─────┬────────────────────────+
+-------------------+                            │
                                                 │ Cathode 2-Phase Line (DN40 Sch 80, 30 bar)
                                                 ▼
                                     ┌───────────────────────┐
                                     │ V-300 HP KO DRUM      │──[PSV-301]──► Relief Header
                                     │ (Dia 700, 30 bar ASME)│
                                     └───────────┬───────────┘
                                                 │ 30-bar Wet Gas (DN32 Sch 80)
                                                 ▼
                                     ┌───────────────────────┐
                                     │ DR-500 DEOXO + TSA    │ (<1 ppm O2, -70°C Dewpoint)
                                     └───────────┬───────────┘
                                                 │ 30-bar Dry Gas (DN25 Sch 80)
                                                 ▼
                                     ┌───────────────────────┐
                                     │ CMP-700 3-STAGE       │──[PSV-701]──► Vent
                                     │ DIAPHRAGM COMPRESSOR  │ (65 kW VFD)
                                     └───────────┬───────────┘
                                                 │ 350-bar Discharge (DN15 High-Pressure Tubing)
                                                 ▼
                                     ┌───────────────────────┐
                                     │ ST-800 4-TUBE BUFFER  │──[PSV-801]──► Dispensing Station
                                     │ STORAGE ARRAY (350bar)│
                                     └───────────────────────┘
```

### 2.1 Complete Pipe Schedule & Material Take-Off (MTO)

| Line ID | Service Description | Size / Rating | Material of Construction | Design P / T | Insulation / Tracing |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **L-101** | DI Water Pump Suction | DN50 (2") Class 150 | ASTM A312 TP316L Seamless | $2.5\ \text{bar} / 40^\circ\text{C}$ | Bare SS |
| **L-102** | DI Water Pump Discharge to PHE | DN40 (1.5") Class 300 | ASTM A312 TP316L Seamless | $35.0\ \text{bar} / 45^\circ\text{C}$ | Bare SS |
| **L-103** | Temperature-Regulated Stack Feed | DN40 (1.5") Class 300 | ASTM A312 TP316L Seamless | $32.0\ \text{bar} / 60^\circ\text{C}$ | $25\ \text{mm}$ Mineral Wool |
| **L-201** | Anode Two-Phase $\text{O}_2 + \text{H}_2\text{O}$ Header | DN50 (2") Class 150 | ASTM A312 TP316L Seamless | $3.5\ \text{bar} / 80^\circ\text{C}$ | $25\ \text{mm}$ Thermal Clad |
| **L-202** | Cyclone Gas Vent to Atmospheric Stack | DN80 (3") Class 150 | ASTM A312 TP316L Seamless | $1.5\ \text{bar} / 75^\circ\text{C}$ | Lightning Protected |
| **L-203** | Cyclone Bottom Water Return to Polisher | DN40 (1.5") Class 150 | ASTM A312 TP316L Seamless | $2.0\ \text{bar} / 70^\circ\text{C}$ | Bare SS |
| **L-301** | Cathode Wet $\text{H}_2 + \text{H}_2\text{O}$ Header | DN40 (1.5") Class 300 | ASTM A312 TP316L Seamless | $32.0\ \text{bar} / 80^\circ\text{C}$ | $30\ \text{mm}$ Mineral Wool |
| **L-302** | Knockout Drum Automatic Condensate Blowdown | DN25 (1") Class 300 | ASTM A312 TP316L Seamless | $30.0\ \text{bar} / 75^\circ\text{C}$ | Bare SS |
| **L-501** | 30-bar Wet Gas to Catalytic DeOxo | DN32 (1.25") Class 300 | ASTM A312 TP316L Seamless | $30.0\ \text{bar} / 70^\circ\text{C}$ | Bare SS |
| **L-502** | DeOxo Discharge to Twin TSA Towers | DN32 (1.25") Class 300 | ASTM A312 TP316L Seamless | $30.0\ \text{bar} / 95^\circ\text{C}$ | Heat Traced (Regeneration) |
| **L-701** | Purified $\text{H}_2$ Compressor Feed | DN25 (1") Class 300 | ASTM A312 TP316L Seamless | $30.0\ \text{bar} / 35^\circ\text{C}$ | Bare SS |
| **L-702** | Compressor High-Pressure 350-bar Discharge | DN15 (0.5") Class 2500 | ASTM A269 TP316L Heavy-Wall | $385.0\ \text{bar} / 45^\circ\text{C}$ | High-Pressure Cone & Thread |
| **L-801** | 350-bar Buffer Storage Manifold | DN15 (0.5") Class 2500 | ASTM A269 TP316L Heavy-Wall | $350.0\ \text{bar} / 40^\circ\text{C}$ | High-Pressure Cone & Thread |

---

## 3. Electrical Power, Instrumentation & SCADA Signal Architecture

### 3.1 5000A DC Heavy Copper Busway Distribution
- **Transformer (TR-901):** $11\ \text{kV} \pm 5\%$ primary, $400\ \text{V}$ secondary, 3.5 MVA oil-immersed step-down transformer (Dyn11 vector group).
- **Thyristor Rectifier (REC-901):** 12-pulse bridge rectifier delivering $0 \to 120\ \text{V DC}$ continuously adjustable at up to **$5,000\ \text{A DC}$** (Total Harmonic Distortion $\text{THD}_i < 3.0\%$ per IEEE 519).
- **Busways:** Dual solid C11000 ETP copper busbars ($100\ \text{mm} \times 12\ \text{mm}$ twin bars per polarity) routed along structural mezzanine supports on heavy GPO-3 polyester fiberglass insulator blocks.
- **Voltage Drop:** Less than $0.45\ \text{V}$ across the entire $12\ \text{m}$ busway length at full $5000\ \text{A}$ rating ($99.6\%$ electrical transmission efficiency).

### 3.2 24V DC Field Instrumentation & SIL-2 Signal Conduits
All field instruments are wired via shielded twisted pairs (STP) in perforated galvanized cable trays into the **Siemens S7-1500F Fail-Safe Automation Suite (PLC-1001)**:
- **Level Measurement:** Siemens SITRANS LR110 80GHz Radar Level Transmitters (LIT-301, LIT-401, LIT-601) with 4–20 mA HART output.
- **Pressure Measurement:** Siemens SITRANS P320 High-Accuracy Gauge & Differential Transmitters (PIT-201, PIT-301, PIT-501, PIT-701, PIT-801).
- **Safety Interlocks:** Dual-channel SIL-2 pressure switches (PSHH-301, PSHH-701) configured in 1oo2 fail-safe de-energize-to-trip topology.
- **Trace Gas Analyzers:** Thermal conductivity hydrogen-in-oxygen analyzer (AIT-401) continuously validating that $\text{H}_2 \text{ in } \text{O}_2 \le 1.0\ \text{vol}\%$.

---

## 4. Dual-Monetization Engine: AERO-PURE Water Ultrafiltration Skid (Unit 650)

As requested, we leverage our water treatment capabilities into a standalone, highly lucrative industrial machinery offering:

```
+----------------------------------------------------------------------------------------------------+
|                         AERO-PURE INDUSTRIAL WATER ULTRAFILTRATION SKID                            |
+----------------------------------------------------------------------------------------------------+
|  1. MULTI-SECTOR COMMERCIALIZATION:                                                                |
|     - Semiconductor Fabs (Ultrapure Water 18.2 MΩ·cm)                                             |
|     - Pharmaceutical & Biotech (WFI Water For Injection USP)                                      |
|     - Power Plants & Boilers (Demineralized High-Pressure Feedwater)                               |
|     - Green Hydrogen Electrolyzers (ASTM D1193 Type 1 Feedwater)                                   |
|                                                                                                    |
|  2. TECHNOLOGICAL ARCHITECTURE:                                                                    |
|     - Stage 1: Ceramic / PVDF Hollow-Fiber Ultrafiltration (0.02 µm pore size, 99.9% turbidity removal) |
|     - Stage 2: Dual Granular Activated Carbon (GAC) Beds (removes trace chlorine & organics)       |
|     - Stage 3: Continuous Electro-Deionization (CEDI) Stacks (continuous self-regenerating resin)  |
|     - Stage 4: Dual-Wavelength 185/254 nm UV TOC Destruction & Polishing Mixed-Bed               |
|                                                                                                    |
|  3. FINANCIAL RETURN:                                                                              |
|     - Standalone Skid Sale Price: $85,000 USD / ₹70,00,000 INR                                    |
|     - Eliminates Hazardous Acid / Caustic Chemical Regeneration Costs (Zero Liquid Discharge - ZLD)|
|     - Gross Margin: 58% on machinery fabrication; 35% on annual consumable replacement            |
+----------------------------------------------------------------------------------------------------+
```

This dual-revenue engine ensures that whether clients are building a green hydrogen plant, a semiconductor facility, or a chemical refinery, our equipment packages capture substantial industrial machinery capital.
