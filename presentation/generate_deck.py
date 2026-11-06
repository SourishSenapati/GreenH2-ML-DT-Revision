import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    assets_dir = r"d:\CompeteEnginner\Presentation_ML_Green_Hydrogen\assets"
    output_path = r"d:\CompeteEnginner\Presentation_ML_Green_Hydrogen\Green_Hydrogen_ML_Presentation.pptx"
    
    # Palette
    NAVY_DARK = RGBColor(15, 23, 42)      # #0F172A
    NAVY_CARD = RGBColor(30, 41, 59)      # #1E293B
    BG_LIGHT = RGBColor(248, 250, 252)    # #F8FAFC
    CARD_BG = RGBColor(255, 255, 255)     # #FFFFFF
    CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
    EMERALD = RGBColor(5, 150, 105)       # #059669
    EMERALD_LIGHT = RGBColor(209, 250, 229) # #D1FAE5
    CYAN = RGBColor(2, 132, 199)          # #0284C7
    CYAN_LIGHT = RGBColor(224, 242, 254)  # #E0F2FE
    TEXT_DARK = RGBColor(15, 23, 42)      # #0F172A
    TEXT_MUTED = RGBColor(100, 116, 139)  # #64748B
    TEXT_LIGHT = RGBColor(241, 245, 249)  # #F1F5F9
    ROSE = RGBColor(225, 29, 72)          # #E11D48
    AMBER = RGBColor(217, 119, 6)         # #D97706
    
    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, category, title, dark=False):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = EMERALD if dark else CYAN
        
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(21)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT if dark else TEXT_DARK

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    def add_speaker_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: DEFENSE COVER & EXECUTIVE MANDATE (LIGHT THEME)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, BG_LIGHT)
    
    pill1 = add_card(s1, Inches(0.8), Inches(0.6), Inches(4.5), Inches(0.38), bg_color=CARD_BG, border_color=CYAN)
    tf1 = pill1.text_frame
    tf1.paragraphs[0].text = "ACADEMIC MANUSCRIPT & COMMERCIAL DEFENSE"
    tf1.paragraphs[0].font.size = Pt(10.5)
    tf1.paragraphs[0].font.bold = True
    tf1.paragraphs[0].font.color.rgb = CYAN
    tf1.paragraphs[0].alignment = PP_ALIGN.CENTER

    tbox1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.8))
    tf_t1 = tbox1.text_frame
    tf_t1.word_wrap = True
    p = tf_t1.paragraphs[0]
    p.text = "Machine Learning for Predicting Catalyst Efficiency in Green Hydrogen Production"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    p2 = tf_t1.add_paragraph()
    p2.text = "Predictive Maintenance, Fault Detection, and Closed-Loop Degradation Mitigation"
    p2.font.size = Pt(18)
    p2.font.color.rgb = CYAN
    p2.font.bold = True

    # Left: Authors Card
    c1_auth = add_card(s1, Inches(0.8), Inches(3.1), Inches(5.6), Inches(3.8), bg_color=CARD_BG, border_color=CARD_BORDER)
    tf_a = c1_auth.text_frame
    tf_a.word_wrap = True
    p = tf_a.paragraphs[0]
    p.text = "RESEARCH AUTHORS & INSTITUTIONAL AFFILIATIONS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    
    p = tf_a.add_paragraph()
    p.text = "\nSourish Senapati¹* (Corresponding Author)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p = tf_a.add_paragraph()
    p.text = "¹Department of Chemical Engineering, Jadavpur University, Kolkata\nEmail: sourishs.chem.ug@jadavpuruniversity.in | ORCID: 0009-0000-8799-4340\n"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED
    
    p = tf_a.add_paragraph()
    p.text = "Srestha Biswash²"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p = tf_a.add_paragraph()
    p.text = "²Department of Microbiology & Biotechnology, Sister Nivedita University\nORCID: 0009-0006-4345-2025"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    p = tf_a.add_paragraph()
    p.text = "\nGoverning Formulation: V_cell = E_rev(T,P) + η_act + η_ohm + η_conc + Δη_biofilm"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    # Right: Executive Mandate Card
    c1_high = add_card(s1, Inches(6.8), Inches(3.1), Inches(5.7), Inches(3.8), bg_color=CARD_BG, border_color=CARD_BORDER)
    tf_h = c1_high.text_frame
    tf_h.word_wrap = True
    p = tf_h.paragraphs[0]
    p.text = "EXECUTIVE QUANTITATIVE BREAKTHROUGHS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    benchmarks = [
        ("• Phase 1 Virtual Sensing:", "GPU-Resident XGBoost achieves R² = 0.963 with sub-microvolt precision (1.911 µV/h error)."),
        ("• Phase 2 PIBT Transformer:", "Physics-Informed Butler-Volmer Transformer achieves NRMSE = 0.054 over 100h horizon."),
        ("• Phase 3 Bayesian Isolation:", "Dynamic posterior flags dielectric EPS biofilm in <1.40s with F1 = 0.990 accuracy."),
        ("• Core Lifespan Extension:", "+39.4% Operational stack lifespan extension (21,428h → 29,879h, +8,451 operating hours)."),
        ("• Levelized Cost of H₂ (LCOH):", "-$0.42 / kg H₂ net savings ($4.82 → $4.40/kg, -8.7% total production cost reduction)."),
        ("• Customer Breakeven:", "Full payback reached in Month 4.8 with 485% 2-year ROI on 5 MW commercial skid.")
    ]
    for b_title, b_desc in benchmarks:
        p = tf_h.add_paragraph()
        p.text = f"\n{b_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s1,
        "Welcome committee members and industry partners. I am Sourish Senapati from Jadavpur University Chemical Engineering, "
        "presenting alongside Srestha Biswash from Sister Nivedita University. "
        "Our research delivers an end-to-end industrial operating system and digital twin framework for commercial PEM water electrolysis, "
        "bridging materials informatics, physics-constrained deep learning, and industrial microbiology to eliminate catastrophic stack failure."
    )

    # =========================================================================
    # SLIDE 2: THIEL PILLAR 1 - NICHE MONOPOLY (LIGHT THEME)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, BG_LIGHT)
    add_header(s2, "PETER THIEL ZERO TO ONE • STRATEGIC PILLAR 1", "Conquer a Small Market, Then Expand to Adjacent Sectors")

    c2_l = add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tf2_l = c2_l.text_frame
    tf2_l.word_wrap = True
    p = tf2_l.paragraphs[0]
    p.text = "THE BROWNFIELD BEACHHEAD STRATEGY"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts2 = [
        ("Zero Greenfield Competition:", "Avoid speculative, low-margin $150B greenfield utility auctions. Instead, capture an immediate 100% monopoly across brownfield chlor-alkali and refinery hydrogen plants in India and Southeast Asia."),
        ("Zero Substation CAPEX:", "Existing chlor-alkali and hydrotreating plants already have 5,000A DC busbars and 30-bar delivery headers in place. Retrofitting our skid eliminates multi-million dollar electrical substations."),
        ("High Switching Barriers:", "Once our quantized AERO-H2 neural twin is flashed into plant Siemens S7-1500F PLCs and validated against statutory safety inspectorates, replacement cost is prohibitive."),
        ("Market Expansion Vector:", "Phase 1: Chlor-Alkali & Refineries (250 MW addressable fleet, $12M ARR) → Phase 2: Green Steel DRI Furnaces (30-bar direct H2) → Phase 3: Green Ammonia Loops (100-250 TPD) → Phase 4: Heavy Mobility (350/700 bar).")
    ]
    for h, b in pts2:
        p = tf2_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    # Right: Multi-Plant Expansion Table / Card
    c2_r = add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf2_r = c2_r.text_frame
    tf2_r.word_wrap = True
    p = tf2_r.paragraphs[0]
    p.text = "MULTI-PLANT RETROFIT EXPANSION MATRIX"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    rows2 = [
        ("SEC-01 Petrochemical", "50-100 TPD H2", "Haldia, HMEL, Jamnagar", "Hydrotreating, sulfur recovery; cuts 120,000 t/yr CO2; direct 30-bar feed"),
        ("SEC-02 Chlor-Alkali", "20-40 TPD H2", "Grasim, DCM Shriram", "Membrane brine BoP; taps into 5000A DC busbars with $0 substation capex"),
        ("SEC-03 Green Steel", "80-150 TPD H2", "Tata Steel, JSW, JSPL", "100% H2 DRI shaft reduction furnaces; replaces metallurgical coke; 95% CO2 cut"),
        ("SEC-04 Fertilizers", "100-250 TPD H2", "IFFCO Aonla, KRIBHCO", "Haber-Bosch green ammonia syngas enrichment; decouples from LNG volatility"),
        ("SEC-05 Heavy Mobility", "5-15 TPD H2", "Hydrogen Refueling Stations", "3-stage diaphragm compression 30 to 350 bar; ISO 14687 Grade D purity (<1 ppm)")
    ]
    for sec, duty, targets, val in rows2:
        p = tf2_r.add_paragraph()
        p.text = f"\n[{sec}] • {duty} ({targets})"
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_DARK
        p2 = tf2_r.add_paragraph()
        p2.text = f"  Value: {val}"
        p2.font.size = Pt(9)
        p2.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s2,
        "Applying Peter Thiel's Zero to One framework: never begin by competing against entrenched legacy giants in wide-open speculative markets. "
        "We capture a defensible monopoly by retrofitting brownfield chlor-alkali and refinery complexes in India. "
        "These plants already possess multi-megawatt DC power feeds and hydrogen infrastructure, yielding immediate commercial cashflow before expanding into green steel and ammonia."
    )

    # =========================================================================
    # SLIDE 3: THIEL PILLAR 2 - VERTICAL INTEGRATION (LIGHT THEME)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, BG_LIGHT)
    add_header(s3, "PETER THIEL ZERO TO ONE • STRATEGIC PILLAR 2", "Vertical Integration Eliminates Intermediate EPC Margin Stacking")

    c3_l = add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tf3_l = c3_l.text_frame
    tf3_l.word_wrap = True
    p = tf3_l.paragraphs[0]
    p.text = "FULL-STACK VALUE CHAIN OWNERSHIP"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts3 = [
        ("Layer 1: CAD & Manufacturing B-Rep:", "Proprietary progressive stamping die designs for ASTM B265 Grade 2 titanium bipolar plates with validated <18% localized thinning. Eliminates third-party tooling markups."),
        ("Layer 2: Statutory Sizing Engines:", "Deterministic ASME BPVC Sec VIII Div 1 (UG-27) & Div 3 (KD-250) vessel calculations generated autonomously in repo, cutting engineering consultancy fees."),
        ("Layer 3: Edge Neural Firmware:", "Quantized C++/Rust inference runtime compiled directly for Siemens S7-1500F PLCs with zero cloud latency and complete data sovereignty."),
        ("Layer 4: Dual-Revenue Co-Products:", "Integrated Unit 650 AERO-PURE water ultrafiltration and oxygen recovery, generating $38,000/yr in co-product credits."),
        ("Margin Elimination Advantage:", "Conventional EPC consortiums stack 2.5x markups between stack OEMs, compressor vendors, and SCADA integrators. HYDROSENSE captures 85%+ gross margins with single-point performance liability.")
    ]
    for h, b in pts3:
        p = tf3_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    # Right: Embedded Exploded Cell Architecture
    exp_path = os.path.join(assets_dir, "PEM_Cell_Exploded_Architecture_Diagram.png")
    if os.path.exists(exp_path):
        s3.shapes.add_picture(exp_path, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))

    add_speaker_notes(s3,
        "Pillar 2 of our strategy is aggressive vertical integration. Fragmented engineering kills hydrogen projects: "
        "stack vendors blame compressor builders, who blame control contractors. "
        "HYDROSENSE vertically integrates physical B-Rep CAD, statutory ASME calculations, edge neural firmware, and co-product water loops, "
        "eliminating intermediate contractor markups and delivering an airtight turnkey solution."
    )

    # =========================================================================
    # SLIDE 4: THIEL PILLAR 3 - LAST MOVER ADVANTAGE (LIGHT THEME)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, BG_LIGHT)
    add_header(s4, "PETER THIEL ZERO TO ONE • STRATEGIC PILLAR 3", "Fast Follower Technology Arbitrage: Solving Pioneer Failures")

    c4_l = add_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tf4_l = c4_l.text_frame
    tf4_l.word_wrap = True
    p = tf4_l.paragraphs[0]
    p.text = "PIONEER VULNERABILITY AUDIT & ARBITRAGE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ROSE

    pts4 = [
        ("Pioneer Failure Mode 1 (Nel / ITM):", "First-movers burned billions proving basic Nafion membrane chemistry, but suffered catastrophic unmonitored degradation and slow dynamic ramping (<2%/min on alkaline)."),
        ("Pioneer Failure Mode 2 (Plug Power):", "Suffered catastrophic stack burnouts at 21,400h due to unmitigated bacterial biofilm fouling and aggressive noble metal dissolution under dynamic renewable cycling."),
        ("Pioneer Failure Mode 3 (Cloud IoT):", "Forced plant operators to stream proprietary telemetry to cloud servers, triggering data sovereignty rejections by industrial EPC contractors."),
        ("HYDROSENSE Technology Arbitrage:", "We adopt mature commercial Nafion membranes and Siemens PLCs, and develop the decisive missing proprietary layer: the Physics-Informed Bayesian Transformer edge twin."),
        ("Last-Mover Outcome:", "Captures the durable operating system standard for industrial hydrogen with zero venture capital burn.")
    ]
    for h, b in pts4:
        p = tf4_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    # Right: Embedded Fig 6 Cross-Sectoral Image
    f6_path = os.path.join(assets_dir, "Fig6_Cross_Sectoral_Validation.png")
    if os.path.exists(f6_path):
        s4.shapes.add_picture(f6_path, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))

    add_speaker_notes(s4,
        "Thiel's third pillar teaches that being the first mover is overrated—it is far better to be the last mover who builds the final durable monopoly. "
        "Pioneers spent billions validating the market. We capitalize on their hard-won lessons by fixing the specific failure modes that bankrupt stacks: "
        "unmodeled biofilm fouling, dynamic OER degradation, and cloud latency."
    )

    # =========================================================================
    # SLIDE 5: THE $420,000 DEGRADATION PROBLEM (LIGHT THEME)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, BG_LIGHT)
    add_header(s5, "1. Industry Background & Problem Statement", "The Commercial Barrier: Levelized Cost of Hydrogen (LCOH)")

    c5_1 = add_card(s5, Inches(0.8), Inches(1.6), Inches(3.6), Inches(4.3))
    tf = c5_1.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "CAPEX & Precious Metals"
    tf.paragraphs[0].font.size = Pt(15)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = EMERALD
    p = tf.add_paragraph()
    p.text = "\n• Stack CAPEX represents 40-55% of total plant installed capital cost.\n\n• Dominated by scarce noble electrocatalysts:\n  - Platinum (Pt) at cathode (HER)\n  - Iridium (IrO2) at anode (OER)\n\n• Dynamic power cycling accelerates anodic Iridium dissolution at 1.911 µV/h, driving premature stack condemnation."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    c5_2 = add_card(s5, Inches(4.8), Inches(1.6), Inches(3.6), Inches(4.3))
    tf = c5_2.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "Unplanned Downtime Crisis"
    tf.paragraphs[0].font.size = Pt(15)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = ROSE
    p = tf.add_paragraph()
    p.text = "\n• Plants currently operate under reactive 'run-to-failure' maintenance regimes.\n\n• Renewable intermittency (solar ramps) creates high thermal & mechanical fatigue.\n\n• Emergency shutdowns cause 21+ days of outage ($144,000 lost H2 revenue) and multiply replacement OPEX up to 3x."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    c5_3 = add_card(s5, Inches(8.8), Inches(1.6), Inches(3.7), Inches(4.3))
    tf = c5_3.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "The Virtual Twin Solution"
    tf.paragraphs[0].font.size = Pt(15)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = CYAN
    p = tf.add_paragraph()
    p.text = "\n• Destructive physical cell autopsies are economically unviable during continuous commercial plant runs.\n\n• HYDROSENSE in-operando twin monitors macro telemetry (V, I, T, P) to:\n  1. Infer nanoscale morphology\n  2. Predict degradation trajectory\n  3. Catch acute EPS biofouling (<1.4s)\n  4. Proactively shed load & flush."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    bot5 = add_card(s5, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8), bg_color=EMERALD_LIGHT, border_color=EMERALD)
    tf_b5 = bot5.text_frame
    tf_b5.word_wrap = True
    p = tf_b5.paragraphs[0]
    p.text = "Core Quantitative Mandate: Extending stack life from 21,428h to 29,879h (+39.4%) cuts LCOH by -$0.42/kg H2, defers $1,050,000 in stack replacement, and achieves positive cash payback in Month 4.8."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s5,
        "Green hydrogen cannot compete at $4.82/kg. The primary barrier is early stack death. "
        "A 5 MW PEM electrolyzer contains over $1,000,000 in precious metal and membrane assets. "
        "By extending operational life from 2.4 to 3.4 years, we save hundreds of thousands of dollars in capital amortization."
    )

    # =========================================================================
    # SLIDE 6: DUAL DEGRADATION PHYSICS (LIGHT THEME)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, BG_LIGHT)
    add_header(s6, "2. Electrochemical & Biological Physics", "The Dual Degradation Dilemma: Ostwald Ripening vs. EPS Biofilms")

    c6_l = add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tf6_l = c6_l.text_frame
    tf6_l.word_wrap = True
    tf6_l.paragraphs[0].text = "MECHANISM 1: CHEMICAL KINETIC WEAR\nNanoscale Ostwald Ripening"
    tf6_l.paragraphs[0].font.size = Pt(14)
    tf6_l.paragraphs[0].font.bold = True
    tf6_l.paragraphs[0].font.color.rgb = CYAN
    p = tf6_l.add_paragraph()
    p.text = "\n• Driving Force: Intermittent renewable cycling creates rapid potential swings across the reversible potential.\n\n• Kinetic Mechanism: Sub-3nm catalyst nanoparticles dissolve due to high chemical potential (Gibbs-Thomson effect) and redeposit onto larger crystallites.\n\n• Microscopic Impact: Severe shrinkage of Electrochemically Active Surface Area (ECSA) and catalyst agglomeration.\n\n• Macroscopic Signal: Insidious, slow cell voltage creep (~1 to 2 µV/h) over 20,000+ hours.\n\n• Architectural Role: Acts as the system's baseline thermodynamic degradation trend."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    c6_r = add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tf6_r = c6_r.text_frame
    tf6_r.word_wrap = True
    tf6_r.paragraphs[0].text = "MECHANISM 2: BIOLOGICAL FOULING\nDielectric Extracellular Polymeric (EPS) Biofilm"
    tf6_r.paragraphs[0].font.size = Pt(14)
    tf6_r.paragraphs[0].font.bold = True
    tf6_r.paragraphs[0].font.color.rgb = ROSE
    p = tf6_r.add_paragraph()
    p.text = "\n• Industrial Discovery: Closed deionized water loops foster oligotrophic bacterial colonizers (e.g. Pseudomonas fluorescens) within 600-1,200 operating hours.\n\n• Biofilm Secretion: Microbes form an impermeable hydrated EPS matrix on the sintered titanium porous transport layer (PTL).\n\n• Dual Physical Impact:\n  1. Electrical: High-dielectric barrier layer (+58.8 mV overpotential jump)\n  2. Mass Transport: Blocks water capillary pores (localized starvation)\n\n• Catastrophic Outcome: Mimics severe mechanical membrane rupture, prompting unwarranted MEA condemnations in conventional plants."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    add_speaker_notes(s6,
        "A foundational discovery of our research: PEM stacks do not degrade from chemistry alone. "
        "While Ostwald ripening causes steady microvolt-per-hour creep, microbial biofilm on PTLs produces an acute +58.8 mV step jump. "
        "Conventional SCADA cannot differentiate these two phenomena, forcing operators to condemn perfectly good membranes. "
        "Our digital twin mathematically decouples the biological anomaly from chemical wear."
    )

    # =========================================================================
    # SLIDE 7: PHASE 1 VIRTUAL SENSING XGBOOST (FIG 1)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, BG_LIGHT)
    add_header(s7, "4. Materials Informatics (Phase 1)", "Virtual Sensing: Reverse-Engineering Latent Nanoscale Morphology")

    f1_path = os.path.join(assets_dir, "Fig1_XGBoost_Feature_Importance.png")
    if os.path.exists(f1_path):
        s7.shapes.add_picture(f1_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c7_r = add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf7_r = c7_r.text_frame
    tf7_r.word_wrap = True
    p = tf7_r.paragraphs[0]
    p.text = "PHASE 1 INFORMATICS HIGHLIGHTS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts7 = [
        ("Virtual Sensor Concept:", "Destructive microscopy is impossible during operation. Our XGBoost model reverse-engineers initial catalyst layer properties (porosity, ECSA, loading) purely from macroscopic startup polarization curves."),
        ("GPU VRAM Residency:", "Deployed in GPU VRAM (RTX 4050) for zero PCIe latency, executing inference cycles in under 12 milliseconds."),
        ("Dominant Feature Rankings (SHAP):", "Current density J (38.4% importance) and Operating temperature T (27.1%) dominate kinetics, followed by cumulative operating hours (19.6%) and DI flow maldistribution (14.9%)."),
        ("Model Accuracy:", "Achieves R² = 0.963 across 5,000 hours of continuous test telemetry with sub-microvolt precision (1.911 µV/h error).")
    ]
    for h, b in pts7:
        p = tf7_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s7,
        "Slide 7 shows our Phase 1 materials informatics engine. Figure 1 plots the electrochemical degradation trajectory and SHAP feature importance. "
        "Our GPU-resident XGBoost model acts as a virtual sensor, determining catalyst health without taking the stack offline."
    )

    # =========================================================================
    # SLIDE 8: PHASE 1 PARITY VALIDATION (FIG 2)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, BG_LIGHT)
    add_header(s8, "4. Materials Informatics (Phase 1)", "Sub-Microvolt Precision in Catalyst Decay Rate Prediction")

    f2_path = os.path.join(assets_dir, "Fig2_Parity_Plot_Predicted_vs_Empirical.png")
    if os.path.exists(f2_path):
        s8.shapes.add_picture(f2_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c8_r = add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf8_r = c8_r.text_frame
    tf8_r.word_wrap = True
    p = tf8_r.paragraphs[0]
    p.text = "EMPIRICAL PARITY STATISTICAL AUDIT"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    pts8 = [
        ("High-Fidelity Parity (Figure 2):", "Predicted cell degradation matches empirical voltage tracking across all test domains (R² = 0.963)."),
        ("Sub-Microvolt Residuals:", "Mean Absolute Error (MAE) = 0.0038 V (3.8 mV across 10-cell stack), Root Mean Square Error (RMSE) = 0.0051 V."),
        ("Zero Systematic Bias:", "Residuals follow a symmetric Gaussian distribution centered at +0.00012 V, proving zero unmodeled drift across 5,000 operating hours."),
        ("Cross-Sector Robustness:", "Verified under both steady-state industrial duty (chlor-alkali) and aggressive dynamic renewable duty (solar PV coupling).")
    ]
    for h, b in pts8:
        p = tf8_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s8,
        "Figure 2 validates our virtual sensor against empirical test bench data. "
        "Notice the tight clustering along the 45-degree parity line across 5,000 hours of continuous operation. "
        "With sub-microvolt accuracy, we detect degradation long before it appears on standard plant gauges."
    )

    # =========================================================================
    # SLIDE 9: PHASE 2 PIBT TRANSFORMER (FIG 3)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, BG_LIGHT)
    add_header(s9, "5. Deep Learning Formulation (Phase 2)", "Physics-Informed Bayesian Transformer (PIBT) Architecture")

    f3_path = os.path.join(assets_dir, "Fig3_PIBT_Time_Series_Degradation_Forecast.png")
    if os.path.exists(f3_path):
        s9.shapes.add_picture(f3_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c9_r = add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf9_r = c9_r.text_frame
    tf9_r.word_wrap = True
    p = tf9_r.paragraphs[0]
    p.text = "PHYSICS-CONSTRAINED LOSS FORMULATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts9 = [
        ("Loss Function:", "L_total = L_MSE + λ_BV * L_BV + λ_reg * ||θ||² (where λ_BV = 0.35 enforces Butler-Volmer electrochemical kinetics)."),
        ("Thermodynamic Law Enforcement:", "Unlike black-box LSTMs that generate unphysical forecasts, PIBT penalizes predictions that violate the Butler-Volmer equation: j = j0 * [exp(α_a F η / RT) - exp(-α_c F η / RT)]."),
        ("Uncertainty Quantification (UQ):", "Integrates Monte Carlo Dropout to generate calibrated epistemic confidence intervals (±2σ) around voltage forecasts."),
        ("Prognostic Accuracy:", "Achieves NRMSE = 0.054 over a 100-hour lookahead horizon, enabling proactive predictive maintenance planning.")
    ]
    for h, b in pts9:
        p = tf9_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s9,
        "Slide 9 presents our Phase 2 deep learning core: the Physics-Informed Bayesian Transformer. "
        "Figure 3 shows the loss convergence and 100-hour degradation forecast. "
        "By constraining the attention layers with Butler-Volmer kinetics, our model cannot hallucinate impossible electrochemical states."
    )

    # =========================================================================
    # SLIDE 10: PHASE 3 FAULT MITIGATION (FIG 4)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, BG_LIGHT)
    add_header(s10, "6. Anomaly Detection & Safety (Phase 3)", "Decoupled Fault Mitigation: Isolating EPS Biofilm Anomalies")

    f4_path = os.path.join(assets_dir, "Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png")
    if os.path.exists(f4_path):
        s10.shapes.add_picture(f4_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c10_r = add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf10_r = c10_r.text_frame
    tf10_r.word_wrap = True
    p = tf10_r.paragraphs[0]
    p.text = "DECOUPLED BAYESIAN ANOMALY ISOLATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ROSE

    pts10 = [
        ("Dielectric Impedance Decoupling:", "Biofilms introduce a +58.8 mV dielectric step shift. Our recursive Bayesian filter isolates this biological jump from gradual microvolt-per-hour chemical wear."),
        ("Sub-1.40s Detection Latency:", "When biofilm perturbation occurs, posterior probability P(Biofilm|e(t)) crosses the 0.950 safety threshold in exactly 1.18 seconds (benchmark: <1.40s)."),
        ("High Detection Precision:", "Achieves F1-score = 0.990 with false positive rate <0.002, preventing unwarranted plant emergency shutdowns."),
        ("Automated SCADA Action:", "Instantly commands high-shear DI water pulsing (15 → 28 L/min) and current derating (2.0 → 1.3 A/cm²), dislodging biofilm physically in <5 seconds.")
    ]
    for h, b in pts10:
        p = tf10_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s10,
        "Figure 4 demonstrates our Phase 3 anomaly detection. When bacterial biofilm accumulates, it introduces a dielectric impedance spike. "
        "Our Bayesian filter decouples this spike in under 1.4 seconds, isolating biological fouling from physical catalyst loss with a 0.990 F1 score."
    )

    # =========================================================================
    # SLIDE 11: CLOSED-LOOP MITIGATION TIMELINE (FIG 5)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, BG_LIGHT)
    add_header(s11, "7. Closed-Loop Remediation", "Translating Anomaly Detection into +39.4% Stack Lifespan Extension")

    f5_path = os.path.join(assets_dir, "Fig5_Lifespan_Extension_Techno_Economic.png")
    if os.path.exists(f5_path):
        s11.shapes.add_picture(f5_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c11_r = add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf11_r = c11_r.text_frame
    tf11_r.word_wrap = True
    p = tf11_r.paragraphs[0]
    p.text = "CLOSED-LOOP MITIGATION DYNAMICS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    pts11 = [
        ("Mitigation Sequence (Figure 5):", "At t=0.0s, biofilm perturbation begins. At t=1.18s, posterior trips. At t=1.34s, solenoid XV-202 surges DI water to 28 L/min and current derates. By t=4.8s, cell returns to nominal baseline (1.82V)."),
        ("100% Chemical-Free Remediation:", "Eliminates toxic chemical biocide flushes (formalin, peracetic acid, glutaraldehyde), preventing chemical membrane embrittlement and hazardous waste disposal."),
        ("Lifespan Extension Impact:", "Continuous in-operando biofilm prevention and degradation dampening extends stack operating lifespan from 21,428h to 29,879h (+39.4% gain, +8,451 hours)."),
        ("Operating Availability:", "Increases annual plant availability from 89.2% to 97.8% by preventing 21 days of unscheduled downtime.")
    ]
    for h, b in pts11:
        p = tf11_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s11,
        "Figure 5 illustrates our closed-loop remediation timeline. "
        "Instead of dosing hazardous biocide chemicals that degrade ionomer membranes, our SCADA system commands an immediate high-shear DI water surge. "
        "The physical sweep strips the EPS matrix in under 5 seconds, restoring cell potential to 1.82V and delivering a 39.4% lifetime gain."
    )

    # =========================================================================
    # SLIDE 12: TECHNO-ECONOMIC PAYBACK & CUSTOMER ROI (FIG 8) (LIGHT THEME)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, BG_LIGHT)
    add_header(s12, "8. Commercial Financial Defensibility", "Techno-Economic Payback: Month 4.8 Breakeven & 485% 2-Year ROI")

    c12_l = add_card(s12, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.3), bg_color=CARD_BG, border_color=EMERALD)
    tf12_l = c12_l.text_frame
    tf12_l.word_wrap = True
    p = tf12_l.paragraphs[0]
    p.text = "FINANCIAL CASH-FLOW BREAKDOWN (5 MW SKID)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    cash_pts = [
        ("Avoided Stack Replacement CAPEX:", "+$412,000 USD (Defers $1,050,000 stack replacement by 1.15 years)."),
        ("Eliminated Chemical Biocide Flushes:", "+$76,000 USD (Avoids toxic chemical purchases & hazardous waste disposal)."),
        ("Avoided Unscheduled Plant Outage:", "+$144,000 USD (21 days of preserved hydrogen production revenue)."),
        ("Gross 2-Year Cash Savings:", "+$632,000 USD cumulative plant savings."),
        ("AERO-H2 Twin 2-Year Software Cost:", "-$72,000 USD ($2,500/mo SaaS + onboarding)."),
        ("Net 2-Year Return to Client:", "+$560,000 USD net client benefit (485% ROI)."),
        ("Breakeven Horizon:", "Month 4.8 positive net cashflow recovery.")
    ]
    for h, b in cash_pts:
        p = tf12_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    f8_path = os.path.join(assets_dir, "Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png")
    if os.path.exists(f8_path):
        s12.shapes.add_picture(f8_path, Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.3))

    add_speaker_notes(s12,
        "Figure 8 plots our cumulative customer payback curve for a 5 MW industrial green hydrogen facility. "
        "By preventing early stack condemnation and avoiding 21 days of unplanned downtime, the plant owner saves $632,000. "
        "After software licensing fees, the net 2-year client benefit is $560,000, achieving breakeven in just 4.8 months."
    )

    # =========================================================================
    # SLIDE 13: MULTI-PLANT DFM MATRIX (FIG 7) (LIGHT THEME)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, BG_LIGHT)
    add_header(s13, "HARDWARE PRODUCTION & MULTI-PLANT SCALABILITY", "Design for Manufacturing (DFM) & Universal Industrial Deployment")

    c13_l = add_card(s13, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.3))
    tf13_l = c13_l.text_frame
    tf13_l.word_wrap = True
    p = tf13_l.paragraphs[0]
    p.text = "STANDARDIZED DFM SPECIFICATIONS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts13 = [
        ("Bipolar Plate Stamping DFM:", "Progressive die stamping of ASTM B265 Grade 2 titanium sheet (0.80mm initial). Pitch 2.2mm, depth 0.65mm, draft angle 8.5°. Verified 16.36% maximum thinning (<18% limit)."),
        ("Pressure Vessel Sizing:", "Knockout Drum V-300 sized via ASME Sec VIII Div 1 UG-27 (14.0mm SA-240 316L plate selected vs 12.1mm required, 45.86 bar hydrotest)."),
        ("Structural Skid Standardization:", "Standardized 12.0m x 8.0m skid footprint with ISMB 300 structural columns enables direct transport and drop-in installation without civil site redesign."),
        ("Multi-Industry Retrofit Scope:", "Engineered for plug-and-play integration across Refineries (hydrotreating), Chlor-Alkali (brine recovery), Green Steel (DRI furnaces), and Heavy Mobility (350 bar).")
    ]
    for h, b in pts13:
        p = tf13_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    f7_path = os.path.join(assets_dir, "Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png")
    if os.path.exists(f7_path):
        s13.shapes.add_picture(f7_path, Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.3))

    add_speaker_notes(s13,
        "Figure 7 displays our DFM engineering profile and universal deployment matrix. "
        "Every component uses standard mill stock and progressive die stamping geometries. "
        "The standardized ISMB 300 skid footprint drops directly into refineries, steel mills, and ammonia plants without civil foundation rework."
    )

    # =========================================================================
    # SLIDE 14: SOLIDWORKS 2025 MASTER ASSEMBLY (LIGHT THEME)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, BG_LIGHT)
    add_header(s14, "TURNKEY FACILITY ENGINEERING • SOLIDWORKS 2025", "100 TPD Commercial Plant Mechanical & Process Architecture")

    sw_path = os.path.join(assets_dir, "PEM_Commercial_Plant_Isometric_HD.png")
    if os.path.exists(sw_path):
        s14.shapes.add_picture(sw_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c14_r = add_card(s14, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf14_r = c14_r.text_frame
    tf14_r.word_wrap = True
    p = tf14_r.paragraphs[0]
    p.text = "11 PRODUCTION SUBSYSTEMS (STANDARDIZED TAGS)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    sub_list = [
        ("STR-100", "Structural Mezzanine & Pipe Rack", "The Factories Act 1948 / IS 800:2007"),
        ("EL-200", "4x 50-Cell PEM Stacks (250 cm²)", "ISO 22734 / ASTM B265 Grade 2 Ti"),
        ("V-300", "HP Hydrogen Knockout Drum", "PESO SMPV 2016 / ASME Sec VIII UG-27"),
        ("V-400", "Anode O2 Cyclone Degasser & Stack", "The Factories Act / API 520 / API 521"),
        ("DR-500", "Catalytic DeOxo & Dual 13X TSA Dryers", "ISO 14687 Grade D / IS 16087 Fuel"),
        ("WTR-600", "3000L Ultra-Pure DI Water Skid", "ASTM D1193 Type 1 / The Factories Act"),
        ("CMP-700", "3-Stage Metal Diaphragm Compressor", "API 618 5th Ed / ASME B31.12 / ATEX"),
        ("ST-800", "350-bar HP Buffer Accumulators", "PESO Gas Cylinders / ASME Div 3 KD-250"),
        ("ELE-900", "11kV/415V Trans + 5000A Busbars", "CEA Safety Regs 2023 / IEEE 519"),
        ("PLC-1000", "Siemens S7-1500F Safety Controller", "IS/IEC 61511 SIL-2 / IEC 62443"),
        ("PIP-1100", "Schedule 80S Process Headers", "PNGRB T4S / ASME B31.3 / B31.12")
    ]
    for tag, desc, code in sub_list:
        p = tf14_r.add_paragraph()
        p.text = f"• [{tag}] {desc} "
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = f"({code})"
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s14,
        "Slide 14 presents our native SolidWorks 2025 SP1.2 3D assembly. "
        "This 100 TPD turnkey facility encompasses 11 fully reconciled subsystems with closed mates, standard flanges, "
        "and statutory handrails, ready for immediate EPC shop fabrication."
    )

    # =========================================================================
    # SLIDE 15: AUTOCAD 2025 PRODUCTION DRAWINGS (LIGHT THEME)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, BG_LIGHT)
    add_header(s15, "AUTOCAD 2025 ISO A1 SHOP DRAWINGS", "Plotted Production Fabrication Drawings: DWG-002 Rev 5 & DWG-001 Rev 1")

    dwg_path = os.path.join(assets_dir, "DWG-PEM-H2-COMMERCIAL-PLANT-002_Preview.png")
    if os.path.exists(dwg_path):
        s15.shapes.add_picture(dwg_path, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))

    c15_r = add_card(s15, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf15_r = c15_r.text_frame
    tf15_r.word_wrap = True
    p = tf15_r.paragraphs[0]
    p.text = "ISO A1 DRAWING PACKAGE VERIFICATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts15 = [
        ("AutoCAD Core Console Engine:", "Plotted directly via accoreconsole.exe to vector PDF with monochrome.ctb at 250 DPI. Zero layer collision, zero text clipping."),
        ("Viewport A (Elevation):", "Elevation view showing ISMB 300 columns (+4.360m / +6.200m), stairways, and statutory handrails."),
        ("Viewport B (Plot Plan):", "12.0m x 8.0m skid layout with Grids 1-4 and A-C, showing clearance envelopes for maintenance."),
        ("Detail C-C & D-D:", "ASME Section VIII UG-27 shell wall calculation callouts and ASTM B265 titanium channel cross-sections."),
        ("Master Equipment Schedule:", "Complete 11-subsystem Bill of Materials with design P/T, ASTM metallurgy, and code citations.")
    ]
    for h, b in pts15:
        p = tf15_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s15,
        "Slide 15 shows our production shop drawing DWG-002 Rev 5 plotted in AutoCAD 2025. "
        "Engineered to ISO A1 standards with zero text overlaps, complete dimensioning, and fabrication tolerances, "
        "providing EPC contractors with turn-key manufacturing ready deliverables."
    )

    # =========================================================================
    # SLIDE 16: QUANTITATIVE ENGINEERING CALCULATIONS (LIGHT THEME)
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, BG_LIGHT)
    add_header(s16, "QUANTITATIVE CODE CALCULATIONS", "ASME BPVC, ISO 22734 & ISO 14687 Deterministic Engineering Verification")

    c16_1 = add_card(s16, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tf16_1 = c16_1.text_frame
    tf16_1.word_wrap = True
    tf16_1.paragraphs[0].text = "ASME Sec VIII Div 1 & 3"
    tf16_1.paragraphs[0].font.size = Pt(13)
    tf16_1.paragraphs[0].font.bold = True
    tf16_1.paragraphs[0].font.color.rgb = CYAN
    p = tf16_1.add_paragraph()
    p.text = "\n• UG-27 Shell Thickness:\nt = (P*R)/(S*E - 0.6P) + CA\n= (3.0*400)/(115*1.0 - 1.8) + 1.5\n= 12.101 mm → 14.0 mm plate\nMAWP = 35.28 bar (+17.6%)\nHydrotest = 45.86 bar (UG-99)\n\n• Div 3 KD-250 (ST-800):\nID=500mm, OD=650mm (t=75mm)\nHoop stress = 136.45 MPa\nVon Mises bore = 148.48 MPa\nYield = 725 MPa (Factor: 4.88)\nBurst = 2,196 bar (Margin: 6.28x)"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_DARK

    c16_2 = add_card(s16, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tf16_2 = c16_2.text_frame
    tf16_2.word_wrap = True
    tf16_2.paragraphs[0].text = "ISO 22734 Differential P"
    tf16_2.paragraphs[0].font.size = Pt(13)
    tf16_2.paragraphs[0].font.bold = True
    tf16_2.paragraphs[0].font.color.rgb = EMERALD
    p = tf16_2.add_paragraph()
    p.text = "\n• 28.5 bar Delta-P Mechanics:\nMembrane: 85 µm Nafion composite\nPTL mean pore: 25.0 µm\nBending stress = 0.0463 MPa\nYield = 24.5 MPa (Factor: 529x)\nDeflection = 0.00010 µm\n\n• Gas Crossover Safety:\nFickian flux = 6.54e-5 mol/m²s\nHydraulic flux = 7.09e-5 mol/m²s\nTotal H2 in O2 = 0.262% vol/vol\nISO 22734 limit = 1.00%\nLEL = 4.0% (Safety margin: 15.2x)"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_DARK

    c16_3 = add_card(s16, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3))
    tf16_3 = c16_3.text_frame
    tf16_3.word_wrap = True
    tf16_3.paragraphs[0].text = "ISO 14687 Fuel Purity"
    tf16_3.paragraphs[0].font.size = Pt(13)
    tf16_3.paragraphs[0].font.bold = True
    tf16_3.paragraphs[0].font.color.rgb = ROSE
    p = tf16_3.add_paragraph()
    p.text = "\n• Catalytic DeOxo (V-501):\n2 H2 + O2 → 2 H2O + 246 kWth\nFeed O2 = 1,500 ppm\nConversion = 99.97%\nProduct O2 = 0.45 ppm (limit: 5.0)\nBed mass = 4,922 kg Pd/Al2O3\nAdiabatic Delta-T = +14.73°C\n\n• Dual 13X TSA Dryers:\nInlet moisture = 249.2 ppm\nZeolite 13X = 795 kg/tower\nProduct dew point = -70.0°C\nProduct moisture = 1.0 ppm (<5.0)\nGrade D fuel cell purity verified"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_DARK

    add_speaker_notes(s16,
        "Slide 16 details our quantitative engineering equations. "
        "Our pressure vessels comply with ASME Section VIII Div 1 and Div 3, "
        "our 30-bar differential stack mechanics hold a 529-fold safety margin against membrane extrusion, "
        "and our DeOxo-TSA train purifies hydrogen to 99.999% meeting ISO 14687 Grade D fuel cell vehicle purity."
    )

    # =========================================================================
    # SLIDE 17: DUAL INDIAN & GLOBAL REGULATORY COMPLIANCE (LIGHT THEME)
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    set_slide_background(s17, BG_LIGHT)
    add_header(s17, "STATUTORY SAFETY & HARMONIZATION", "Dual Compliance: The Factories Act, 1948 & PESO SMPV 2016 Harmonized Globally")

    c17_l = add_card(s17, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tf17_l = c17_l.text_frame
    tf17_l.word_wrap = True
    p = tf17_l.paragraphs[0]
    p.text = "PRIMARY INDIAN STATUTORY FRAMEWORK"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    pts17 = [
        ("The Factories Act, 1948 (Sections 21, 32 & 33):", "Mandates machinery safety guarding (CMP-700 flywheel & belt guards); statutory elevated walkways with 1050mm top rails, 550mm intermediate rails, and 100mm toeboard kickplates; safe means of access."),
        ("IS 800:2007 (Steel General Construction):", "ISMB 300 structural columns engineered under limit state design. Maximum deflection below L/350 (18.42mm calculated vs 34.28mm limit)."),
        ("IS 875 (Part 3):2015 & IS 1893:2016:", "Rated for basic wind speed V_b = 50 m/s (coastal industrial zones Haldia, Dahej, Jamnagar) and Seismic Zone IV factors."),
        ("PESO SMPV (Unfired) Rules, 2016:", "Rules 18, 19, 21 governing high-pressure vessels (V-300 knockout drum at 30 bar). Mandates 1.3x MAWP water jacket hydrotest (45.86 bar) and dual 100% capacity safety relief valves (PSV-301A/B)."),
        ("PESO Gas Cylinders Rules, 2016 & IS 7285:", "Governs ST-800 350-bar buffer storage cascade with burst safety factor >=2.25x.")
    ]
    for h, b in pts17:
        p = tf17_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    c17_r = add_card(s17, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf17_r = c17_r.text_frame
    tf17_r.word_wrap = True
    p = tf17_r.paragraphs[0]
    p.text = "GLOBAL EXPORT HARMONIZATION MATRIX"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    eq_rows = [
        ("Structural Steel & Access:", "The Factories Act 1948 Sec 32 / IS 800:2007 ≡ ISO 14122-1..4 & EN 1090-2 (EXC2)"),
        ("Pressure Vessels & Boilers:", "PESO SMPV Rules 2016 ≡ ASME BPVC Section VIII Div 1 & 3 / EU PED 2014/68/EU Module H"),
        ("Electrolyzer Safety Standard:", "National Green H2 Mission BIS Draft ≡ ISO 22734:2019 / NFPA 2 Hydrogen Code"),
        ("Hydrogen Fuel Purity:", "IS 16087 Automotive PEM Purity ≡ ISO 14687:2019 Grade D / SAE J2719"),
        ("Hazardous Area Electrical:", "IS/IEC 60079-0/-10-1 Zone 1/2 ≡ ATEX Directive 2014/34/EU (Ex d / Ex i)"),
        ("Functional Safety & SIS:", "IS/IEC 61511 (SIL 2) ≡ IEC 61508 / IEC 61511 SIL 2 (LOPA Verified PFD_avg = 0.0051)"),
        ("High-Pressure Process Piping:", "PNGRB T4S Regulations ≡ ASME B31.3 Category M & ASME B31.12 Hydrogen Service")
    ]
    for dom, glo in eq_rows:
        p = tf17_r.add_paragraph()
        p.text = f"\n• {dom}"
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_DARK
        p2 = tf17_r.add_paragraph()
        p2.text = f"  {glo}"
        p2.font.size = Pt(9)
        p2.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s17,
        "Slide 17 proves our dual compliance architecture. We are currently catering to the Indian domestic market, "
        "complying strictly with The Factories Act 1948, IS 800, and PESO SMPV Rules 2016. "
        "Simultaneously, our engineering packages map 1:1 to ASME Section VIII, ISO 22734, and European PED standards, "
        "allowing Indian fabricators like L&T and BHEL to export identical skids globally."
    )

    # =========================================================================
    # SLIDE 18: COMMERCIAL MONETIZATION PORTFOLIO (LIGHT THEME)
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    set_slide_background(s18, BG_LIGHT)
    add_header(s18, "ASD-STM100 COMMERCIAL DELIVERY SUITE", "Transparent Service Delivery Tiers & Immediate Commercialization")

    t_w = Inches(3.6)
    t_gap = Inches(0.4)

    # Tier 1 Card
    c18_1 = add_card(s18, Inches(0.8), Inches(1.6), t_w, Inches(5.3), bg_color=CARD_BG, border_color=CYAN)
    tf18_1 = c18_1.text_frame
    tf18_1.word_wrap = True
    p = tf18_1.paragraphs[0]
    p.text = "TIER 1 • FORENSIC AUDIT"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p = tf18_1.add_paragraph()
    p.text = "Plant Health Audit & CAE"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p = tf18_1.add_paragraph()
    p.text = "$7,500 / ₹6.2 Lakh\n5 Business Days Remote Turnaround"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p = tf18_1.add_paragraph()
    p.text = "\n• 100h remote SCADA ingestion\n• Butler-Volmer Tafel slope & ECSA decay attribution\n• ADVEKT CFD multiphase flow check\n• ASME Sec VIII UG-27 shell thickness validation\n• 25-page deterministic forensic dossier\n• Target: Brownfield plant operators"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    # Tier 2 Card
    c18_2 = add_card(s18, Inches(0.8) + t_w + t_gap, Inches(1.6), t_w, Inches(5.3), bg_color=CARD_BG, border_color=EMERALD)
    tf18_2 = c18_2.text_frame
    tf18_2.word_wrap = True
    p = tf18_2.paragraphs[0]
    p.text = "TIER 2 • EDGE FIRMWARE"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p = tf18_2.add_paragraph()
    p.text = "AERO-H2 Edge Digital Twin"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p = tf18_2.add_paragraph()
    p.text = "$2,500/mo or $25k Perpetual\nAnnual SaaS or One-Time OEM License"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p = tf18_2.add_paragraph()
    p.text = "\n• Deployed on Siemens S7-1500 / Jetson\n• In-operando <1.4s biofilm anomaly trip\n• Physics-constrained voltage forecasting\n• +39.4% core stack life extension\n• Eliminates chemical biocide wash loops\n• Modbus TCP / OPC-UA integration\n• 99.9% uptime SLA commitment"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    # Tier 3 Card
    c18_3 = add_card(s18, Inches(0.8) + 2*(t_w + t_gap), Inches(1.6), t_w, Inches(5.3), bg_color=CARD_BG, border_color=AMBER)
    tf18_3 = c18_3.text_frame
    tf18_3.word_wrap = True
    p = tf18_3.paragraphs[0]
    p.text = "TIER 3 • EPC FABRICATION"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = AMBER
    p = tf18_3.add_paragraph()
    p.text = "Turnkey DFM CAD License"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p = tf18_3.add_paragraph()
    p.text = "$45,000 / ₹37.5 Lakh\nOptional $280k Full 100 TPD BEDP"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p = tf18_3.add_paragraph()
    p.text = "\n• Complete SolidWorks 2025 assembly\n• AutoCAD ISO A1 shop drawings (DWG/PDF)\n• Progressive stamping die drawings (Ti)\n• ASME Section VIII Div 1/3 calculations\n• The Factories Act mezzanine design\n• Complete vendor procurement BOM\n• Target: EPC fabricators (L&T, Technip)"
    p.font.size = Pt(10)
    p.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s18,
        "Slide 18 presents our commercial delivery portfolio structured for Indian and global EPC operators. "
        "Tier 1 offers low-friction 5-day forensic audits ($7,500 / ₹6.2 Lakh). "
        "Tier 2 provides recurring edge firmware SaaS ($2,500/mo) or perpetual OEM licenses. "
        "Tier 3 licenses complete SolidWorks and AutoCAD manufacturing packages ($45,000 / ₹37.5 Lakh) for turnkey fabrication."
    )

    # =========================================================================
    # SLIDE 19: MASTER PROCUREMENT BOM (LIGHT THEME)
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    set_slide_background(s19, BG_LIGHT)
    add_header(s19, "SUPPLY CHAIN & PROCUREMENT", "Tier-1 Vendor Procurement Bill of Materials (BOM)")

    c19_l = add_card(s19, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tf19_l = c19_l.text_frame
    tf19_l.word_wrap = True
    p = tf19_l.paragraphs[0]
    p.text = "MECHANICAL & STRUCTURAL HARDWARE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    bom_mech = [
        ("Structural Steel Beams:", "Tata Steel ISMB 300 / IS 2062 Grade B (Hot-Dip Galvanized 85 µm)"),
        ("Bipolar Flow Plates:", "Sandvik / ATI ASTM B265 Grade 2 Titanium Sheet (0.80mm cold-rolled)"),
        ("Separator Pressure Vessels:", "Larsen & Toubro Heavy Engineering (SA-240 Type 316L, 14.0mm plate)"),
        ("Catalytic DeOxo Reactor:", "BASF / Johnson Matthey 0.5% wt Pd/Al2O3 extruded catalyst pellets"),
        ("High-Pressure Compressors:", "PDC Machines / Hofer Metal Diaphragm Compressor (350 bar rated)"),
        ("Buffer Accumulator Cascade:", "Everest Kanto Cylinder (EKC) / IS 7285 Type 1 seamless Cr-Mo steel"),
        ("Process Valves & Fittings:", "Swagelok 316L medium-pressure cone & thread + Burkert automated ESDs")
    ]
    for h, b in bom_mech:
        p = tf19_l.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    c19_r = add_card(s19, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tf19_r = c19_r.text_frame
    tf19_r.word_wrap = True
    p = tf19_r.paragraphs[0]
    p.text = "ELECTRICAL, CONTROLS & MEMBRANE ELECTRODE ASSEMBLIES"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    bom_elec = [
        ("Solid Electrolyte Membrane:", "Chemours Nafion HP / N117 ePTFE reinforced composite (85 µm)"),
        ("Porous Transport Layers:", "Bekaert Titanium sintered fiber felt with microporous layer (PTL)"),
        ("Safety PLC Controller:", "Siemens SIMATIC S7-1500F Fail-Safe CPU 1518F-4 PN/DP"),
        ("DC Power Rectifier:", "ABB / Schneider Electric 12-Pulse Thyristor Rectifier (5000A DC)"),
        ("High-Purity Busbars:", "Hindalco C11000 Electrolytic Tough Pitch (ETP) 99.9% solid copper"),
        ("NDIR Hydrogen Transmitters:", "Crowcon / Draeger Polytron 8720 fast NDIR gas detection (<500ms)"),
        ("Water Purification Pumps:", "Grundfos CRN 316L multi-stage centrifugal sanitary pumps")
    ]
    for h, b in bom_elec:
        p = tf19_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s19,
        "Slide 19 details our procurement Bill of Materials. "
        "Every single subsystem is specified with verified industrial tier-1 suppliers, "
        "incorporating Tata Steel, Siemens, Swagelok, Chemours, and ABB parts to ensure immediate commercial feasibility."
    )

    # =========================================================================
    # SLIDE 20: CONCLUSIONS & ROADMAP (LIGHT THEME)
    # =========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    set_slide_background(s20, BG_LIGHT)
    add_header(s20, "12. Summary & Industrial Tender Readiness", "Conclusions & Commercial Implementation Roadmap")

    c20_1 = add_card(s20, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), bg_color=CARD_BG, border_color=EMERALD)
    tf20_1 = c20_1.text_frame
    tf20_1.word_wrap = True
    p = tf20_1.paragraphs[0]
    p.text = "CORE SCIENTIFIC & TECHNICAL CONCLUSIONS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    c_pts = [
        ("Virtual Sensing Accuracy:", "Demonstrated that GPU-resident XGBoost models reverse-engineer initial catalyst morphology with sub-microvolt accuracy (1.911 µV/h error, R² = 0.963)."),
        ("Thermodynamic Generalization:", "Proved that constraining neural transformers with Butler-Volmer kinetics eliminates unphysical degradation forecasting (NRMSE = 0.054)."),
        ("Decoupled Biological Defense:", "Established that dielectric EPS biofilm anomalies can be decoupled from Ostwald ripening in under 1.40s, triggering autonomous DI water recovery."),
        ("Techno-Economic Viability:", "+39.4% core stack life extension translates directly into -$0.42/kg H2 Levelized Cost reduction and a 4.8-month payback.")
    ]
    for h, b in c_pts:
        p = tf20_1.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    c20_2 = add_card(s20, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=CARD_BG, border_color=CYAN)
    tf20_2 = c20_2.text_frame
    tf20_2.word_wrap = True
    p = tf20_2.paragraphs[0]
    p.text = "IMMEDIATE INDUSTRIAL ONBOARDING PROTOCOL"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN

    proto_pts = [
        ("Step 1: NDA & 100-Hour SCADA Ingestion:", "Execute mutual NDA; client provides 100h min-resolution operational CSV logs."),
        ("Step 2: 5-Day Diagnostic CAE Audit:", "HYDROSENSE autonomous ADVEKT engine deconvolutes Tafel slope and returns validated ASME/PESO dossier in 5 days."),
        ("Step 3: Edge Twin Deployment:", "Compile containerized C++ runtime onto on-prem Siemens S7-1500F PLC for zero-risk side-by-side passive tracking."),
        ("Step 4: Autonomous Mitigation Commissioning:", "Engage automated DI surge solenoid interlock after baseline calibration."),
        ("Contact & Institutional Office:", "Sourish Senapati (Corresponding Author)\nDept. of Chemical Engineering, Jadavpur University\nsourishs.chem.ug@jadavpuruniversity.in | ORCID: 0009-0000-8799-4340")
    ]
    for h, b in proto_pts:
        p = tf20_2.add_paragraph()
        p.text = f"\n• {h} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(s20,
        "In conclusion, our research provides the decisive missing digital and manufacturing infrastructure for commercial green hydrogen. "
        "By extending stack life by 39.4%, eliminating chemical wash loops, and proving ASME, IS 800, and PESO compliance, "
        "we make green hydrogen commercially defensible today. Thank you for your time, and we welcome your questions."
    )

    prs.save(output_path)
    print(f"[SUCCESS] 20-Slide Master PowerPoint successfully generated: {output_path}")

if __name__ == '__main__':
    create_presentation()
