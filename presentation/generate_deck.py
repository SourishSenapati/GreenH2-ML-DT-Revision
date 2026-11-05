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
    
    # Colors
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
    
    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, category, title, dark=False):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = EMERALD if dark else CYAN
        
        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT if dark else TEXT_DARK

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
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

    # ==========================================
    # SLIDE 1: TITLE SLIDE (Dark Navy Theme)
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, NAVY_DARK)
    
    # Pill Tag
    pill = add_card(slide1, Inches(0.8), Inches(0.8), Inches(3.2), Inches(0.4), bg_color=NAVY_CARD, border_color=EMERALD)
    tf_p = pill.text_frame
    tf_p.paragraphs[0].text = "ALTFS MANUSCRIPT DEFENSE"
    tf_p.paragraphs[0].font.size = Pt(11)
    tf_p.paragraphs[0].font.bold = True
    tf_p.paragraphs[0].font.color.rgb = EMERALD
    tf_p.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Main Title
    t_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Machine Learning for Predicting Catalyst Efficiency in Green Hydrogen Production"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    
    p2 = tf1.add_paragraph()
    p2.text = "Predictive Maintenance, Fault Detection, and Closed-Loop Degradation Mitigation"
    p2.font.size = Pt(20)
    p2.font.color.rgb = CYAN
    p2.font.bold = True

    # Authors Card
    auth_card = add_card(slide1, Inches(0.8), Inches(3.5), Inches(5.6), Inches(3.2), bg_color=NAVY_CARD, border_color=CARD_BORDER)
    tf_a = auth_card.text_frame
    tf_a.word_wrap = True
    p = tf_a.paragraphs[0]
    p.text = "RESEARCH AUTHORS & AFFILIATIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    
    p = tf_a.add_paragraph()
    p.text = "\nSourish Senapati¹* (Corresponding Author)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT
    
    p = tf_a.add_paragraph()
    p.text = "¹Department of Chemical Engineering, Jadavpur University, Kolkata\nEmail: sourishs.chem.ug@jadavpuruniversity.in\nORCID: 0009-0000-8799-4340\n"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED
    
    p = tf_a.add_paragraph()
    p.text = "Srestha Biswash²"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT
    
    p = tf_a.add_paragraph()
    p.text = "²Dept. of Microbiology & Biotechnology, Sister Nivedita University\nORCID: 0009-0006-4345-2025"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    # Core Breakthrough Highlights Card
    high_card = add_card(slide1, Inches(6.8), Inches(3.5), Inches(5.7), Inches(3.2), bg_color=NAVY_CARD, border_color=CARD_BORDER)
    tf_h = high_card.text_frame
    tf_h.word_wrap = True
    p = tf_h.paragraphs[0]
    p.text = "EXECUTIVE ARCHITECTURE HIGHLIGHTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CYAN
    
    metrics = [
        ("• Phase 1 Materials Informatics:", "GPU-Resident XGBoost (RTX 4050 VRAM) with R² = 0.9287 and sub-microvolt precision (1.911 µV/h)."),
        ("• Phase 2 Degradation Tracking:", "Physics-Informed Bayesian Transformer (PIBT) with Butler-Volmer thermodynamic loss constraint (NRMSE = 0.5428)."),
        ("• Decoupled Fault Mitigation:", "Dynamic 99.5th percentile Bayesian threshold isolates EPS biofilm matrix fouling from Ostwald ripening (F1 = 0.990)."),
        ("• Industrial Techno-Economics:", "+39.4% Operational Stack Lifespan Extension (21,428h to 29,879h), substantially reducing Levelized Cost of Hydrogen (LCOH).")
    ]
    for m_title, m_desc in metrics:
        p = tf_h.add_paragraph()
        p.text = f"\n{m_title} "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        run = p.add_run()
        run.text = m_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide1, 
        "Good morning/afternoon, esteemed committee members, reviewers, and colleagues. "
        "I am Sourish Senapati from Jadavpur University's Department of Chemical Engineering. "
        "Together with my co-author Srestha Biswash from Sister Nivedita University, I am presenting our work "
        "on developing a multidisciplinary digital twin framework that bridges materials informatics, "
        "physics-informed deep learning, and industrial microbiology to solve critical bottlenecks in green hydrogen electrolysis."
    )

    # ==========================================
    # SLIDE 2: MOTIVATION & LCOH BOTTLENECK
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, BG_LIGHT)
    add_header(slide2, "1. Industry Background & Problem Statement", "The Commercial Barrier: Levelized Cost of Hydrogen (LCOH)")

    # 3 Cards Layout
    c1 = add_card(slide2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(4.3))
    tf = c1.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "CAPEX & Precious Metals"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = EMERALD
    
    p = tf.add_paragraph()
    p.text = "\n• PEM Water Electrolysis is critical for green H₂ from intermittent renewables (solar/wind).\n\n• Stack CAPEX is heavily dominated by scarce noble-metal electrocatalysts:\n  - Platinum (Pt) at cathode (HER)\n  - Iridium (IrO₂) at anode (OER)\n\n• Noble metal market volatility directly dictates plant financial feasibility."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    c2 = add_card(slide2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(4.3))
    tf = c2.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "Unplanned Downtime Crisis"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = ROSE
    
    p = tf.add_paragraph()
    p.text = "\n• Industrial stacks currently operate under reactive 'run-to-failure' paradigms.\n\n• Operating on fluctuating renewable power exacerbates dynamic stress.\n\n• Unplanned plant shutdowns cause massive efficiency losses, emergency tear-downs, and multiplied maintenance OPEX (up to 3×)."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    c3 = add_card(slide2, Inches(8.8), Inches(1.6), Inches(3.7), Inches(4.3))
    tf = c3.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "The Virtual Twin Solution"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = CYAN
    
    p = tf.add_paragraph()
    p.text = "\n• Destructive physical cell autopsies are economically unviable in commercial multi-MW arrays.\n\n• Proposed: An in-operando Digital Twin using macro telemetry (voltage, current, temperature) to:\n  1. Infer nanoscale morphology\n  2. Predict degradation over time\n  3. Detect catastrophic anomalies\n  4. Proactively shed load."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    # Bottom Callout Banner
    bot_card = add_card(slide2, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8), bg_color=EMERALD_LIGHT, border_color=EMERALD)
    tf_b = bot_card.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Core Thesis: Extending catalyst lifetime by 35-40% through AI-driven active mitigation fundamentally alters the economics of Levelized Cost of Hydrogen (LCOH)."
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p.alignment = PP_ALIGN.CENTER

    add_speaker_notes(slide2,
        "To make green hydrogen commercially competitive against fossil-derived grey hydrogen, we must tackle LCOH. "
        "A disproportionate share of stack capital expenditure stems from Platinum and Iridium electrocatalysts. "
        "In commercial plants, unexpected cell degradation forces premature stack replacements, multiplying maintenance OPEX. "
        "Because tearing down pressurized multi-megawatt stacks for physical inspection is impossible during operation, "
        "we need an intelligent digital twin to monitor and protect catalyst integrity purely from operational sensor data."
    )

    # ==========================================
    # SLIDE 3: THE DUAL DEGRADATION DILEMMA
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, BG_LIGHT)
    add_header(slide3, "2. Electrochemical & Biological Physics", "The Dual Degradation Dilemma: Ostwald Ripening vs. EPS Biofilms")

    card_left = add_card(slide3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tf_l = card_left.text_frame
    tf_l.word_wrap = True
    tf_l.paragraphs[0].text = "MECHANISM 1: CHEMICAL WEAR\nNanoscale Ostwald Ripening"
    tf_l.paragraphs[0].font.size = Pt(16)
    tf_l.paragraphs[0].font.bold = True
    tf_l.paragraphs[0].font.color.rgb = CYAN
    
    p = tf_l.add_paragraph()
    p.text = "\n• Driver: Intermittent renewable load cycling creates dynamic potential swings.\n\n• Physical Process: Smaller catalyst nanoparticles thermodynamically dissolve due to higher surface energy and redeposit onto larger crystallites.\n\n• Microscopic Result: Rapid shrinkage of Electrochemically Active Surface Area (ECSA).\n\n• Macroscopic Signal: Insidious, slow cell voltage creep (~1 to 2 µV/h) over 20,000+ hours.\n\n• Architectural Role: Serves as the system's active baseline noise floor."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    card_right = add_card(slide3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tf_r = card_right.text_frame
    tf_r.word_wrap = True
    tf_r.paragraphs[0].text = "MECHANISM 2: BIOLOGICAL INTERFERENCE\nMicrobial Biofouling (Pseudomonas spp.)"
    tf_r.paragraphs[0].font.size = Pt(16)
    tf_r.paragraphs[0].font.bold = True
    tf_r.paragraphs[0].font.color.rgb = ROSE
    
    p = tf_r.add_paragraph()
    p.text = "\n• Driver: Recirculating ultra-pure Type 1 deionized feedwater in oligotrophic environments.\n\n• Biological Process: Oligocarbotolerant bacteria (Pseudomonas spp.) attach to porous transport layers (PTL) and fluoropolymer membranes.\n\n• Secretion: Rapid deposition of an Exopolysaccharide (EPS) biofilm matrix.\n\n• Physical Impact: Forms an insulating dielectric barrier and mass-transport obstruction.\n\n• Macroscopic Signal: Sudden non-linear ohmic heating and pressure surges that mimic catastrophic mechanical rupture!"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    add_speaker_notes(slide3,
        "A critical innovation of this paper is recognizing that PEM stacks degrade through two decoupled phenomena. "
        "First is Ostwald ripening: under fluctuating power, tiny catalyst nanoparticles thermodynamically dissolve and merge, "
        "shrinking active surface area and causing a slow microvolt-per-hour voltage drift. "
        "Second is biological biofouling. Even in ultrapure deionized water, oligotrophic bacteria like Pseudomonas colonize the porous layers "
        "and secrete an exopolysaccharide biofilm. This biofilm acts as an electrical insulator and mass-flow barrier, causing sudden voltage and pressure spikes. "
        "Our digital twin must cleanly decouple this sudden biological anomaly from gradual chemical wear."
    )

    # ==========================================
    # SLIDE 4: SYSTEM ARCHITECTURE FLOWCHART
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, BG_LIGHT)
    add_header(slide4, "3. Methodology & System Engineering", "Dual-Layer Digital Twin Framework: End-to-End Pipeline")

    # 4 horizontal progression cards
    step_width = Inches(2.7)
    gap = Inches(0.25)
    
    steps = [
        ("LAYER 0: SENSING", "In-Operando Telemetry", 
         "• Cell Voltage V(t)\n• Current Density j(t)\n• Operating Temp T\n• Cathode/Anode p\n• Local Thermal Gradients\n\nNo destructive cell autopsies required.", 
         TEXT_DARK),
        ("PHASE 1: INFORMATICS", "XGBoost Virtual Sensor", 
         "• Resident in GPU VRAM (RTX 4050)\n• Zero PCIe latency\n• Evaluates Tafel Slope, ECSA, Porosity\n• Sub-microvolt precision\n• R² = 0.9287", 
         CYAN),
        ("PHASE 2: PROGNOSTICS", "Physics-Informed PIBT", 
         "• Multi-Head Self-Attention\n• Butler-Volmer NLL loss constraint\n• Monte Carlo Epistemic UQ\n• Dynamic FP16 variance clamping\n• NRMSE = 0.5428", 
         EMERALD),
        ("SAFETY CONTROL", "Decoupled Mitigation", 
         "• Active noise floor tracks Ostwald wear\n• Dynamic 99.5% Bayesian threshold catches EPS biofilms (F1 = 0.990)\n• Closed-loop load shedding\n• +39.4% Lifespan", 
         ROSE)
    ]
    
    for i, (tag, stitle, desc, color) in enumerate(steps):
        s_left = Inches(0.8) + i * (step_width + gap)
        sc = add_card(slide4, s_left, Inches(1.6), step_width, Inches(4.3))
        tf = sc.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = tag
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = color
        
        p = tf.add_paragraph()
        p.text = stitle
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        
        p = tf.add_paragraph()
        p.text = f"\n{desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    # Bottom summary box
    bot4 = add_card(slide4, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.8), bg_color=CYAN_LIGHT, border_color=CYAN)
    tf4 = bot4.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "Hybrid Integration: Combines static materials informatics (initial morphology ranking) with dynamic physics-informed deep learning (time-series prognosis) into a single unified supervisory loop."
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.alignment = PP_ALIGN.CENTER

    add_speaker_notes(slide4,
        "Here is our complete digital twin workflow. Operational sensor data flows into our Phase 1 GPU-resident XGBoost model, "
        "which acts as a virtual sensor to determine baseline catalyst morphology without taking the cell apart. "
        "Next, Phase 2 deploys our Physics-Informed Bayesian Transformer to model degradation across time. "
        "By enforcing Butler-Volmer kinetics in the loss function, the Transformer cannot produce unphysical forecasts. "
        "Finally, our decoupled Bayesian threshold triggers proactive load curtailment whenever acute EPS biofouling is flagged, "
        "protecting the membrane and extending stack life by nearly 40%."
    )

    # ==========================================
    # SLIDE 5: PHASE 1 - XGBOOST FEATURE IMPORTANCE (FIGURE 1)
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, BG_LIGHT)
    add_header(slide5, "4. Materials Informatics (Phase 1)", "Virtual Sensing: Reverse-Engineering Latent Nanoscale Morphology")

    # Image Left
    f1_path = os.path.join(assets_dir, "Fig1_XGBoost_Feature_Importance.png")
    if os.path.exists(f1_path):
        slide5.shapes.add_picture(f1_path, Inches(0.8), Inches(1.6), width=Inches(6.2))

    # Explanation Right
    r_card = add_card(slide5, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2))
    tf_r = r_card.text_frame
    tf_r.word_wrap = True
    tf_r.paragraphs[0].text = "GPU-Resident XGBoost Regressor"
    tf_r.paragraphs[0].font.size = Pt(16)
    tf_r.paragraphs[0].font.bold = True
    tf_r.paragraphs[0].font.color.rgb = CYAN

    points5 = [
        ("Hardware Optimization:", "Deployed entirely in NVIDIA RTX 4050 VRAM, eliminating PCIe host-to-device transfer latency (< 0.4 ms query)."),
        ("Regularized Objective:", "L(t) = ∑ l(y_i, ŷ_i) + γT + 0.5λ||w||² actively penalizes tree complexity, isolating real physical features from synthesis noise."),
        ("Dominant Morphological Descriptors:", "Tafel Slope, Electrochemically Active Surface Area (ECSA), and Porosity Index account for OVER 74% of cumulative model gain!"),
        ("Five-Fold Cross-Validation:", "Error bars confirm consistent ranking across randomized data folds."),
        ("Core Takeaway:", "Macroscopic operational variables carry sufficient latent information to accurately evaluate catalyst degradation states without destructive physical cell teardowns.")
    ]
    for h, b in points5:
        p = tf_r.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide5,
        "Figure 1 displays the normalized XGBoost feature importance scores across 5-fold cross validation. "
        "Notice that Tafel slope, ECSA, and Porosity Index collectively account for over 74% of the total predictive gain. "
        "This is an intuitive and physically sound result: Tafel slope reflects charge-transfer activation resistance, "
        "ECSA measures available catalytic sites, and porosity dictates two-phase gas/water mass transport. "
        "By hosting this model directly inside GPU VRAM, we eliminate PCIe latency bottlenecks, enabling continuous real-time virtual sensing."
    )

    # ==========================================
    # SLIDE 6: PREDICTION FIDELITY - PARITY PLOT (FIGURE 2)
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, BG_LIGHT)
    add_header(slide6, "4. Materials Informatics (Phase 1)", "Sub-Microvolt Accuracy in Catalyst Decay Rate Prediction")

    # Image Left
    f2_path = os.path.join(assets_dir, "Fig2_Parity_Plot_Predicted_vs_Empirical.png")
    if os.path.exists(f2_path):
        slide6.shapes.add_picture(f2_path, Inches(0.8), Inches(1.6), width=Inches(5.4))

    # Right Content Cards
    m1 = add_card(slide6, Inches(6.5), Inches(1.6), Inches(6.0), Inches(1.5), bg_color=EMERALD_LIGHT, border_color=EMERALD)
    tf_m1 = m1.text_frame
    tf_m1.word_wrap = True
    tf_m1.paragraphs[0].text = "QUANTITATIVE ACCURACY METRICS (TEST SET)"
    tf_m1.paragraphs[0].font.size = Pt(11)
    tf_m1.paragraphs[0].font.bold = True
    tf_m1.paragraphs[0].font.color.rgb = EMERALD
    
    p = tf_m1.add_paragraph()
    p.text = "R² = 0.9287   |   RMSE = 1.9110 µV/h"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    c_r6 = add_card(slide6, Inches(6.5), Inches(3.3), Inches(6.0), Inches(3.5))
    tf_r6 = c_r6.text_frame
    tf_r6.word_wrap = True
    tf_r6.paragraphs[0].text = "Statistical & Physical Validation"
    tf_r6.paragraphs[0].font.size = Pt(15)
    tf_r6.paragraphs[0].font.bold = True
    tf_r6.paragraphs[0].font.color.rgb = CYAN

    points6 = [
        ("Strict 1:1 Unity Alignment:", "Predictions tightly cluster around the ideal diagonal with zero systematic over- or under-prediction bias."),
        ("Homoscedastic Error Distribution:", "No fanning out or variance inflation at extreme degradation rates (both low and high stress regimes remain equally accurate)."),
        ("Sub-Microvolt Resolution:", "RMSE of 1.91 µV/h confirms that virtual sensing can resolve subtle degradation differences that dictate thousands of hours of stack longevity."),
        ("Zero Physical Disruption:", "Eliminates the multi-thousand dollar cost of disassembling membrane electrode assemblies (MEAs) for ex-situ characterization.")
    ]
    for h, b in points6:
        p = tf_r6.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide6,
        "Figure 2 illustrates the parity plot comparing our virtual sensor's predicted degradation rates against empirical ground-truth in microvolts per hour. "
        "The model achieves an R-squared of 0.9287 and an RMSE of 1.9110 µV/h on held-out test data. "
        "Notice the homoscedastic spread: errors do not expand at higher degradation rates. "
        "Achieving sub-microvolt resolution confirms that operational telemetry can serve as an exceptionally high-fidelity proxy for catalyst state."
    )

    # ==========================================
    # SLIDE 7: PHASE 2 - PIBT ARCHITECTURE & BUTLER-VOLMER CONSTRAINT
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, BG_LIGHT)
    add_header(slide7, "5. Deep Learning Formulation (Phase 2)", "Physics-Informed Bayesian Transformer (PIBT) Architecture")

    # 3 Cards Layout
    c1 = add_card(slide7, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.paragraphs[0].text = "Multi-Head Self-Attention"
    tf1.paragraphs[0].font.size = Pt(15)
    tf1.paragraphs[0].font.bold = True
    tf1.paragraphs[0].font.color.rgb = CYAN
    
    p = tf1.add_paragraph()
    p.text = "\nAttention(Q,K,V) = softmax(QKᵀ / √d_k) V\n\n• Scaled dot-product attention correlates operational states across distant horizons (tens of thousands of operating steps).\n\n• Completely overcomes the vanishing gradient and memory dilution bottlenecks inherent to recurrent LSTM/GRU architectures.\n\n• √d_k prevents softmax saturation into flat gradient regions."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    c2 = add_card(slide7, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.paragraphs[0].text = "Butler-Volmer Physics Loss"
    tf2.paragraphs[0].font.size = Pt(15)
    tf2.paragraphs[0].font.bold = True
    tf2.paragraphs[0].font.color.rgb = EMERALD
    
    p = tf2.add_paragraph()
    p.text = "\nL_total = L_NLL + λ || η̂ - η_BV ||²\n\nwhere:\nη_BV = arcsinh(j / 2i₀) · (2RT / zF)\n\n• Enforces electrochemical charge transfer kinetics directly during backpropagation.\n\n• Prevents unphysical overpotentials or negative resistance values.\n\n• Anchors steady-state predictions to true thermodynamic manifolds."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    c3 = add_card(slide7, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2))
    tf3 = c3.text_frame
    tf3.word_wrap = True
    tf3.paragraphs[0].text = "AMP & Variance Clamping"
    tf3.paragraphs[0].font.size = Pt(15)
    tf3.paragraphs[0].font.bold = True
    tf3.paragraphs[0].font.color.rgb = ROSE
    
    p = tf3.add_paragraph()
    p.text = "\nHardware Tensor Core Acceleration:\n\n• Automatic Mixed Precision (AMP FP16) for real-time sub-millisecond execution.\n\n• Dynamic Variance Clamping:\n  Clamps aleatoric variance σ̂² during backward passes to prevent mathematical variance collapse / NaN overflow.\n\n• AdamW + Cosine Annealing Warm Restarts to escape localized saddle points."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    add_speaker_notes(slide7,
        "Slide 7 details our Phase 2 neural engine: the Physics-Informed Bayesian Transformer. "
        "Standard sequence models like LSTMs suffer from catastrophic forgetting over thousands of operational hours. "
        "Transformers solve this via Multi-Head Self-Attention, computing dependencies across the entire timeline simultaneously. "
        "To keep the model scientifically valid, we penalize the negative log-likelihood loss with the inverted Butler-Volmer equation. "
        "Furthermore, running in 16-bit Automatic Mixed Precision requires our novel dynamic variance clamping to eliminate gradient overflows."
    )

    # ==========================================
    # SLIDE 8: TIME-SERIES PROGNOSTIC DEGRADATION (FIGURE 3)
    # ==========================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, BG_LIGHT)
    add_header(slide8, "5. Deep Learning Formulation (Phase 2)", "Time-Series Prognostic Degradation Tracking & Uncertainty Quantification")

    # Image Left
    f3_path = os.path.join(assets_dir, "Fig3_PIBT_Time_Series_Degradation_Forecast.png")
    if os.path.exists(f3_path):
        slide8.shapes.add_picture(f3_path, Inches(0.8), Inches(1.6), width=Inches(6.2))

    # Explanation Right
    r8 = add_card(slide8, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2))
    tf8 = r8.text_frame
    tf8.word_wrap = True
    tf8.paragraphs[0].text = "PIBT Tracking Performance"
    tf8.paragraphs[0].font.size = Pt(16)
    tf8.paragraphs[0].font.bold = True
    tf8.paragraphs[0].font.color.rgb = CYAN

    points8 = [
        ("Normalized Error Metric:", "NRMSE = 0.5428 across the scaled multi-dimensional feature space."),
        ("Sustained Trajectory Convergence:", "Solid blue curve (PIBT point-estimate) tracks dashed black curve (empirical ground truth) across dynamic load cycles without divergence."),
        ("Bayesian Uncertainty Envelope:", "Shaded cyan band denotes 95th percentile aleatoric uncertainty quantified via stochastic Monte Carlo (MC) inference passes."),
        ("Physics-Anchored Narrowing:", "Notice how the uncertainty envelope narrows during steady-state phases: proof that the Butler-Volmer constraint actively anchors the predictive manifold."),
        ("Non-Sequential Correlation:", "Self-attention successfully maps slow Ostwald ripening transitions across long time horizons.")
    ]
    for h, b in points8:
        p = tf8.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide8,
        "Figure 3 shows our PIBT time-series prognostic forecast. The solid blue line is the model forecast, "
        "the dashed black line is the empirical ground truth, and the shaded envelope is the 95th percentile uncertainty interval. "
        "With an NRMSE of 0.5428, the Transformer tracks degradation accurately without divergence. "
        "Notice how the uncertainty envelope contracts during steady-state operation; that demonstrates the physics constraint actively bounding the model's confidence."
    )

    # ==========================================
    # SLIDE 9: DECOUPLED FAULT DETECTION - EPS BIOFILMS (FIGURE 4)
    # ==========================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, BG_LIGHT)
    add_header(slide9, "6. Anomaly Detection & Safety", "Decoupled Fault Mitigation: Isolating EPS Biofilm Anomalies")

    # Image Left
    f4_path = os.path.join(assets_dir, "Fig4_Bayesian_Anomaly_Detection_EPS_Biofilm.png")
    if os.path.exists(f4_path):
        slide9.shapes.add_picture(f4_path, Inches(0.8), Inches(1.6), width=Inches(6.2))

    # Explanation Right
    r9 = add_card(slide9, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.2))
    tf9 = r9.text_frame
    tf9.word_wrap = True
    tf9.paragraphs[0].text = "Decoupled Safety Architecture"
    tf9.paragraphs[0].font.size = Pt(16)
    tf9.paragraphs[0].font.bold = True
    tf9.paragraphs[0].font.color.rgb = ROSE

    points9 = [
        ("Upper Panel (Residual Noise Floor):", "Raw normalized voltage residual signal e(t) = |V(t) - μ̂(t)| captures background Ostwald ripening without false alarms."),
        ("Lower Panel (Bayesian Threshold):", "Dynamic 99.5th percentile probabilistic boundary automatically adjusts to current operating load."),
        ("Injected EPS Dielectric Markers:", "Acute non-linear resistance jumps mimic exopolysaccharide biofilm matrices formed by Pseudomonas spp. bacteria."),
        ("Out-of-Distribution F1-Score: 0.990:", "Achieved on an isolated holdout test set with near-perfect True Positive (red) flags and zero False Alarms (green normal regions)."),
        ("Actionable Output:", "Instantly triggers supervisory current density curtailment before membrane tearing occurs.")
    ]
    for h, b in points9:
        p = tf9.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide9,
        "Figure 4 highlights our decoupled fault mitigation system. The upper panel shows the raw voltage residual, "
        "which reflects normal background Ostwald ripening wear. In the lower panel, we enforce a dynamic 99.5th percentile Bayesian threshold. "
        "When an acute biofouling event occurs, where Pseudomonas bacteria secrete an insulating EPS biofilm, "
        "the signal pierces the threshold instantly. Evaluated on an out-of-distribution test set, the model achieved an F1-score of 0.990. "
        "This allows operators to trigger automated load shedding in real time."
    )

    # ==========================================
    # SLIDE 10: TECHNO-ECONOMIC IMPACT - 39.4% EXTENSION (FIGURE 5)
    # ==========================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, BG_LIGHT)
    add_header(slide10, "7. Techno-Economic Feasibility", "Translating Anomaly Detection into +39.4% Stack Lifespan Extension")

    # Image Left
    f5_path = os.path.join(assets_dir, "Fig5_Lifespan_Extension_Techno_Economic.png")
    if os.path.exists(f5_path):
        slide10.shapes.add_picture(f5_path, Inches(0.8), Inches(1.6), width=Inches(5.4))

    # Right Content Cards
    m10 = add_card(slide10, Inches(6.5), Inches(1.6), Inches(6.0), Inches(1.5), bg_color=EMERALD_LIGHT, border_color=EMERALD)
    tf10 = m10.text_frame
    tf10.word_wrap = True
    tf10.paragraphs[0].text = "OPERATIONAL LIFESPAN COMPARISON"
    tf10.paragraphs[0].font.size = Pt(11)
    tf10.paragraphs[0].font.bold = True
    tf10.paragraphs[0].font.color.rgb = EMERALD
    
    p = tf10.add_paragraph()
    p.text = "21,428 hrs  ➔  29,879 hrs (+39.4%)"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    c_r10 = add_card(slide10, Inches(6.5), Inches(3.3), Inches(6.0), Inches(3.5))
    tf_r10 = c_r10.text_frame
    tf_r10.word_wrap = True
    tf_r10.paragraphs[0].text = "Economic & Operational Impact"
    tf_r10.paragraphs[0].font.size = Pt(15)
    tf_r10.paragraphs[0].font.bold = True
    tf_r10.paragraphs[0].font.color.rgb = CYAN

    points10 = [
        ("+8,451 Additional Operating Hours:", "Extends stack longevity by over 1 full year of continuous 24/7 industrial duty."),
        ("Drastic Catalyst CAPEX Amortization:", "Postpones expensive Platinum and Iridium recoating / MEA stack overhaul cycles by nearly 40%."),
        ("Substantial LCOH Reduction:", "Directly decreases Levelized Cost of Hydrogen ($/kg H₂), making green hydrogen competitive against grey fossil fuels."),
        ("Proactive vs Reactive Paradigm:", "Replaces catastrophic run-to-failure shutdowns with controlled, scheduled maintenance windows.")
    ]
    for h, b in points10:
        p = tf_r10.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide10,
        "Figure 5 shows the techno-economic value of our framework. Under reactive, run-to-failure operation, "
        "a PEM stack lasts approximately 21,428 hours. By utilizing our PIBT anomaly alerts to automatically modulate current density, "
        "we prevent localized hot-spotting and membrane rupture, extending stack lifespan to 29,879 hours—a 39.4% extension. "
        "In a multi-megawatt plant, deferring Platinum and Iridium catalyst replacement cycles by more than 8,400 hours "
        "fundamentally transforms the Levelized Cost of Hydrogen."
    )

    # ==========================================
    # SLIDE 11: CROSS-SECTORAL VALIDATION & GREEN CHEMISTRY (FIGURE 6)
    # ==========================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, BG_LIGHT)
    add_header(slide11, "8. Generalizability & Sustainability", "Cross-Sectoral Validation & Green Chemistry Remediation")

    # Image Left
    f6_path = os.path.join(assets_dir, "Fig6_Cross_Sectoral_Validation.png")
    if os.path.exists(f6_path):
        slide11.shapes.add_picture(f6_path, Inches(0.8), Inches(1.6), width=Inches(5.4))

    # Right Content Cards
    c_r11 = add_card(slide11, Inches(6.5), Inches(1.6), Inches(6.0), Inches(5.2))
    tf_r11 = c_r11.text_frame
    tf_r11.word_wrap = True
    tf_r11.paragraphs[0].text = "Cross-Scale Validation & Ecology"
    tf_r11.paragraphs[0].font.size = Pt(16)
    tf_r11.paragraphs[0].font.bold = True
    tf_r11.paragraphs[0].font.color.rgb = EMERALD

    points11 = [
        ("Independent Cross-Sectoral Validation:", "Tested against independent bench-top galvanostatic datasets from Zenodo repositories, confirming universal multi-scale transferability."),
        ("Physics Constraints Ensure Robustness:", "Because the Butler-Volmer kinetics are hard-coded into the loss, the model does not overfit to stack geometry."),
        ("Preventing Heavy Metal Effluent Leaching:", "Thermodynamic dissolution of Pt and Ir catalysts releases toxic heavy metal leachate into downstream effluent water. Algorithmic load dampening minimizes dissolution."),
        ("Elimination of Chemical Biocides:", "Industrial operators typically flush biofouled stacks with toxic chemical biocides (e.g. quaternary ammonium compounds). Early AI detection eliminates chemical biocide flushes entirely!"),
        ("Non-Chemical Remediation:", "The digital twin acts as an algorithmic environmental remediation tool.")
    ]
    for h, b in points11:
        p = tf_r11.add_paragraph()
        p.text = f"\n• {h} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide11,
        "Figure 6 demonstrates cross-sectoral empirical validation against independent bench-top datasets. "
        "The physics constraints ensure that the model transfers across different stack dimensions. "
        "Furthermore, our system delivers significant Green Chemistry benefits: by preventing rapid catalyst dissolution, "
        "we eliminate cytotoxic Platinum and Iridium leaching into effluent water. And by catching microbial biofilms early, "
        "we eliminate the need to flush stacks with harsh synthetic biocides. The software acts as non-chemical environmental remediation."
    )

    # =========================================================================
    # SLIDE 12: DESIGN FOR MANUFACTURING (DFM) & UNIVERSAL MULTI-PLANT (FIG 7)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, BG_LIGHT)
    add_header(slide12, "HARDWARE PRODUCTION & MULTI-PLANT SCALABILITY", "Design for Manufacturing (DFM) & Universal Industrial Deployment")
    
    # Left Card: DFM Specifications & Multi-Industry Scope
    card12_l = add_card(slide12, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.3))
    tf12_l = card12_l.text_frame
    tf12_l.word_wrap = True
    
    p = tf12_l.paragraphs[0]
    p.text = "DFM ENGINEERING & CODE COMPLIANCE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    
    bullets12 = [
        ("Bipolar Plate Stamping:", "Progressive die stamping of ASTM B265 Gr.2 Titanium. Draft angle θ=7°, R_min=0.3mm to eliminate localized thinning; EPDM O-ring retention groove."),
        ("ASME Section VIII Div 1:", "Knockout Drum & Separators sized via ADVEKT autonomous CAE (SA-240 316L, 12mm wall, MAWP=33.89 bar, 39.0 bar hydrotest)."),
        ("SolidWorks 2025 B-Rep:", "11 reconciled subsystems assembled in SolidWorks (.SLDASM, .STEP) with ISMB 300 steel mezzanine and flanged process headers."),
        ("Multi-Plant Deployment:", "Engineered for plug-and-play retrofit across Refineries (hydrotreating), Chlor-Alkali (brine H2 recovery), Green Steel (DRI reduction), and Heavy Mobility (350 bar).")
    ]
    for b_title, b_desc in bullets12:
        p = tf12_l.add_paragraph()
        p.text = f"\n• {b_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    # Right: Embedded Fig 7 Image
    fig7_path = os.path.join(assets_dir, "Fig7_Design_For_Manufacturing_DFM_Multi_Plant.png")
    if os.path.exists(fig7_path):
        slide12.shapes.add_picture(fig7_path, Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.3))

    add_speaker_notes(slide12,
        "Slide 12 details our Design for Manufacturing framework. "
        "Every single part of our electrolyzer stack and turnkey facility is modeled in SolidWorks 2025 "
        "using standard mill stock and high-throughput fabrication routes. "
        "Our titanium flow fields feature a 7-degree draft angle to allow progressive die stamping without tearing. "
        "Our ASME Section VIII pressure vessels are sized using our ADVEKT calculation engine, "
        "and our modular balance-of-plant drops directly into refineries, chlor-alkali plants, green steel DRI furnaces, and mobility refueling stations."
    )

    # =========================================================================
    # SLIDE 13: SOLIDWORKS 2025 MASTER ASSEMBLY & SUBSYSTEM SCHEDULE
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13, BG_LIGHT)
    add_header(slide13, "TURNKEY FACILITY ENGINEERING • SOLIDWORKS 2025", "100 TPD Commercial Plant Mechanical & Process Architecture")
    
    # Left: SolidWorks Master Render
    sw_render_path = os.path.join(assets_dir, "PEM_Commercial_Plant_Catalog_Render.png")
    if os.path.exists(sw_render_path):
        slide13.shapes.add_picture(sw_render_path, Inches(0.8), Inches(1.6), width=Inches(6.2))
        
    # Right: Equipment Schedule Table / Card
    card13_r = add_card(slide13, Inches(7.2), Inches(1.6), Inches(5.3), Inches(5.3))
    tf13_r = card13_r.text_frame
    tf13_r.word_wrap = True
    p = tf13_r.paragraphs[0]
    p.text = "11 RECONCILED SUBSYSTEMS (ISA-5.1 TAGS)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    
    subsystems = [
        ("STK-100", "10-Cell High-Pressure PEM Stack", "ASME BPVC VIII"),
        ("SEP-201", "H2 Gas-Liquid Separator (SA-240 316L)", "ASME UG-27"),
        ("O2S-202", "O2/Water Separation Vessel (30 bar)", "ASME UG-27"),
        ("HEX-301", "Process Heat Exchanger", "TEMA Class C"),
        ("DEO-401", "Catalytic Deox Reactor (Pd/Al2O3)", "API 520"),
        ("CMP-501", "Diaphragm H2 Compressor (350 bar)", "API 618"),
        ("BUF-502", "High-Pressure Buffer Rack (350 bar)", "DOT-3AA / ISO"),
        ("WTR-650", "AERO-PURE Water Ultrafiltration Skid", "ASTM D1193 T1"),
        ("STR-701", "ISMB 300 Mezzanine Skid & Piping", "IS 800 / AISC"),
        ("SAF-801", "LOPA Triple Relief & Bleed Header", "IEC 61511 SIL-2"),
        ("DCS-901", "Prognostic Digital Twin Telemetry", "IP66 / NEMA 4X")
    ]
    for tag, desc, code in subsystems:
        p = tf13_r.add_paragraph()
        p.text = f"• [{tag}] {desc} "
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = f"({code})"
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    add_speaker_notes(slide13,
        "Slide 13 displays our fully assembled commercial plant modeled in SolidWorks 2025 SP1.2 and plotted in AutoCAD 2025. "
        "This 100 TPD multi-megawatt facility includes eleven reconciled subsystems: 10-cell high-pressure PEM stacks, "
        "dual high-pressure gas-liquid separators, heat exchangers, deoxygenation reactors, 350-bar hydrogen compressors, "
        "buffer storage, and the AERO-PURE Unit 650 water ultrafiltration skid. Every pipe, valve, and LOPA safety interconnect is physically modeled."
    )

    # =========================================================================
    # SLIDE 14: ASD-STM100 MONETIZATION & COMMERCIAL SERVICES (FIG 8)
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14, NAVY_DARK)
    add_header(slide14, "ASD-STM100 COMMERCIALIZATION & MONETIZATION", "AERO-H2 Commercial Service Portfolio & Industrial Payback Model", dark=True)
    
    # Left: Commercial Service Tiers
    card14_l = add_card(slide14, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.3), bg_color=NAVY_CARD, border_color=EMERALD)
    tf14_l = card14_l.text_frame
    tf14_l.word_wrap = True
    
    p = tf14_l.paragraphs[0]
    p.text = "ASD-STM100 COMMERCIAL SERVICE SUITE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    
    tiers14 = [
        ("Tier 1: Plant Health Diagnostic Audit ($7,500 / ₹6.2 Lakh):", "Non-invasive baseline degradation audit, CFD flow field review, and ASME code compliance check. 5-day remote delivery."),
        ("Tier 2: 'AERO-H2' Edge Digital Twin ($2,500/mo or $25k/MW):", "Quantized PIBT inference deployed onto Siemens S7-1500 / Jetson / STM32 for 24/7 biofilm tracking and automated load mitigation."),
        ("Tier 3: Turnkey DFM SolidWorks Package ($45,000 / ₹37.5 Lakh):", "Complete SolidWorks 2025 3D B-Rep master assembly, AutoCAD ISO A1 shop drawings, BOM, and stamping tool designs for EPC fabricators."),
        ("Audited Payback: 4.8 Months (ROI = 485%):", "Defers $420,000 stack replacement CAPEX on a 5 MW plant, cutting LCOH by $0.42/kg H2.")
    ]
    for b_title, b_desc in tiers14:
        p = tf14_l.add_paragraph()
        p.text = f"\n• {b_title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_MUTED

    # Right: Embedded Fig 8 Image
    fig8_path = os.path.join(assets_dir, "Fig8_Commercial_Monetization_Service_Matrix_ASD_STM100.png")
    if os.path.exists(fig8_path):
        slide14.shapes.add_picture(fig8_path, Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.3))

    add_speaker_notes(slide14,
        "Finally, Slide 14 presents our commercialization roadmap governed by the ASD-STM100 Zero-Loss Charter. "
        "We monetize this technology through three high-margin digital offerings: "
        "first, remote diagnostic audits; second, edge-embedded digital twin firmware licensing on PLCs; "
        "and third, full turnkey DFM CAD packages for EPC contractors. "
        "For an industrial client operating a 5 MW stack, the system pays for itself in just 4.8 months by deferring $420,000 in stack replacement costs. "
        "Thank you for your time, and we look forward to industrial partnerships."
    )

    # ==========================================
    # SLIDE 15: CONCLUSIONS & ROADMAP (DARK THEME)
    # ==========================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15, NAVY_DARK)
    add_header(slide15, "12. Summary & Future Horizons", "Conclusions & The Next-Generation Green Hydrogen Roadmap", dark=True)

    # 4 Conclusion Cards
    c15_w = Inches(2.7)
    gap15 = Inches(0.25)
    
    cards15 = [
        ("Materials Informatics", "XGBoost Virtual Sensor", 
         "• R² = 0.9287, RMSE = 1.91 µV/h\n• Pinned in GPU VRAM\n• Eliminates destructive teardowns\n• Dominant descriptors: Tafel slope, ECSA, porosity (>74%).", 
         CYAN),
        ("Physics-Informed AI", "PIBT Sequence Engine", 
         "• NRMSE = 0.5428\n• Scaled dot-product attention\n• Butler-Volmer constraint\n• Dynamic FP16 variance clamping prevents overflow.", 
         EMERALD),
        ("Decoupled Safety", "Bayesian Fault Detection", 
         "• F1-Score = 0.990\n• Decouples Ostwald chemical drift from acute EPS biofilms\n• Dynamic 99.5% boundary\n• Proactive load shedding.", 
         ROSE),
        ("Industrial Impact", "Stack Economics", 
         "• +39.4% Lifespan Extension (21,428h to 29,879h)\n• Substantial LCOH reduction\n• Zero Pt/Ir toxic leachate\n• Eliminates chemical biocides.", 
         EMERALD)
    ]
    for i, (tag, stitle, desc, color) in enumerate(cards15):
        c_left = Inches(0.8) + i * (c15_w + gap15)
        sc = add_card(slide15, c_left, Inches(1.6), c15_w, Inches(3.6), bg_color=NAVY_CARD, border_color=color)
        tf = sc.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = tag.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = color
        
        p = tf.add_paragraph()
        p.text = stitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT
        
        p = tf.add_paragraph()
        p.text = f"\n{desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED

    # Bottom Future Work & Contact Card
    bot15 = add_card(slide15, Inches(0.8), Inches(5.5), Inches(11.7), Inches(1.4), bg_color=NAVY_CARD, border_color=CARD_BORDER)
    tf15 = bot15.text_frame
    tf15.word_wrap = True
    p = tf15.paragraphs[0]
    p.text = "FUTURE WORK & COLLABORATIVE HORIZONS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    
    p = tf15.add_paragraph()
    p.text = "1. Edge-AI deployment onto bipolar plate microcontrollers (sub-ms closed loop)   |   2. In-situ feedwater metagenomic sequencing cross-referenced with voltage residuals   |   3. Self-healing polymer membranes.\n"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT
    
    p = tf15.add_paragraph()
    p.text = "Thank you! Corresponding Author: Sourish Senapati (sourishs.chem.ug@jadavpuruniversity.in) | Department of Chemical Engineering, Jadavpur University"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    add_speaker_notes(slide15,
        "In conclusion, we have presented an integrated digital twin that connects materials science, electrochemistry, "
        "and applied AI to deliver sub-microvolt virtual sensing, physics-constrained degradation tracking, and a 39.4% stack lifespan extension. "
        "Our next steps involve deploying PIBT onto edge microcontrollers directly mounted on stack bipolar plates, "
        "and exploring real-time metagenomic sensing in circulating feedwater. "
        "Thank you for your attention. We are delighted to take your questions."
    )

    output_path = r"d:\CompeteEnginner\Presentation_ML_Green_Hydrogen\Green_Hydrogen_ML_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    create_presentation()
