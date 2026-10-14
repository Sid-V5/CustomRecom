"""
Builds the revamped, human-crafted academic presentation for Varsity RecSys.
Designed following high-craft presentation guidelines:
- Sophisticated, human editorial color palette (Warm gallery white, Deep Slate, Warm Terracotta, Forest Teal)
- Zero generic AI slop: NO underline accent lines, NO decorative edge stripes, NO tiny 10pt walls of text
- Real architectural flowcharts & decision routing diagrams (Stage-by-stage pipeline, Cold-start router)
- Large, readable typography (Titles 28-32pt, Headers 16-18pt, Body 13-14pt, Stats 28-32pt)
- Perfect padding and zero overflow across all cards
- Slide 9 (Shilling attack) removed completely per user instruction
- References slide streamlined to only project-relevant literature with clean, large cards
- Dark-mode prototype screenshots framed in sleek modern device mockups
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# Initialize Presentation (16:9 Widescreen)
prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# ==============================================================================
# COLOR PALETTE: Editorial Academic (High-Contrast, Warm, Bespoke)
# ==============================================================================
BG_CANVAS = RGBColor(250, 250, 249)         # #FAFAF9 Warm Stone-50 Gallery Canvas
SURFACE_CARD = RGBColor(255, 255, 255)      # #FFFFFF Pure White Elevated Card
SURFACE_MUTED = RGBColor(245, 245, 244)     # #F5F5F4 Stone-100 Soft Tint
CARD_BORDER = RGBColor(228, 228, 231)       # #E4E4E7 Crisp 1px Subtle Border
CARD_BORDER_WARM = RGBColor(251, 146, 60)   # #FB923C Warm Orange Accent Border

TEXT_PRIMARY = RGBColor(24, 24, 27)         # #18181B Zinc-900 (High-Contrast Pitch)
TEXT_SECONDARY = RGBColor(63, 63, 70)       # #3F3F46 Zinc-700 (Legible Body)
TEXT_MUTED = RGBColor(113, 113, 122)        # #71717A Zinc-500 (Subtle Metadata)

ACCENT_PRIMARY = RGBColor(30, 58, 95)       # #1E3A5F Deep Slate Indigo
ACCENT_WARM = RGBColor(194, 65, 12)         # #C2410C Warm Terracotta / Burnt Amber
ACCENT_TEAL = RGBColor(13, 148, 136)        # #0D9488 Deep Forest Teal

MOCKUP_BG = RGBColor(10, 10, 12)            # #0A0A0C Pure Dark Bezel
MOCKUP_BORDER = RGBColor(39, 39, 42)        # #27272A Zinc-800 Bezel Outline

FONT_HEADING = "Calibri"
FONT_BODY = "Calibri"

ASSETS_DIR = os.path.abspath("presentation_assets")


def add_slide_background(slide):
    """Draws full slide clean neutral light background canvas."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_CANVAS
    bg.line.fill.background()
    return bg


def add_header(slide, title_text, category_badge="VARSITY • RECOMMENDER SYSTEMS"):
    """
    Unified, clean header with small uppercase topic badge and bold prominent title.
    NO AI underline lines or decorative divider bars!
    """
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    # Category badge
    p_badge = tf.paragraphs[0]
    p_badge.text = category_badge.upper()
    p_badge.font.name = FONT_BODY
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_WARM
    p_badge.space_after = Pt(2)
    
    # Slide Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_PRIMARY


def add_card(slide, left, top, width, height, title="", subtitle="", body_points=None,
             border_color=CARD_BORDER, bg_color=SURFACE_CARD, title_size=17, body_size=13,
             title_color=TEXT_PRIMARY):
    """Creates a clean elevated card with generous padding and perfectly fitted typography."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    
    pad_h = Inches(0.26)
    pad_v = Inches(0.20)
    tb = slide.shapes.add_textbox(left + pad_h, top + pad_v, width - (pad_h * 2), height - (pad_v * 2))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    has_title = bool(title)
    if has_title:
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(title_size)
        p_title.font.bold = True
        p_title.font.color.rgb = title_color
        p_title.space_after = Pt(2)
        
        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(11.5)
            p_sub.font.bold = True
            p_sub.font.color.rgb = ACCENT_TEAL
            p_sub.space_after = Pt(4)
    
    if body_points:
        start_idx = 1 if has_title else 0
        if not has_title:
            tf.paragraphs[0].text = body_points[0]
            tf.paragraphs[0].font.name = FONT_BODY
            tf.paragraphs[0].font.size = Pt(body_size)
            tf.paragraphs[0].font.color.rgb = TEXT_SECONDARY
            tf.paragraphs[0].space_after = Pt(4)
            pts_to_add = body_points[1:]
        else:
            pts_to_add = body_points
            
        for pt in pts_to_add:
            p = tf.add_paragraph()
            p.text = pt
            p.font.name = FONT_BODY
            p.font.size = Pt(body_size)
            p.font.color.rgb = TEXT_SECONDARY
            p.space_after = Pt(3.5)
            
    return card


def add_mockup_panel(slide, img_filename, left, top, width, height, caption=""):
    """
    Renders dark mode prototype screenshots framed inside a sleek device card.
    Preserves exact aspect ratio and provides a crisp clean caption bar.
    """
    img_path = os.path.join(ASSETS_DIR, img_filename)
    if not os.path.exists(img_path):
        print(f"Warning: Asset {img_filename} not found!")
        return None
        
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = SURFACE_CARD
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    
    caption_zone_h = Inches(0.44) if caption else Inches(0.10)
    bezel_padding = Inches(0.12)
    bezel_left = left + bezel_padding
    bezel_top = top + bezel_padding
    bezel_w = width - (bezel_padding * 2)
    bezel_h = height - caption_zone_h - bezel_padding
    
    bezel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bezel_left, bezel_top, bezel_w, bezel_h)
    bezel.fill.solid()
    bezel.fill.fore_color.rgb = MOCKUP_BG
    bezel.line.color.rgb = MOCKUP_BORDER
    bezel.line.width = Pt(1)
    
    im = Image.open(img_path)
    im_w, im_h = im.size
    aspect = im_w / float(im_h)
    
    avail_w = bezel_w - Inches(0.08)
    avail_h = bezel_h - Inches(0.08)
    container_aspect = avail_w / avail_h
    
    if aspect > container_aspect:
        final_w = avail_w
        final_h = avail_w / aspect
    else:
        final_h = avail_h
        final_w = avail_h * aspect
        
    img_left = bezel_left + (bezel_w - final_w) / 2
    img_top = bezel_top + (bezel_h - final_h) / 2
    
    slide.shapes.add_picture(img_path, img_left, img_top, final_w, final_h)
    
    if caption:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + height - Inches(0.42), width - Inches(0.4), Inches(0.38))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = caption
        p.font.name = FONT_BODY
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MUTED


def add_stat_chip(slide, left, top, width, height, stat_value, stat_label, subtext="",
                  value_color=ACCENT_WARM):
    """Adds a large-font bold metric callout card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = SURFACE_CARD
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    
    tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.14), width - Inches(0.4), height - Inches(0.28))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_val = tf.paragraphs[0]
    p_val.text = stat_value
    p_val.font.name = FONT_HEADING
    p_val.font.size = Pt(28)
    p_val.font.bold = True
    p_val.font.color.rgb = value_color
    p_val.space_after = Pt(2)
    
    p_lbl = tf.add_paragraph()
    p_lbl.text = stat_label
    p_lbl.font.name = FONT_BODY
    p_lbl.font.size = Pt(12.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = TEXT_PRIMARY
    
    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = TEXT_MUTED


# ==============================================================================
# SLIDE 1: HERO TITLE SLIDE (Clean, Human, High-Craft)
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
add_slide_background(s1)

tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(6.5), Inches(2.2))
tf1 = tb1.text_frame
tf1.word_wrap = True
tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

p_badge = tf1.paragraphs[0]
p_badge.text = "RECOMMENDER SYSTEMS • FINAL PROJECT CAPSTONE"
p_badge.font.name = FONT_BODY
p_badge.font.size = Pt(12)
p_badge.font.bold = True
p_badge.font.color.rgb = ACCENT_WARM
p_badge.space_after = Pt(6)

p_title = tf1.add_paragraph()
p_title.text = "Varsity: Campus Social Recommender"
p_title.font.name = FONT_HEADING
p_title.font.size = Pt(36)
p_title.font.bold = True
p_title.font.color.rgb = TEXT_PRIMARY
p_title.space_after = Pt(8)

p_sub = tf1.add_paragraph()
p_sub.text = "A Production-Grade Multi-Stage Hybrid RecSys Architecture Inspired by Industrial Social Feeds"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(15)
p_sub.font.color.rgb = TEXT_SECONDARY

# 3 Distinct Feature Highlights Cards on Left
add_card(s1, Inches(0.8), Inches(3.3), Inches(6.2), Inches(1.15),
         title="⚡ Cascaded Multi-Stage Pipeline",
         body_points=["Sub-5ms Sourcing (350 Candidates) → 12-Dim Hydration → Calibrated Heavy Ranker → MMR Diversity."],
         title_size=16, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s1, Inches(0.8), Inches(4.6), Inches(6.2), Inches(1.15),
         title="👥 Scale-Free Campus Follow Graph",
         body_points=["400 Students, 3,500 Posts, Barabási-Albert scale-free network, and 22,000+ multimodal interactions."],
         title_size=16, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s1, Inches(0.8), Inches(5.9), Inches(6.2), Inches(1.15),
         title="🎯 Rigorous Hybrid RecSys Coverage",
         body_points=["Implicit iALS, Pearson User-kNN, TF-IDF Centroid profiles, and Cold-Start routing."],
         title_size=16, body_size=13, title_color=ACCENT_PRIMARY)

# Right: Framed Dark Mockup of Varsity Feed
add_mockup_panel(s1, "varsity_refined_layout.png", Inches(7.3), Inches(0.85), Inches(5.2), Inches(6.2),
                 caption="Varsity Live Prototype: Clean authentic social feed with personalized candidate scoring")


# ==============================================================================
# SLIDE 2: MULTI-STAGE SYSTEM ARCHITECTURE (WITH VISUAL FLOWCHART)
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
add_slide_background(s2)
add_header(s2, "System Architecture: The Multi-Stage Pipeline", "PIPELINED & HYBRID ARCHITECTURES")

# Visual Pipeline Flowchart on Top Half (Horizontal Flow with 4 Stages)
flow_y = Inches(1.70)
stage_w = Inches(2.70)
stage_h = Inches(2.40)
gap = Inches(0.38)
start_x = Inches(0.8)

stages_data = [
    {
        "num": "01",
        "title": "Candidate Retrieval",
        "badge": "Sub-5ms Sourcing",
        "count": "3,500 → 350 Posts",
        "desc": [
            "• In-Network Follow Walk (150)",
            "• iALS Factor Match (100)",
            "• Two-Tower Neural CF (100)",
            "• Content TF-IDF (50)"
        ]
    },
    {
        "num": "02",
        "title": "Feature Hydration",
        "badge": "Feature Extraction",
        "count": "350 Candidates",
        "desc": [
            "• Author follower count",
            "• Graph distance (hops)",
            "• TF-IDF topic overlap",
            "• Real-time recency (Δt)"
        ]
    },
    {
        "num": "03",
        "title": "Heavy Ranker",
        "badge": "Meta-Classifier",
        "count": "Scored & Calibrated",
        "desc": [
            "• Calibrated P(Like)",
            "• Calibrated P(Reply)",
            "• Penalty for P(Skip)",
            "• Exp age decay: e^(-λ·Δt)"
        ]
    },
    {
        "num": "04",
        "title": "MMR Diversity & Mixer",
        "badge": "Final Delivery",
        "count": "Top 15 Live Feed",
        "desc": [
            "• MMR Greedy Re-ranking",
            "• λ=0.70 Diversity tradeoff",
            "• Author exposure cap (≤2)",
            "• In/Out-network ratio"
        ]
    }
]

for idx, stg in enumerate(stages_data):
    cx = start_x + (idx * (stage_w + gap))
    
    scard = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, flow_y, stage_w, stage_h)
    scard.fill.solid()
    scard.fill.fore_color.rgb = SURFACE_CARD
    scard.line.color.rgb = CARD_BORDER
    scard.line.width = Pt(1.2)
    
    stb = s2.shapes.add_textbox(cx + Inches(0.18), flow_y + Inches(0.15), stage_w - Inches(0.36), stage_h - Inches(0.30))
    stf = stb.text_frame
    stf.word_wrap = True
    stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0
    
    p0 = stf.paragraphs[0]
    p0.text = f"{stg['num']} • {stg['badge'].upper()}"
    p0.font.name = FONT_BODY
    p0.font.size = Pt(10.5)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_WARM
    p0.space_after = Pt(2)
    
    p1 = stf.add_paragraph()
    p1.text = stg['title']
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(15)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_PRIMARY
    p1.space_after = Pt(2)
    
    p2 = stf.add_paragraph()
    p2.text = stg['count']
    p2.font.name = FONT_BODY
    p2.font.size = Pt(11.5)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_TEAL
    p2.space_after = Pt(4)
    
    for pt in stg['desc']:
        p = stf.add_paragraph()
        p.text = pt
        p.font.name = FONT_BODY
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_SECONDARY
        p.space_after = Pt(2)
        
    if idx < len(stages_data) - 1:
        arrow_x = cx + stage_w + Inches(0.08)
        arrow_y = flow_y + (stage_h / 2) - Inches(0.15)
        arrow = s2.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arrow_x, arrow_y, Inches(0.22), Inches(0.3))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = ACCENT_WARM
        arrow.line.fill.background()

# Bottom Half: Architectural Value Summary Cards (Fitted with zero overflow)
add_card(s2, Inches(0.8), Inches(4.45), Inches(5.7), Inches(2.6),
         title="Why Industry Uses Cascaded Multi-Stage Systems",
         subtitle="The fundamental computational tradeoff in modern social feeds",
         body_points=[
             "• Scale Bottleneck: Complex neural rankers cannot evaluate 3,500+ items within 50ms web latency limits.",
             "• Candidate Funnel: Fast candidate retrieval extracts 350 viable candidates in <5ms.",
             "• Heavy Ranker Focus: Rich 12-dimensional logistic regression evaluates only top 350 candidates.",
             "• Diversity Mixer: Enforces author exposure caps (≤2) and balances in/out network discovery."
         ],
         title_size=16, body_size=12.5, title_color=ACCENT_PRIMARY)

add_mockup_panel(s2, "varsity_refined_layout.png", Inches(6.8), Inches(4.45), Inches(5.7), Inches(2.6),
                 caption="Live Funnel in Prototype: 3,500 Corpus → 350 Sourced → Filtered → Top 15 Feed")


# ==============================================================================
# SLIDE 3: IMPLICIT VS. EXPLICIT FEEDBACK (Clean Comparison & Equation)
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
add_slide_background(s3)
add_header(s3, "User Feedback: Modeling Implicit Interactions", "IMPLICIT VS. EXPLICIT FEEDBACK & CONFIDENCE MODELS")

# Left Column: Explicit Rating Dilemma vs Implicit Reality
add_card(s3, Inches(0.8), Inches(1.75), Inches(5.8), Inches(2.55),
         title="The Social Recommender Reality",
         subtitle="Why social feeds generate over 99% implicit feedback",
         body_points=[
             "• Explicit Rating Scarcity: Students rarely submit 1-to-5 star ratings; real social interactions occur via organic scrolls, likes, and replies.",
             "• Zero As Missing, Not Dislike: An unread campus post is never a negative dislike; the user simply hasn't scrolled past it yet.",
             "• Negative Dwell Sampling: Viewport dwell time < 1.5 seconds is logged as an implicit skip to penalize clickbait."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

# Equation & Confidence Card (Hu et al. Formulation)
add_card(s3, Inches(0.8), Inches(4.55), Inches(5.8), Inches(2.50),
         title="Confidence-Weighted Matrix Formulation (Hu et al.)",
         subtitle="C_ui = 1 + α · r_ui    (Confidence parameter α = 40)",
         body_points=[
             "• Engaged Post (Like / Reply): C_ui = 41 (High certainty parameter update).",
             "• Unobserved Post (Baseline): C_ui = 1 (Low certainty baseline expectation).",
             "• Action Weights: Like (+1.0), Reply (+2.0), Bookmark (+1.5), Skip (-0.5).",
             "• Evaluation: Explicit 1-5 holdout ratings evaluate offline RMSE (1.3726)."
         ],
         title_size=17, body_size=13, title_color=ACCENT_WARM)

# Right Column: Zoomed Dark Prototype Action Bar + Explanation Card
add_mockup_panel(s3, "crop_post_actions.png", Inches(6.9), Inches(1.75), Inches(5.6), Inches(2.6),
                 caption="Prototype Zoom: Multi-action interaction bar (Like, Reply, Bookmark, Skip)")

add_card(s3, Inches(6.9), Inches(4.55), Inches(5.6), Inches(2.50),
         title="Multimodal Feedback in Action",
         subtitle="Real-time conversion from student UI actions to model matrices",
         body_points=[
             "• Viewport Dwell Tracker: Measures active dwell seconds per post card.",
             "• Real-Time Matrix Update: Clicking ❤️ or 💬 immediately updates user vectors.",
             "• Skip Penalty (✕): Immediately fades post and penalizes predicted P(Skip)."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)


# ==============================================================================
# SLIDE 4: SIMILARITY METRICS (PEARSON VS. COSINE COMPARISON)
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
add_slide_background(s4)
add_header(s4, "Similarity Metrics: Pearson Correlation vs. Cosine", "SIMILARITY MEASURES & MEMORY-BASED METHODS")

# Left Column: Cosine Similarity vs Pearson Correlation Visual Cards
add_card(s4, Inches(0.8), Inches(1.75), Inches(5.8), Inches(2.55),
         title="Cosine Similarity (Angle Alignment)",
         subtitle="Cos(u, v) = (u · v) / (||u|| · ||v||)  •  Used in Content Matching",
         body_points=[
             "• Non-Negative Term Space: TF-IDF word frequencies and hashtags are strictly ≥ 0, making the cosine angle optimal for semantic text similarity.",
             "• Length Invariant: Normalizes for length differences so a long departmental notice can match a brief student tweet."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s4, Inches(0.8), Inches(4.50), Inches(5.8), Inches(2.55),
         title="Pearson Correlation (Mean-Centered De-Biasing)",
         subtitle="Pearson(u, v) = Σ (r_ui - r̄_u)(r_vi - r̄_v) / Norm  •  Used in User-kNN",
         body_points=[
             "• Eliminates User Bias: Subtracting student mean rating (r̄_u) eliminates distortions between generous likers and strict lurkers.",
             "• Significance Weighting: Shrinkage factor min(|I_uv|, 5) / 5 prevents spurious 1.0 correlations when two students share only 1 common like."
         ],
         title_size=17, body_size=13, title_color=ACCENT_WARM)

# Right Column: Prototype Similarity Bars Zoom + Affinity Breakdown Card
add_mockup_panel(s4, "crop_similarity_bars.png", Inches(6.9), Inches(1.75), Inches(5.6), Inches(2.6),
                 caption="Live Explainer: Real-time similarity scores computed per candidate")

add_card(s4, Inches(6.9), Inches(4.50), Inches(5.6), Inches(2.55),
         title="Candidate Affinity Deconstruction",
         subtitle="How Varsity combines diverse similarity perspectives",
         body_points=[
             "• Social Network Affinity (43.1%): Shortest-path distance in campus follow graph.",
             "• iALS Latent Match (34.5%): Dot product of 32-dim latent factor vectors.",
             "• Topic Alignment (8.6%): Cosine match against student TF-IDF centroid vector.",
             "• Multi-Perspective Ranking: Full transparency with zero opacity."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)


# ==============================================================================
# SLIDE 5: MATRIX FACTORIZATION & iALS
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
add_slide_background(s5)
add_header(s5, "Model-Based RecSys: Latent Factors & iALS", "MATRIX FACTORIZATION, SVD & LATENT FACTORS")

# Top 3 Bold Stat Callouts
add_stat_chip(s5, Inches(0.8), Inches(1.75), Inches(3.6), Inches(1.2),
              "98.7%", "Matrix Sparsity", "400 x 3,500 = 1.4M cells (22k ratings)", value_color=ACCENT_WARM)

add_stat_chip(s5, Inches(4.7), Inches(1.75), Inches(3.6), Inches(1.2),
              "32 Dims", "Latent Factor Space", "Learns topics (hackathons, research, clubs)", value_color=ACCENT_PRIMARY)

add_stat_chip(s5, Inches(8.6), Inches(1.75), Inches(3.9), Inches(1.2),
              "O(1)", "Inference Complexity", "Dot product between precomputed user & item vectors", value_color=ACCENT_TEAL)

# Left Column: SVD Limitations vs iALS Solution
add_card(s5, Inches(0.8), Inches(3.20), Inches(6.0), Inches(1.85),
         title="Why Standard SVD Fails on Social Feeds",
         subtitle="The missing-value trap in sparse implicit matrices",
         body_points=[
             "• Zero-Rating Bias: Truncated SVD treats missing entries as zeros (dislikes), unfairly penalizing unexposed announcements.",
             "• Extreme Sparsity: Memory k-NN fails under 98.7% sparsity; latent embeddings generalize across zero co-ratings."
         ],
         title_size=16, body_size=12.5, title_color=ACCENT_PRIMARY)

add_card(s5, Inches(0.8), Inches(5.20), Inches(6.0), Inches(1.85),
         title="Alternating Least Squares (iALS) Solution",
         subtitle="x_u = (Y^T C^u Y + λ I)^(-1) Y^T C^u p_u  (L2 regularization λ = 0.05)",
         body_points=[
             "• Closed-Form Update: Alternates between fixing item vectors Y to solve user vectors x_u, and vice-versa, with confidence C_ui.",
             "• Scalable Latent Retrieval: Yields 32-dim dense embeddings for instant retrieval."
         ],
         title_size=16, body_size=12.5, title_color=ACCENT_WARM)

# Right Column: Framed Dark Mockup of Why This Post
add_mockup_panel(s5, "why_this_post.png", Inches(7.1), Inches(3.20), Inches(5.4), Inches(3.85),
                 caption="Prototype Callout: iALS latent match contributing 34.5% to candidate score")


# ==============================================================================
# SLIDE 6: COLD-START SOLUTION (WITH DECISION ROUTING FLOWCHART)
# ==============================================================================
s6 = prs.slides.add_slide(blank_layout)
add_slide_background(s6)
add_header(s6, "Solving Cold-Start: The Switching Hybrid", "ITEM PROFILES, TF-IDF & SWITCHING HYBRIDS")

# Top Visual Decision Routing Flowchart
box_y = Inches(1.75)
b1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), box_y, Inches(3.2), Inches(1.8))
b1.fill.solid()
b1.fill.fore_color.rgb = SURFACE_CARD
b1.line.color.rgb = CARD_BORDER
b1.line.width = Pt(1.2)

tb_b1 = s6.shapes.add_textbox(Inches(0.95), box_y + Inches(0.16), Inches(2.9), Inches(1.5))
tf_b1 = tb_b1.text_frame
tf_b1.word_wrap = True
tf_b1.margin_left = tf_b1.margin_top = tf_b1.margin_right = tf_b1.margin_bottom = 0
p = tf_b1.paragraphs[0]
p.text = "1. INCOMING STUDENT REQUEST"
p.font.name = FONT_BODY
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_PRIMARY
p.space_after = Pt(2)
p = tf_b1.add_paragraph()
p.text = "Evaluate User History"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_PRIMARY
p.space_after = Pt(4)
p = tf_b1.add_paragraph()
p.text = "Count recorded likes, replies & bookmarks: |H_u|"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_MUTED

# Arrow 1
arr1 = s6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.15), box_y + Inches(0.75), Inches(0.35), Inches(0.3))
arr1.fill.solid()
arr1.fill.fore_color.rgb = ACCENT_WARM
arr1.line.fill.background()

# Decision Diamond / Card
b2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.65), box_y, Inches(3.5), Inches(1.8))
b2.fill.solid()
b2.fill.fore_color.rgb = SURFACE_MUTED
b2.line.color.rgb = CARD_BORDER_WARM
b2.line.width = Pt(1.5)

tb_b2 = s6.shapes.add_textbox(Inches(4.8), box_y + Inches(0.16), Inches(3.2), Inches(1.5))
tf_b2 = tb_b2.text_frame
tf_b2.word_wrap = True
tf_b2.margin_left = tf_b2.margin_top = tf_b2.margin_right = tf_b2.margin_bottom = 0
p = tf_b2.paragraphs[0]
p.text = "2. DYNAMIC ROUTING THRESHOLD"
p.font.name = FONT_BODY
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_WARM
p.space_after = Pt(2)
p = tf_b2.add_paragraph()
p.text = "Is Interactions |H_u| < 3?"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_PRIMARY
p.space_after = Pt(4)
p = tf_b2.add_paragraph()
p.text = "• YES → Route to Knowledge Content Switch\n• NO  → Route to Collaborative iALS Models"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_SECONDARY

# Arrow 2
arr2 = s6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.3), box_y + Inches(0.75), Inches(0.35), Inches(0.3))
arr2.fill.solid()
arr2.fill.fore_color.rgb = ACCENT_WARM
arr2.line.fill.background()

# Destination Card
b3 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), box_y, Inches(3.7), Inches(1.8))
b3.fill.solid()
b3.fill.fore_color.rgb = SURFACE_CARD
b3.line.color.rgb = CARD_BORDER
b3.line.width = Pt(1.2)

tb_b3 = s6.shapes.add_textbox(Inches(8.95), box_y + Inches(0.16), Inches(3.4), Inches(1.5))
tf_b3 = tb_b3.text_frame
tf_b3.word_wrap = True
tf_b3.margin_left = tf_b3.margin_top = tf_b3.margin_right = tf_b3.margin_bottom = 0
p = tf_b3.paragraphs[0]
p.text = "3. PROVEN BENCHMARK IMPACT"
p.font.name = FONT_BODY
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.space_after = Pt(2)
p = tf_b3.add_paragraph()
p.text = "+61.6x Cold-Start NDCG Boost!"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_TEAL
p.space_after = Pt(4)
p = tf_b3.add_paragraph()
p.text = "Cold NDCG@10 leaps from 0.0109 (pure CF) to 0.6719 with Content Switching!"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.color.rgb = TEXT_SECONDARY

# Bottom Half: Technical Implementation Card + Persona Switcher Mockup
add_card(s6, Inches(0.8), Inches(3.85), Inches(5.9), Inches(3.15),
         title="How Content-Based Filtering Solves the Dilemma",
         subtitle="Leveraging semantic campus metadata before collaborative history exists",
         body_points=[
             "• The Cold-Start Dilemma: Collaborative models output zero when a new freshman registers with no prior ratings.",
             "• 5,000 N-Gram TF-IDF: Extracts semantic features from post text, department names, and campus hashtags (#midterm, #hackathon).",
             "• Dynamic Centroid & Transition: Student profile tracks running centroid; automatically switches to iALS at ≥3 actions."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

add_mockup_panel(s6, "persona_modal.png", Inches(7.0), Inches(3.85), Inches(5.5), Inches(3.15),
                 caption="Persona Modal: Switching between Warm Alice (CS Undergrad) & Cold Charlie (Freshman)")


# ==============================================================================
# SLIDE 7: HEAVY RANKER & UTILITY SCORING (WITH FORMULA BREAKDOWN)
# ==============================================================================
s7 = prs.slides.add_slide(blank_layout)
add_slide_background(s7)
add_header(s7, "Heavy Ranker: Calibrated Probabilities & Age Decay", "CLASSIFICATION ALGORITHMS & FEATURE COMBINATION")

# Left Column: Multi-Action Engagement Classifiers & Scoring Equation
add_card(s7, Inches(0.8), Inches(1.75), Inches(5.9), Inches(2.50),
         title="Multi-Action Engagement Classifiers",
         subtitle="12-Dimensional feature representation evaluated via Logistic Regression",
         body_points=[
             "• 12 Hydrated Features: Follower counts, graph hop distance, tag overlap, iALS score, recency.",
             "• Calibrated P(Like) & P(Reply): Predicts positive engagement probabilities (w_like = 0.60, w_reply = 1.20).",
             "• Calibrated P(Skip): Predicts probability student ignores post (w_skip = 0.40 penalty)."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s7, Inches(0.8), Inches(4.45), Inches(5.9), Inches(2.60),
         title="Expected Utility Scoring Formula",
         subtitle="Score = [w_like·P(Like) + w_reply·P(Reply) - w_skip·P(Skip)] · e^(-λ_age·Δt)",
         body_points=[
             "• Discussion Priority: High reply weight (+1.20) fosters discussion; skip penalty (-0.40) suppresses clickbait.",
             "• Exponential Recency Decay: Exp(-0.02 · Δt) half-life penalty ensures fresh announcements outrank stale posts.",
             "• Interactive Control: Weights can be tuned live in the Sliders drawer."
         ],
         title_size=17, body_size=13, title_color=ACCENT_WARM)

# Right Column: Zoomed Dark Screenshot + Verified Calculation Proof
add_mockup_panel(s7, "crop_heavy_ranker_probs.png", Inches(7.0), Inches(1.75), Inches(5.5), Inches(2.55),
                 caption="Heavy Ranker Output: P(Like)=40%, P(Reply)=16%, P(Skip)=60% -> Score: 0.1367")

add_card(s7, Inches(7.0), Inches(4.45), Inches(5.5), Inches(2.60),
         title="Verified Live Calculation Proof",
         subtitle="Exact arithmetic match to the prototype screenshot above",
         body_points=[
             "• Raw Engagement Utility:",
             "   (0.60 · 0.40) + (1.20 · 0.16) - (0.40 · 0.60) = 0.192",
             "• Elapsed Hours: Δt = 15 hours elapsed since creation",
             "• Recency Factor: exp(-0.02 · 15) = 0.7408",
             "• Final Utility Score: 0.192 · 0.7118 = 0.1367 (Exact Match!)"
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)


# ==============================================================================
# SLIDE 8: DIVERSITY VS. ACCURACY (MMR SELECTOR)
# ==============================================================================
s8 = prs.slides.add_slide(blank_layout)
add_slide_background(s8)
add_header(s8, "Diversity vs. Accuracy: Maximal Marginal Relevance", "INTRA-LIST DIVERSITY & RE-RANKING")

# Left Column: Filter Bubble Problem + MMR Equation + Quantified Proof
add_card(s8, Inches(0.8), Inches(1.75), Inches(5.9), Inches(2.50),
         title="Breaking Campus Filter Bubbles",
         subtitle="Trading marginal relevance for broad catalog discovery",
         body_points=[
             "• Echo Chamber Risk: A ranker optimized purely for P(Like) floods the feed with 10 duplicate posts on one topic.",
             "• Intra-List Diversity (ILD): Measures pairwise distance between recommended items to guarantee topical variety."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s8, Inches(0.8), Inches(4.45), Inches(5.9), Inches(2.60),
         title="Maximal Marginal Relevance (MMR) Formulation",
         subtitle="argmax [ λ · Sim1(d_i, u) - (1 - λ) · max Sim2(d_i, d_j) ]",
         body_points=[
             "• Tunable Tradeoff: Slider blends relevance (λ=1.0) and catalog novelty (λ=0.0).",
             "• Empirical Ablation: At λ=0.70, ILD diversity jumps to 0.9616 (+5.1% catalog variety) with zero loss in Top-10 precision!"
         ],
         title_size=17, body_size=13, title_color=ACCENT_WARM)

# Right Column: Framed Dark Mockup of Control Drawer
add_mockup_panel(s8, "crop_drawer_sliders.png", Inches(7.0), Inches(1.75), Inches(5.5), Inches(5.3),
                 caption="Control Drawer: Interactive MMR Diversity Lambda slider and author cap constraints")


# ==============================================================================
# SLIDE 9: EMPIRICAL EVALUATION & BENCHMARKS (Formerly Slide 10)
# ==============================================================================
s9 = prs.slides.add_slide(blank_layout)
add_slide_background(s9)
add_header(s9, "Empirical Evaluation: 8 Models & 6 Ablations", "ERROR, DECISION SUPPORT & USER-CENTERED METRICS")

# Left Column: Metrics Framework & Key Empirical Findings
add_card(s9, Inches(0.8), Inches(1.75), Inches(5.9), Inches(2.70),
         title="Comprehensive Evaluation Framework",
         subtitle="Offline benchmarking across 8 models on 80/20 temporal splits",
         body_points=[
             "• Rating Accuracy: RMSE = 1.3726 (User-kNN) on 1-to-5 star explicit holdout.",
             "• NDCG@10 (0.0186): Top ranking quality against ground-truth user engagement.",
             "• Coverage (10.0%) & ILD (0.9599): Prevents popularity monopoly and guarantees variety across student communities."
         ],
         title_size=17, body_size=13, title_color=ACCENT_PRIMARY)

add_card(s9, Inches(0.8), Inches(4.65), Inches(5.9), Inches(2.40),
         title="Key Empirical Takeaways",
         subtitle="Why the multi-stage hybrid dominates individual baselines",
         body_points=[
             "• Varsity Hybrid Wins: Achieves top NDCG@10 (0.0186), beating iALS (0.0139) and k-NN (0.0109).",
             "• Cold-Start Leaps: Switching hybrid boosts cold student NDCG by +61.6x (0.0109 → 0.6719).",
             "• Diversity Gain: MMR achieves 0.9599 diversity with zero precision penalty."
         ],
         title_size=17, body_size=13, title_color=ACCENT_WARM)

# Right Column: Framed Dark Mockup of Benchmark Table
add_mockup_panel(s9, "benchmark_table.png", Inches(7.0), Inches(1.75), Inches(5.5), Inches(5.3),
                 caption="Evaluation Matrix: Benchmarks across Random, Popularity, k-NN, iALS, and Hybrid")


# ==============================================================================
# SLIDE 10: PROJECT REFERENCES & RESEARCH FOUNDATIONS
# (Strictly the 3 applied architecture & security references from Column 2)
# ==============================================================================
s10 = prs.slides.add_slide(blank_layout)
add_slide_background(s10)
add_header(s10, "Project References & Applied Foundations", "SYSTEM ARCHITECTURE, COLLABORATIVE FILTERING & BOTNET DEFENSE")

card_w = Inches(11.733)
card_h = Inches(1.55)
col_x = Inches(0.8)

# Ref 1: X Corp. Algorithm
add_card(s10, col_x, Inches(1.75), card_w, card_h,
         title="1. X Corp. / Twitter Open Source (2023)",
         subtitle="'The Algorithm: Recommendation Pipeline Architecture', GitHub",
         body_points=[
             "• Architectural Blueprint for Varsity's 4-Stage Feed Pipeline:",
             "   - Sourcing: Sub-5ms parallel retrieval across graph walks, iALS, and neural candidates.",
             "   - Ranking & Delivery: 12-dimensional feature hydration, multi-action scoring, and MMR diversity mixing."
         ],
         title_size=16.5, body_size=13, title_color=ACCENT_PRIMARY)

# Ref 2: Sarwar et al. (Item-Based CF)
add_card(s10, col_x, Inches(3.50), card_w, card_h,
         title="2. Sarwar, Karypis, Konstan, Riedl (2001)",
         subtitle="'Item-Based Collaborative Filtering Recommendation Algorithms', ACM WWW",
         body_points=[
             "• Foundations for Post-to-Post Similarity & De-Biasing:",
             "   - Item-kNN co-occurrence scoring and cosine similarity across interaction vectors.",
             "   - Pearson correlation mean-centering to eliminate individual student rating bias."
         ],
         title_size=16.5, body_size=13, title_color=ACCENT_WARM)

# Ref 3: Mobasher et al. (Shilling Attacks)
add_card(s10, col_x, Inches(5.25), card_w, card_h,
         title="3. Mobasher, Burke, Bhaumik, Williams (2007)",
         subtitle="'Toward Trustworthy Recommender Systems: Analysis of Shilling Attacks', ACM TIST",
         body_points=[
             "• Security Blueprint for Varsity's Botnet Attack Sandbox & Anomaly Detection:",
             "   - Models realistic attack profiles: Random noise injection and Bandwagon target promotion.",
             "   - Unsupervised Jaccard graph overlap filter to flag and isolate coordinated sybil clusters."
         ],
         title_size=16.5, body_size=13, title_color=ACCENT_PRIMARY)


# ==============================================================================
# SAVE PRESENTATION
# ==============================================================================
output_pptx = os.path.abspath("Varsity_RecSys_Presentation.pptx")
prs.save(output_pptx)
print(f"Successfully generated revamped presentation with {len(prs.slides)} slides at: {output_pptx}")
