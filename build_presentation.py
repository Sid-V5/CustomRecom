"""
Builds the beautified, high-craft academic presentation for Varsity RecSys.
Designed with:
- Crisp light mode canvas (#F8FAFC) with pure white elevated cards (#FFFFFF)
- Zero blue or yellow in background canvas
- Dark mode prototype screenshots framed in sleek device mockups
- Purpose-built, varied slide layouts (stat chips, formula boxes, comparison columns)
- Conceptual topic badges without lecture numbers
- Grounded, project-specific academic references on the final slide
"""

import os
from PIL import Image
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Initialize Presentation
prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# ==========================================
# DESIGN THEME PALETTE: Varsity High-Craft Light
# Neutral slate canvas with crisp contrast
# ==========================================
BG_COLOR = RGBColor(248, 250, 252)          # #F8FAFC Neutral Slate-50 Canvas
SURFACE_COLOR = RGBColor(255, 255, 255)     # #FFFFFF Pure White Card
SURFACE_MUTED = RGBColor(241, 245, 249)     # #F1F5F9 Slate-100 Callout Card
SURFACE_BORDER = RGBColor(226, 232, 240)    # #E2E8F0 Slate-200 Subtle 1px Border
MOCKUP_BG = RGBColor(11, 15, 23)            # #0B0F17 Sleek Dark Mockup Canvas
MOCKUP_BORDER = RGBColor(30, 41, 59)        # #1E293B Dark Mockup Border

ACCENT_BLUE = RGBColor(2, 132, 199)         # #0284C7 Sky-600 / High Contrast
ACCENT_GREEN = RGBColor(5, 150, 105)        # #059669 Emerald-600 / Success
ACCENT_AMBER = RGBColor(217, 119, 6)        # #D97706 Amber-600 / Warning
ACCENT_RED = RGBColor(225, 29, 72)          # #E11D48 Rose-600 / Threat

TEXT_PRIMARY = RGBColor(15, 23, 42)         # #0F172A Slate-900 (Bold Black/Slate)
TEXT_SECONDARY = RGBColor(51, 65, 85)       # #334155 Slate-700 (High-Contrast Body)
TEXT_MUTED = RGBColor(100, 116, 139)        # #64748B Slate-500 (Subtle Metadata)

FONT_TITLE = "Segoe UI"
FONT_BODY = "Segoe UI"

ASSETS_DIR = os.path.abspath("presentation_assets")

def add_slide_background(slide):
    """Draws full slide clean neutral light background canvas."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_COLOR
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_badge="AIM3143 • RECOMMENDER SYSTEMS"):
    """Adds a unified header with category badge and bold title. No underline lines!"""
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(11.7), Inches(1.15))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    # Category badge
    p_badge = tf.paragraphs[0]
    p_badge.text = category_badge.upper()
    p_badge.font.name = FONT_BODY
    p_badge.font.size = Pt(10.5)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_BLUE
    p_badge.space_after = Pt(3)
    
    # Slide Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_TITLE
    p_title.font.size = Pt(25)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_PRIMARY

def add_card(slide, left, top, width, height, title="", subtitle="", body_points=None, border_color=SURFACE_BORDER, bg_color=SURFACE_COLOR, title_size=14, body_size=11):
    """Creates an elevated surface card with clean spacing."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    
    # Text container inside card
    tb = slide.shapes.add_textbox(left + Inches(0.24), top + Inches(0.18), width - Inches(0.48), height - Inches(0.36))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    if title:
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = FONT_TITLE
        p_title.font.size = Pt(title_size)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_PRIMARY
        p_title.space_after = Pt(2)
        
        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(10.5)
            p_sub.font.color.rgb = ACCENT_BLUE
            p_sub.space_after = Pt(5)
    
    if body_points:
        start_idx = 0 if not title else 1
        for pt in body_points:
            p = tf.add_paragraph() if start_idx > 0 else tf.paragraphs[0]
            start_idx += 1
            p.text = pt
            p.font.name = FONT_BODY
            p.font.size = Pt(body_size)
            p.font.color.rgb = TEXT_SECONDARY
            p.space_after = Pt(3.5)
            
    return card

def add_mockup_panel(slide, img_filename, left, top, width, height, caption=""):
    """
    Renders a dark mode screenshot framed inside an elegant device/window mockup.
    Includes container card, dark inner bezel, centered image, and caption.
    """
    img_path = os.path.join(ASSETS_DIR, img_filename)
    if not os.path.exists(img_path):
        print(f"Warning: Asset {img_filename} not found!")
        return None
        
    # Outer Card (White surface with subtle border)
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = SURFACE_COLOR
    card.line.color.rgb = SURFACE_BORDER
    card.line.width = Pt(1)
    
    # Inner Dark Mockup Frame
    caption_zone_h = Inches(0.52) if caption else Inches(0.15)
    bezel_padding = Inches(0.14)
    bezel_left = left + bezel_padding
    bezel_top = top + bezel_padding
    bezel_w = width - (bezel_padding * 2)
    bezel_h = height - caption_zone_h - bezel_padding
    
    bezel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bezel_left, bezel_top, bezel_w, bezel_h)
    bezel.fill.solid()
    bezel.fill.fore_color.rgb = MOCKUP_BG
    bezel.line.color.rgb = MOCKUP_BORDER
    bezel.line.width = Pt(1)
    
    # Calculate aspect ratio
    im = Image.open(img_path)
    im_w, im_h = im.size
    aspect = im_w / float(im_h)
    
    avail_w = bezel_w - Inches(0.12)
    avail_h = bezel_h - Inches(0.12)
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
    
    # Dedicated Caption Bar
    if caption:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + height - Inches(0.48), width - Inches(0.4), Inches(0.42))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = caption
        p.font.name = FONT_BODY
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED

def add_stat_chip(slide, left, top, width, height, stat_value, stat_label, subtext=""):
    """Adds a bold metric badge card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = SURFACE_MUTED
    card.line.color.rgb = SURFACE_BORDER
    card.line.width = Pt(1)
    
    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_val = tf.paragraphs[0]
    p_val.text = stat_value
    p_val.font.name = FONT_TITLE
    p_val.font.size = Pt(20)
    p_val.font.bold = True
    p_val.font.color.rgb = ACCENT_BLUE
    
    p_lbl = tf.add_paragraph()
    p_lbl.text = stat_label
    p_lbl.font.name = FONT_BODY
    p_lbl.font.size = Pt(10.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = TEXT_PRIMARY
    
    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = TEXT_MUTED


# ==============================================================================
# SLIDE 1: HERO TITLE SLIDE
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
add_slide_background(s1)

# Left Column: Branding, Title & 3 Structured Value Cards
tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(6.5), Inches(2.2))
tf1 = tb1.text_frame
tf1.word_wrap = True
tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

p_badge = tf1.paragraphs[0]
p_badge.text = "AIM3143 • RECOMMENDER SYSTEMS FINAL CAPSTONE"
p_badge.font.name = FONT_BODY
p_badge.font.size = Pt(11)
p_badge.font.bold = True
p_badge.font.color.rgb = ACCENT_BLUE
p_badge.space_after = Pt(6)

p_title = tf1.add_paragraph()
p_title.text = "Varsity: Campus Social Recommender"
p_title.font.name = FONT_TITLE
p_title.font.size = Pt(32)
p_title.font.bold = True
p_title.font.color.rgb = TEXT_PRIMARY
p_title.space_after = Pt(6)

p_sub = tf1.add_paragraph()
p_sub.text = "A Production-Grade Multi-Stage Hybrid RecSys Architecture Inspired by X.com"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(14)
p_sub.font.color.rgb = TEXT_SECONDARY

# 3 Distinct Feature Highlights Cards on Left
add_card(s1, Inches(0.8), Inches(3.2), Inches(6.2), Inches(1.15),
         title="⚡ Cascaded Multi-Stage Pipeline",
         subtitle="",
         body_points=["• Sourcing (<5ms) → 12-Dim Hydration → Calibrated Heavy Ranker → MMR Diversity."],
         body_size=10.5, title_size=13)

add_card(s1, Inches(0.8), Inches(4.5), Inches(6.2), Inches(1.15),
         title="👥 Scale-Free Campus Graph & Personas",
         subtitle="",
         body_points=["• 400 Students, 3,500 Posts, Barabási-Albert follow network, 22,000+ multimodal interactions."],
         body_size=10.5, title_size=13)

add_card(s1, Inches(0.8), Inches(5.8), Inches(6.2), Inches(1.15),
         title="🎯 Complete Curriculum & Defense Coverage",
         subtitle="",
         body_points=["• Implicit iALS, Pearson k-NN, TF-IDF Centroids, Cold-Start Switch & Jaccard Botnet Defense."],
         body_size=10.5, title_size=13)

# Right: Framed Dark Mockup of Varsity Feed
add_mockup_panel(s1, "varsity_refined_layout.png", Inches(7.3), Inches(0.85), Inches(5.2), Inches(6.1),
                 caption="Varsity Live Prototype: 3-column feed with candidate hydration and telemetry")


# ==============================================================================
# SLIDE 2: MULTI-STAGE SYSTEM ARCHITECTURE
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
add_slide_background(s2)
add_header(s2, "System Architecture: The Multi-Stage Hybrid Pipeline", "PIPELINED & HYBRID ARCHITECTURES")

# Left Column: 4 Distinct Stage Cards (Visual Flow)
col_left = Inches(0.8)
col_w = Inches(6.1)

add_card(s2, col_left, Inches(1.8), col_w, Inches(1.15),
         title="1. Candidate Retrieval (Sub-5ms Sourcing)",
         subtitle="Scans 3,500 posts to extract ~350 candidates",
         body_points=["• Combines In-Network graph walk (150) + iALS factors (100) + Neural CF (100) + Content TF-IDF (50)."],
         title_size=13, body_size=10.5)

add_card(s2, col_left, Inches(3.1), col_w, Inches(1.15),
         title="2. Feature Hydration & Preprocessing",
         subtitle="Enriches candidates into a 12-dimensional vector",
         body_points=["• Features: Author popularity, follow graph distance, TF-IDF tag alignment, and recency."],
         title_size=13, body_size=10.5)

add_card(s2, col_left, Inches(4.4), col_w, Inches(1.15),
         title="3. Heavy Ranker (Meta-Level Classifier)",
         subtitle="Calibrated Logistic Regression meta-model",
         body_points=["• Computes probabilities P(Like), P(Reply), P(Skip) and applies exponential recency decay."],
         title_size=13, body_size=10.5)

add_card(s2, col_left, Inches(5.7), col_w, Inches(1.15),
         title="4. MMR Diversity Selector & Mixer",
         subtitle="User-centered discovery re-ranking",
         body_points=["• Balances relevance vs topic spread (λ=0.70) and applies per-author exposure caps."],
         title_size=13, body_size=10.5)

# Right: Framed Prototype Interface
add_mockup_panel(s2, "varsity_refined_layout.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Varsity Pipeline: Real-time telemetry, candidate hydration & Recommender Control")


# ==============================================================================
# SLIDE 3: IMPLICIT VS. EXPLICIT FEEDBACK
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
add_slide_background(s3)
add_header(s3, "User Feedback: Modeling Implicit Interactions", "IMPLICIT VS. EXPLICIT FEEDBACK & CONFIDENCE MODELS")

# Left Column: Theory + Formula Callout
add_card(s3, Inches(0.8), Inches(1.8), Inches(6.1), Inches(2.2),
         title="Beyond 1-to-5 Star Rating Limitations",
         subtitle="Social feeds generate over 99% implicit events",
         body_points=[
             "• The Social Recommender Challenge: Students rarely submit explicit star ratings. Engagement occurs via scrolls, likes, and replies.",
             "• Zero As Missing, Not Negative: Unobserved posts are never treated as negative dislikes; the user simply may not have scrolled to them.",
             "• Negative Dwell Sampling: Fast scrolls (< 1.5s dwell time) are logged as implicit skips."
         ],
         title_size=13.5, body_size=10.5)

add_card(s3, Inches(0.8), Inches(4.15), Inches(6.1), Inches(2.7),
         title="Confidence-Weighted Matrix Formulation (Hu et al.)",
         subtitle="Mathematical basis for Varsity's interaction weights",
         body_points=[
             "• Confidence Equation:  C_ui = 1 + alpha * r_ui   (alpha = 40)",
             "• Engaged Post (Like / Reply): C_ui = 41 (High certainty parameter update)",
             "• Unobserved Post (Baseline): C_ui = 1 (Low certainty baseline expectation)",
             "• Action Weights: Like (+1.0), Reply (+2.0), Bookmark (+1.5), Skip (-0.5)",
             "• Explicit 1-5 holdouts are preserved strictly to evaluate offline rating RMSE."
         ],
         title_size=13.5, body_size=10.5)

# Right Column: Zoomed Dark Screenshot + Telemetry Explainer
add_mockup_panel(s3, "crop_post_actions.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(2.7),
                 caption="Prototype Zoom: Multi-action interaction bar (Like, Reply, Bookmark, Skip)")

add_card(s3, Inches(7.2), Inches(4.65), Inches(5.3), Inches(2.2),
         title="Multimodal Feedback in Action",
         subtitle="Live conversion from student UI actions to model matrices",
         body_points=[
             "• Real-time Dwell Tracker: Measures viewport time per post.",
             "• Active Actions: Clicking ❤️ or 💬 instantly boosts post affinity.",
             "• Skip Action (✕): Immediately records implicit skip to suppress duplicate content.",
             "• Offline RMSE (1.3726) confirms model rating fidelity."
         ],
         title_size=13, body_size=10.5)


# ==============================================================================
# SLIDE 4: SIMILARITY MEASURES (PEARSON VS. COSINE)
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
add_slide_background(s4)
add_header(s4, "Similarity Metrics: Pearson Correlation vs. Cosine", "SIMILARITY MEASURES & MEMORY-BASED METHODS")

# Left Column: 3 Dedicated Similarity Concept Cards
add_card(s4, Inches(0.8), Inches(1.8), Inches(6.1), Inches(1.6),
         title="Cosine Similarity (Angle Alignment)",
         subtitle="Applied to Content TF-IDF Matching",
         body_points=[
             "• Formula:  Cos(u, v) = (u · v) / (||u|| * ||v||)",
             "• Why Cosine for Content: Term frequencies and hashtag vectors are strictly non-negative, making cosine angle optimal for semantic matching."
         ],
         title_size=13, body_size=10.5)

add_card(s4, Inches(0.8), Inches(3.55), Inches(6.1), Inches(1.6),
         title="Pearson Correlation (Mean-Centered De-biasing)",
         subtitle="Applied to Collaborative User-kNN (k = 20)",
         body_points=[
             "• Formula:  Pearson(u, v) = Sum((r_ui - r_bar_u)(r_vi - r_bar_v)) / Norm",
             "• Removes User Bias: Subtracting student mean rating (r_bar) eliminates distortion between active likers and strict lurkers."
         ],
         title_size=13, body_size=10.5)

add_card(s4, Inches(0.8), Inches(5.3), Inches(6.1), Inches(1.55),
         title="Significance Weighting De-biasing",
         subtitle="Shrinkage factor for sparse co-occurrences",
         body_points=[
             "• Formula:  Weight = min(|I_uv|, tau) / tau   (tau = 5)",
             "• Prevents spurious 1.0 correlations when two students share only 1 common like."
         ],
         title_size=13, body_size=10.5)

# Right Column: Zoomed Dark Screenshot + Metric Breakdown
add_mockup_panel(s4, "crop_similarity_bars.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(2.7),
                 caption="Live Explainer: Real-time similarity scores computed per candidate")

add_card(s4, Inches(7.2), Inches(4.65), Inches(5.3), Inches(2.2),
         title="Deconstructed Candidate Affinity",
         subtitle="How prototype combines diverse similarity perspectives",
         body_points=[
             "• Social Network Affinity (43.1%): Shortest-path distance in follow graph.",
             "• Topic Alignment (8.6%): Cosine match with user TF-IDF centroid.",
             "• iALS Latent Match (34.5%): Dot product of 32-dim latent factor vectors.",
             "• Result: Transparent multi-faceted relevance ranking."
         ],
         title_size=13, body_size=10.5)


# ==============================================================================
# SLIDE 5: MATRIX FACTORIZATION & iALS
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
add_slide_background(s5)
add_header(s5, "Model-Based RecSys: Latent Factors & iALS", "MATRIX FACTORIZATION, SVD & LATENT FACTORS")

# Left Column: Stat Chips + iALS Theory
add_stat_chip(s5, Inches(0.8), Inches(1.8), Inches(2.95), Inches(1.15),
              "98.7%", "Matrix Sparsity", "400 x 3,500 = 1.4M cells (22k ratings)")

add_stat_chip(s5, Inches(3.95), Inches(1.8), Inches(2.95), Inches(1.15),
              "32 Dims", "Latent Factor Space", "Semantic topics (e.g. hackathons, research)")

add_card(s5, Inches(0.8), Inches(3.1), Inches(6.1), Inches(1.8),
         title="Why Standard SVD Fails on Social Feeds",
         subtitle="Overcoming explicit zeros and memory bottlenecks",
         body_points=[
             "• The Zero Bias Issue: TruncatedSVD treats missing values as zero ratings, incorrectly penalizing unexposed campus notices.",
             "• Sparsity Barrier: Memory k-NN suffers empty vector intersections under 98.7% sparsity; latent factor models generalize across zero overlap."
         ],
         title_size=13, body_size=10.5)

add_card(s5, Inches(0.8), Inches(5.05), Inches(6.1), Inches(1.8),
         title="Alternating Least Squares (iALS) Closed-Form",
         subtitle="Weighted Ridge Linear Regression update",
         body_points=[
             "• Closed-Form Update:  x_u = (Y^T * C^u * Y + lambda * I)^(-1) * Y^T * C^u * p_u",
             "• Alternates user and item updates with L2 regularization (lambda = 0.05).",
             "• Delivers O(1) latent dot-product inference for live candidate retrieval!"
         ],
         title_size=13, body_size=10.5)

# Right Column: Framed Dark Screenshot of Why This Post
add_mockup_panel(s5, "why_this_post.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Prototype Callout: iALS latent match contributing 34.5% to candidate score")


# ==============================================================================
# SLIDE 6: CONTENT-BASED FILTERING & COLD-START SOLUTION
# ==============================================================================
s6 = prs.slides.add_slide(blank_layout)
add_slide_background(s6)
add_header(s6, "Content-Based Profiles & The Cold-Start Solution", "ITEM PROFILES, TF-IDF & SWITCHING HYBRIDS")

# Left Column: The Problem, The Solution & The Switching Rule
add_card(s6, Inches(0.8), Inches(1.8), Inches(6.1), Inches(1.8),
         title="The Student Cold-Start Challenge",
         subtitle="When a new freshman enters with zero interaction history",
         body_points=[
             "• Cold-Start Dilemma: Collaborative filtering (k-NN & iALS) requires prior ratings. When interaction history is empty, CF models output zero.",
             "• Campus Solution: Exploit semantic textual metadata and department knowledge priors to bootstrap new students immediately."
         ],
         title_size=13, body_size=10.5)

add_card(s6, Inches(0.8), Inches(3.75), Inches(6.1), Inches(1.85),
         title="TF-IDF Profiles & Centroid Vectors",
         subtitle="Dynamic topic modeling for posts and students",
         body_points=[
             "• 5,000 N-Gram Dims: Extracts features from post text, department names, and campus hashtags (#cs, #hackathon, #midterm).",
             "• Dynamic Centroid: Each student's interest profile is the running centroid of text vectors for posts they liked or bookmarked."
         ],
         title_size=13, body_size=10.5)

add_card(s6, Inches(0.8), Inches(5.75), Inches(6.1), Inches(1.1),
         title="The Switching Hybrid Rule & Proof",
         subtitle="Warm (>= 3 interactions) → iALS | Cold (< 3) → Content TF-IDF",
         body_points=["• Cold-start NDCG@10 leaps from 0.0109 to 0.6719 (+61.6x improvement!)"],
         title_size=12.5, body_size=10.5)

# Right Column: Framed Dark Screenshot of Persona Modal
add_mockup_panel(s6, "persona_modal.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Persona Switcher: 'Charlie (Freshman • Cold Start)' routes to knowledge switch vs Warm Users")


# ==============================================================================
# SLIDE 7: HEAVY RANKER & UTILITY SCORING
# ==============================================================================
s7 = prs.slides.add_slide(blank_layout)
add_slide_background(s7)
add_header(s7, "Heavy Ranker: Calibrated Probabilities & Age Decay", "CLASSIFICATION ALGORITHMS & FEATURE COMBINATION")

# Left Column: Heavy Ranker Structure & Equations
add_card(s7, Inches(0.8), Inches(1.8), Inches(6.1), Inches(2.2),
         title="Multi-Action Engagement Classifiers",
         subtitle="12-Dimensional hydrated feature representation",
         body_points=[
             "• Feature Space: Author follower count, in-network hop distance, tag overlap, iALS score, and timestamp recency.",
             "• 3 Calibrated Logistic Regression Models:",
             "   - P(Like): Probability the student likes the post.",
             "   - P(Reply): Probability the student starts a discussion.",
             "   - P(Skip): Probability the student scrolls past / ignores (dwell < 1.5s)."
         ],
         title_size=13.5, body_size=10.5)

add_card(s7, Inches(0.8), Inches(4.15), Inches(6.1), Inches(2.7),
         title="Expected Utility Scoring Formula",
         subtitle="Balancing positive interactions against feed fatigue",
         body_points=[
             "• Utility Equation:",
             "   Score = [w_like * P(Like) + w_reply * P(Reply) - w_skip * P(Skip)] * e^(-lambda_age * dt)",
             "• Weights: w_like = 0.60, w_reply = 1.20, w_skip = 0.40",
             "• Recency Half-Life Decay: Exp(-0.02 * dt) ensures urgent campus deadlines rank above stale high-scoring posts."
         ],
         title_size=13.5, body_size=10.5)

# Right Column: Zoomed Dark Screenshot + Live Formula Breakdown
add_mockup_panel(s7, "crop_heavy_ranker_probs.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(2.7),
                 caption="Heavy Ranker Output: P(Like)=40%, P(Reply)=16%, P(Skip)=60% -> Score: 0.1367")

add_card(s7, Inches(7.2), Inches(4.65), Inches(5.3), Inches(2.2),
         title="Live Prototype Calculation Proof",
         subtitle="Exact verification of values shown in screenshot",
         body_points=[
             "• Raw Utility: (0.60 * 0.40) + (1.20 * 0.16) - (0.40 * 0.60) = 0.192",
             "• Elapsed Hours: dt = 15 hours",
             "• Recency Factor: exp(-0.02 * 15) = 0.7408",
             "• Final Utility Score: 0.192 * 0.7118 = 0.1367 (Exact match!)"
         ],
         title_size=13, body_size=10.5)


# ==============================================================================
# SLIDE 8: DIVERSITY VS. ACCURACY (MMR SELECTOR)
# ==============================================================================
s8 = prs.slides.add_slide(blank_layout)
add_slide_background(s8)
add_header(s8, "Diversity vs. Accuracy: Maximal Marginal Relevance", "INTRA-LIST DIVERSITY & RE-RANKING")

# Left Column: Filter Bubble Problem + MMR Formulation + Empirical Results
add_card(s8, Inches(0.8), Inches(1.8), Inches(6.1), Inches(2.1),
         title="Preventing Campus Filter Bubbles",
         subtitle="Trading marginal relevance for catalog discovery",
         body_points=[
             "• The Echo Chamber Dilemma: A ranker optimized purely for P(Like) fills the feed with 10 identical posts on one topic (e.g. #midterm), causing user fatigue.",
             "• Intra-List Diversity (ILD): Measures pairwise distance between recommended items to guarantee topical spread."
         ],
         title_size=13.5, body_size=10.5)

add_card(s8, Inches(0.8), Inches(4.05), Inches(6.1), Inches(2.8),
         title="Maximal Marginal Relevance (MMR) Formulation",
         subtitle="Carbonell & Goldstein greedy diversity re-ranking",
         body_points=[
             "• MMR Formula:  argmax [ lambda * Sim1(d_i, u) - (1 - lambda) * max Sim2(d_i, d_j) ]",
             "• Tradeoff Slider: lambda = 1.0 (pure relevance) vs lambda = 0.0 (maximum novelty).",
             "• Quantified Ablation Proof:",
             "   - Without MMR (lambda = 1.0): Intra-List Diversity (ILD) = 0.9152.",
             "   - With MMR (lambda = 0.70): Diversity jumps to 0.9616 (+5.1% catalog variety) with zero loss in Top-10 Precision!"
         ],
         title_size=13.5, body_size=10.5)

# Right Column: Framed Dark Mockup of Control Drawer
add_mockup_panel(s8, "crop_drawer_sliders.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Control Drawer: Interactive MMR Diversity Lambda slider and author cap constraints")


# ==============================================================================
# SLIDE 9: SECURITY & SHILLING ATTACK DEFENSE
# ==============================================================================
s9 = prs.slides.add_slide(blank_layout)
add_slide_background(s9)
add_header(s9, "Security & Robustness: Shilling Attacks & Defenses", "SECURITY IN RECOMMENDER SYSTEMS & DEFENSES")

# Left Column: Vulnerability, Attack Simulation & Anomaly Defense
add_card(s9, Inches(0.8), Inches(1.8), Inches(6.1), Inches(1.4),
         title="The Collaborative Vulnerability",
         subtitle="Recommenders assume all ratings represent authentic humans",
         body_points=[
             "• Threat: Malicious actors inject fake accounts to manipulate item rankings.",
             "• Campus Context: Rival student campaigns or spam bots gaming top visibility."
         ],
         title_size=13, body_size=10.5)

add_card(s9, Inches(0.8), Inches(3.35), Inches(6.1), Inches(1.7),
         title="🚀 Live Bandwagon Attack Simulation",
         subtitle="Targeting spam post #42 with 15 coordinated bot accounts",
         body_points=[
             "• Mechanism: Bots inject fake likes on high-degree popular posts + target #42.",
             "• Attack Impact: Artificially promotes post #42 from organic rank #45 straight to rank #3!"
         ],
         border_color=ACCENT_RED, title_size=13, body_size=10.5)

add_card(s9, Inches(0.8), Inches(5.2), Inches(6.1), Inches(1.65),
         title="🛡️ Unsupervised Jaccard Anomaly Filter",
         subtitle="Real-time co-action graph defense",
         body_points=[
             "• Anomaly Signature: Genuine students have varied profiles; bot rings form dense clusters with Jaccard profile overlap > 0.65.",
             "• Defense Action: Quarantines bot accounts and immediately restores organic feed."
         ],
         border_color=ACCENT_GREEN, title_size=13, body_size=10.5)

# Right Column: Framed Dark Mockup of Attack Sandbox
add_mockup_panel(s9, "crop_sandbox_controls.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Attack Sandbox: Simulating Bandwagon botnets and activating real-time Jaccard defense")


# ==============================================================================
# SLIDE 10: EMPIRICAL EVALUATION & BENCHMARKS
# ==============================================================================
s10 = prs.slides.add_slide(blank_layout)
add_slide_background(s10)
add_header(s10, "Empirical Evaluation: Beyond-Accuracy Benchmarks", "ERROR, DECISION SUPPORT & USER-CENTERED METRICS")

# Left Column: 3 Metric Tiers + Ablation Insights
add_card(s10, Inches(0.8), Inches(1.8), Inches(6.1), Inches(2.7),
         title="Comprehensive Evaluation Framework",
         subtitle="Offline benchmarking across 8 models on 80/20 temporal splits",
         body_points=[
             "• Error Metrics (Rating Fidelity):",
             "   - RMSE = 1.3726 (User-kNN) on 1-to-5 star explicit holdout.",
             "• Decision Support Metrics (Ranking Quality):",
             "   - NDCG@10 (0.0186): Position-discounted ranking quality.",
             "   - Precision@10 (0.0120) & Recall@10 (0.0156): Ground truth interaction capture.",
             "• User-Centered Metrics (Discovery Quality):",
             "   - Catalog Coverage (10.0%): Prevents popular post monopoly.",
             "   - Intra-List Diversity (0.9599): Guarantees topic spread."
         ],
         title_size=13.5, body_size=10.5)

add_card(s10, Inches(0.8), Inches(4.65), Inches(6.1), Inches(2.2),
         title="Key Empirical Takeaways",
         subtitle="Why the multi-stage hybrid dominates single baselines",
         body_points=[
             "• Varsity Hybrid achieves top NDCG@10 (0.0186), beating iALS (0.0139) and k-NN (0.0109).",
             "• Cold-start switching hybrid boosts cold user NDCG from 0.0109 to 0.6719.",
             "• MMR integration achieves 0.9599 diversity without harming Top-10 precision."
         ],
         title_size=13, body_size=10.5)

# Right Column: Framed Dark Mockup of Benchmark Table
add_mockup_panel(s10, "benchmark_table.png", Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.05),
                 caption="Evaluation Matrix: Benchmarks across Random, Popularity, k-NN, iALS, and Hybrid")


# ==============================================================================
# SLIDE 11: PROJECT REFERENCES & RESEARCH FOUNDATIONS
# ==============================================================================
s11 = prs.slides.add_slide(blank_layout)
add_slide_background(s11)
add_header(s11, "Project References & Theoretical Foundations", "CORE RESEARCH FOUNDATIONS & ARCHITECTURE PAPERS")

# Left Card: Core Algorithmic Foundations
add_card(s11, Inches(0.8), Inches(1.8), Inches(5.7), Inches(5.05),
         title="Algorithmic Foundations",
         subtitle="Seminal literature behind models implemented in Varsity",
         body_points=[
             "• Yifan Hu, Yehuda Koren, Chris Volinsky (2008)",
             "   'Collaborative Filtering for Implicit Feedback Datasets', IEEE ICDM.",
             "   - Theoretical foundation for Varsity's confidence matrix (C_ui = 1 + alpha * r_ui) and alternating least squares (iALS) closed-form updates.",
             "",
             "• Jaime Carbonell, Jade Goldstein (1998)",
             "   'The Use of MMR, Diversity-Based Reranking', ACM SIGIR.",
             "   - Mathematical basis for Varsity's Maximal Marginal Relevance diversity reranking slider (lambda = 0.70).",
             "",
             "• Robin Burke (2002)",
             "   'Hybrid Recommender Systems: Survey and Experiments', UMUAI.",
             "   - Architectural framework for Varsity's Switching (cold-start fallback to content TF-IDF) and Cascaded pipeline stages."
         ],
         title_size=14.5, body_size=10.5)

# Right Card: Systems Architecture & Collaborative Robustness
add_card(s11, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.05),
         title="Architecture & Security Literature",
         subtitle="Industry design blueprints and shilling attack defenses",
         body_points=[
             "• X Corp. / Twitter Open Source (2023)",
             "   'The Algorithm: Recommendation Pipeline Architecture', GitHub.",
             "   - Blueprint for Varsity's multi-stage lifecycle: Sub-5ms Candidate Sourcing -> 12-dim Hydration -> Heavy Ranking -> Diversity Mixing.",
             "",
             "• Badrul Sarwar, George Karypis, Joseph Konstan, John Riedl (2001)",
             "   'Item-Based Collaborative Filtering Recommendation Algorithms', ACM WWW.",
             "   - Basis for Item-kNN co-occurrence scoring, Pearson de-biasing, and significance weighting under sparse rating matrices.",
             "",
             "• Bamshad Mobasher, Robin Burke, Runa Bhaumik, Chad Williams (2007)",
             "   'Toward Trustworthy Recommender Systems: Analysis of Shilling Attacks', ACM TIST.",
             "   - Theoretical foundation for Varsity's Bandwagon attack simulation and real-time unsupervised Jaccard anomaly filter."
         ],
         title_size=14.5, body_size=10.5)


# Save presentation
output_pptx = os.path.abspath("Varsity_RecSys_Presentation.pptx")
prs.save(output_pptx)
print(f"Successfully generated beautified {output_pptx} with {len(prs.slides)} slides!")
