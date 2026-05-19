"""
app.py — AuditCV v2
────────────────────────
Adaptive, AI-driven explainable resume intelligence for all Computer Science domains.
Entry point: streamlit run app.py
"""

import streamlit as st
import json
import plotly.graph_objects as go
import requests
from io import BytesIO
import re

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="AuditCV — AI Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (Web Architecture Overhaul) ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAFAFA;
    color: #111111;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stAppDeployDropdown {display: none !important;}
div[data-testid="stSidebar"] {display: none !important;}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* Premium Top Navigation Bar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid #111111;
    padding: 0.75rem 2rem;
    margin-bottom: 2.5rem;
    border-radius: 0px;
}

.nav-brand {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: bold;
    color: #111111;
    letter-spacing: -0.5px;
}

.nav-tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #666666;
    margin-left: 10px;
    border-left: 1px solid #E5E5E5;
    padding-left: 10px;
}

/* Massive Impact Header Title */
.main-title-container {
    text-align: center;
    margin-bottom: 3rem;
    margin-top: 1rem;
}

.main-title {
    font-family: 'DM Mono', monospace;
    font-size: 3.4rem;
    font-weight: 700;
    color: #111111;
    letter-spacing: -2px;
    line-height: 1.1;
    margin-bottom: 10px;
}

.main-sub {
    color: #666666;
    font-size: 0.9rem;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Labels */
.lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #666666;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    margin-top: 1.5rem;
}

/* Score Hero Box */
.score-hero {
    background: #FFFFFF;
    border: 1px solid #111111;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.score-range {
    font-family: 'DM Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
}

.score-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 0px; 
    margin-top: 10px;
    letter-spacing: 1px;
}

/* Cards layout dimensions */
.dim-card {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 0px;
    padding: 1rem 1.2rem;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.01);
}

.dim-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #666666;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}

.dim-score {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    line-height: 1;
}

.dim-range {
    font-size: 0.65rem;
    color: #666666;
    font-family: 'DM Mono', monospace;
    margin-left: 4px;
}

.dim-label {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    margin-top: 4px;
}

.dim-expl {
    font-size: 0.72rem;
    color: #333333;
    margin-top: 6px;
    line-height: 1.5;
}

/* Suggestions Engine UI */
.sug-card {
    background: #FFFFFF;
    border: 1px solid #111111;
    border-left: 6px solid #111111;
    border-radius: 0px;
    padding: 1.2rem;
    margin-bottom: 12px;
}

.sug-cat {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #666666;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

.sug-issue {
    font-size: 0.83rem;
    color: #111111;
    margin-bottom: 6px;
    line-height: 1.6;
}

.sug-action {
    font-size: 0.8rem;
    color: #111111;
    font-weight: bold;
    line-height: 1.6;
}

.sug-example {
    font-size: 0.75rem;
    color: #333333;
    font-family: 'DM Mono', monospace;
    margin-top: 6px;
    line-height: 1.5;
    padding: 6px 8px;
    background: #EEEEEE;
    border-radius: 0px;
}

/* Risk Badges */
.risk-high { background: #111111; color: #FFFFFF; border: 1px solid #111111; }
.risk-medium { background: #666666; color: #FFFFFF; border: 1px solid #666666; }
.risk-low { background: #FFFFFF; color: #111111; border: 1px solid #111111; }

.pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 0px;
    margin: 2px;
}
.pill-warn { background: #111111; color: #FFFFFF; border: 1px solid #111111; }
.pill-ok { background: #FFFFFF; color: #111111; border: 1px solid #111111; }

.skill-row-ok { color: #111111; font-family: 'DM Mono', monospace; font-size: 0.78rem; }
.skill-row-warn { color: #666666; font-family: 'DM Mono', monospace; font-size: 0.78rem; }
.ev-item { font-size: 0.75rem; color: #111111; padding: 2px 0; font-family: 'DM Mono', monospace; }
.pen-item { font-size: 0.75rem; color: #666666; text-decoration: underline; padding: 2px 0; font-family: 'DM Mono', monospace; }

hr.s {
    border: none;
    border-top: 1px solid #111111;
    margin: 2rem 0;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: #111111 !important;
    color: white !important;
    border: 1px solid #111111 !important;
    font-weight: 600 !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 0px !important;
    width: 100% !important;
    padding: 0.5rem 1rem !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #111111 !important;
    font-weight: 600 !important;
}

.profile-card {
    background: #FFFFFF;
    border: 1px solid #111111;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.social-link {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #111111;
    text-decoration: none;
    margin-right: 15px;
    font-weight: bold;
}
.social-link:hover {
    text-decoration: underline;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #666666 !important;
}

.stTabs [aria-selected="true"] {
    color: #111111 !important;
    border-bottom-color: #111111 !important;
}

/* Premium Website Footer */
.web-footer {
    margin-top: 5rem;
    background: #FFFFFF;
    border-top: 1px solid #111111;
    padding: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #111111;
}

.footer-right {
    display: flex;
    gap: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state Initialization ──────────────────────────────────────────────
for key in ["result", "filename"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── HELPER DATA GENERATION PARSERS ────────────────────────────────────────────
def _color(score: int) -> str:
    if score >= 80: return "#111111"
    if score >= 60: return "#333333"
    if score >= 40: return "#555555"
    return "#777777"


def _risk_class(risk: str) -> str:
    return {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk, "risk-low")


def _bar_chart(dims) -> go.Figure:
    names = [d.name.replace(" & ", "\n& ") for d in dims]
    scores = [d.raw_score for d in dims]
    colors = [_color(s) for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s}" for s in scores], textposition="outside",
        textfont=dict(family="DM Mono", size=11, color="#111111"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], visible=False),
        yaxis=dict(tickfont=dict(family="DM Mono", size=10, color="#111111")),
        margin=dict(t=10, b=10, l=10, r=60), height=260, showlegend=False,
    )
    return fig


def _radar_chart(dims) -> go.Figure:
    labels = [d.name for d in dims]
    values = [d.raw_score for d in dims]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(17,17,17,0.05)",
        line=dict(color="#111111", width=1.5), marker=dict(color="#111111", size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#111111", size=9, family="DM Mono"),
                            gridcolor="#E5E5E5", linecolor="#111111"),
            angularaxis=dict(tickfont=dict(color="#111111", size=10, family="DM Mono"), gridcolor="#E5E5E5",
                             linecolor="#111111"),
        ),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20), height=300,
    )
    return fig


def _generate_markdown_report(filename, scoring, ag, suggestions) -> str:
    report = [f"# AuditCV Evaluation Report — {filename}", "=" * 50 + "\n", "## OVERALL SUMMARY"]
    report.append(
        f"- **Overall Strength:** {scoring.overall_range}\n- **Score Class:** {scoring.overall_label}\n- **Experience Tier Mode:** {scoring.tier_mode}\n- **Authenticity Risk:** {ag.overall_risk}\n- **Confidence Level:** {scoring.confidence}\n")
    report.append("## DIMENSION BREAKDOWN")
    for d in scoring.dimensions:
        report.append(
            f"### {d.name}\n- **Score:** {d.raw_score}/100 ({d.score_range})\n- **Assessment:** {d.label}\n- **Explanation:** {d.explanation}")
    return "\n".join(report)


def _build_master_prompt(filename, scoring, ag) -> str:
    return f"""You are an elite Resume Engineering Consultant and Technical Copywriter. I have audited my resume using Aravind Yedida's AuditCV engine for file [{filename}], and I need you to completely re-write and update my resume lines to fix my weak dimensions.

Here is the exact analysis payload data from my audit session report:
- Overall strength level evaluated: {scoring.overall_range} ({scoring.overall_label})
- Experience Level Evaluation Target: {scoring.tier_mode}
- Fraud and Authenticity Vetting Risk: {ag.overall_risk}
- Authenticity Alignment Score: {ag.authenticity_score}/100

CRITICAL RULES FOR RE-WRITING MY CONTENT:
1. DO NOT add generic corporate buzzwords or stuffed keywords. Keep the tone completely objective and metric-backed.
2. Remove any suspicious or exaggerated metric claims. Keep the growth rates realistic for an entry-level professional.
3. Every project and experience description line MUST explicitly link the listed technical skills to direct, concrete evidence. Do not just say I know a skill; rewrite the bullet point to describe exactly HOW I applied it.
4. Format all technical summaries using standard high-impact Action Verbs.

Please review my raw text lines and rewrite them to perfectly maximize my objective strength according to this engineering framework!"""


# ── WEBSITE TOP NAVIGATION BAR HEADER ─────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div>
        <span class="nav-brand">// AuditCV</span>
        <span class="nav-tagline">Adaptive AI Engine · Supporting all Computer Science Domains</span>
    </div>
    <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem; font-weight: 500; color: #666666;">
        aravindyedida.com
    </div>
</div>
""", unsafe_allow_html=True)

# ── CENTRALIZED APP PAGE CONTROL ──────────────────────────────────────────────
app_mode = st.selectbox(
    "Select App Page Selection Pipeline",
    ["🎯 Resume Intelligence Engine", "👨‍💻 About Developer Portfolio", "💬 Strategic Feedback Hub"],
    label_visibility="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1: RESUME INTELLIGENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "🎯 Resume Intelligence Engine":
    from core.extractor import extract
    from core.parser import parse
    from core.anti_gaming import analyse as ag_analyse
    from core.embeddings import match_skills
    from core.scorer import score
    from core.suggestions import generate as gen_suggestions

    st.markdown("""
    <div class="main-title-container">
        <p class="main-title">AuditCV</p>
        <p class="main-sub">Adaptive Domain Mapping · Fluid Skill Extraction · Tiered Scoring for CS Freshers & Experts</p>
    </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded = st.file_uploader("Upload candidate resume string data (PDF format)", type=["pdf"],
                                    label_visibility="visible")
    with up_col2:
        st.markdown("<p style='margin-top:1.8rem;'></p>", unsafe_allow_html=True)
        analyse_btn = st.button("🎯 Run Adaptive Vetting Audit", use_container_width=True)

    # Secure Token Retrieval from Streamlit Secret Store
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")

    if analyse_btn and uploaded:
        file_bytes = uploaded.read()
        prog = st.progress(0)

        # 1. High-Fidelity Extraction Block
        extraction = extract(file_bytes, uploaded.name)
        prog.progress(20)

        # 2. ADAPTIVE AI DOMAIN CLASSIFIER AND PROFILE TIER GATEKEEPER
        # Calls the dynamic parsing module to verify domain fit via Gemini
        from core.parser import classify_profile_domain

        domain_meta = classify_profile_domain(extraction.text, gemini_key)

        if not domain_meta.get("is_computer_science", False):
            st.error(
                f"🚨 Domain Access Restriction: The system cannot score your resume out of the domain as it currently supports Computer Science domains only. "
                f"However, Aravind is trying to develop and improving best upgrades in features to expand backend classification matrices soon!"
            )
            st.stop()

        prog.progress(40)

        # 3. Fluid JSON Structural Breakdowns & Extracted Skills Handling
        parsed = parse(extraction.text, identified_skills=domain_meta.get("extracted_skills", []))
        parsed.is_fresher = domain_meta.get("is_fresher", True)

        prog.progress(60)
        ag = ag_analyse(parsed, file_bytes)
        prog.progress(80)

        # 4. Semantic Evidence Alignment & Variable Scoring Matrix Call
        emb = match_skills(parsed.skills, parsed.projects, parsed.experience, parsed.internship)
        scoring = score(parsed, ag, emb, extraction.confidence,
                        tier_mode="FRESHER" if parsed.is_fresher else "EXPERIENCED")
        suggestions = gen_suggestions(parsed, scoring, ag, gemini_key)
        prog.progress(100)

        st.session_state.result = {"extraction": extraction, "parsed": parsed, "ag": ag, "emb": emb, "scoring": scoring,
                                   "suggestions": suggestions}
        st.session_state.filename = uploaded.name

    if st.session_state.get("result") is not None:
        R = st.session_state.result
        extraction, parsed, ag, emb, scoring, suggestions = R["extraction"], R["parsed"], R["ag"], R["emb"], R[
            "scoring"], R["suggestions"]
        overall_color = _color(scoring.overall_score)

        st.markdown("<hr class='s'>", unsafe_allow_html=True)

        act_col1, act_col2 = st.columns([1, 1])
        with act_col1:
            report_data = _generate_markdown_report(st.session_state.filename, scoring, ag, suggestions)
            st.markdown(
                "<span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#666666;'>STEP 1: GET SYSTEM ANALYSIS METRICS</span>",
                unsafe_allow_html=True)
            st.download_button(label="📥 Download System Audit Report (.txt)", data=report_data,
                               file_name=f"AuditCV_Report_{st.session_state.filename.replace('.pdf', '')}.txt")
        with act_col2:
            st.markdown(
                "<span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#111111;font-weight:bold;'>🛠️ STEP 2: REWRITE RESUME VIA MASTER PROMPT</span>",
                unsafe_allow_html=True)
            generated_prompt = _build_master_prompt(st.session_state.filename, scoring, ag)
            with st.popover("🔥 Open Master Optimization AI Prompt"):
                st.markdown(
                    "<p style='font-size:0.78rem;color:#666666;'>Copy the engineered prompt below and paste it into ChatGPT/Claude along with your downloaded audit report to update your text architecture:</p>",
                    unsafe_allow_html=True)
                st.code(generated_prompt, language="text")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Overview Matrix", "🔍 Deep Evidence Vetting", "💡 Optimization Suggestions", "📄 Core Raw JSON Data"])

        with tab1:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                risk_cls = _risk_class(ag.overall_risk)
                tier_badge_txt = "FRESHER EVAL TIERS" if scoring.tier_mode == "FRESHER" else "EXPERT EVAL TIERS"
                st.markdown(
                    f"""<div class='score-hero'><div class='lbl' style='margin-top:0;'>// overall_strength</div><div class='score-range' style='color:{overall_color};'>{scoring.overall_range}</div><div style='font-family:DM Mono,monospace;font-size:0.8rem;color:{overall_color};margin-top:6px;'>{scoring.overall_label}</div><div style='margin-top:10px;'><span class='score-badge' style='background:#111111;color:#FFFFFF;'>{tier_badge_txt}</span></div><div style='margin-top:6px;'><span class='score-badge {risk_cls}'>AUTH RISK: {ag.overall_risk}</span></div><div style='margin-top:10px; font-family:DM Mono,monospace; font-size:0.65rem; color:#111111;'>Confidence: {scoring.confidence}</div></div>""",
                    unsafe_allow_html=True)
            with c2:
                st.markdown("<p class='lbl' style='margin-top:0;'>// extraction_quality</p>", unsafe_allow_html=True)
                st.markdown(
                    f"""<div class='dim-card'><div class='dim-name'>PDF EXTRACTION</div><div class='dim-score' style='color:{_color(int(extraction.confidence * 100))};'>{extraction.confidence_label}</div><div style='font-size:0.72rem;color:#111111;margin-top:6px;font-family:DM Mono,monospace;'>Method: {extraction.method}<br>Pages: {extraction.pages} · Chars: {extraction.char_count:,}</div></div>""",
                    unsafe_allow_html=True)
                if extraction.warnings:
                    for w in extraction.warnings: st.markdown(f"<div class='pen-item'>⚠ {w}</div>",
                                                              unsafe_allow_html=True)
            with c3:
                st.markdown("<p class='lbl' style='margin-top:0;'>// dynamic_extracted_skills</p>",
                            unsafe_allow_html=True)
                if parsed.skills:
                    skills_html = "".join(f"<span class='pill pill-ok'>{s}</span>" for s in parsed.skills[:12])
                    st.markdown(f"<div style='max-height:120px; overflow-y:auto;'>{skills_html}</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<span class='pill pill-warn'>✗ No distinct skills extracted</span>",
                                unsafe_allow_html=True)

            st.markdown("<hr class='s'>", unsafe_allow_html=True)
            col_bar, col_radar = st.columns([1.2, 1])
            with col_bar:
                st.markdown("<p class='lbl'>// dimension_scores</p>", unsafe_allow_html=True)
                st.plotly_chart(_bar_chart(scoring.dimensions), use_container_width=True)
            with col_radar:
                st.markdown("<p class='lbl'>// radar</p>", unsafe_allow_html=True)
                st.plotly_chart(_radar_chart(scoring.dimensions), use_container_width=True)

            st.markdown("<hr class='s'>", unsafe_allow_html=True)
            st.markdown("<p class='lbl'>// dimension_breakdown</p>", unsafe_allow_html=True)
            cols = st.columns(4)
            for i, dim in enumerate(scoring.dimensions):
                c = _color(dim.raw_score)
                with cols[i % 4]:
                    ev_html = "".join(f"<div class='ev-item'>✓ {e}</div>" for e in dim.evidence[:2])
                    pen_html = "".join(f"<div class='pen-item'>✗ {p}</div>" for p in dim.penalties[:2])
                    st.markdown(
                        f"""<div class='dim-card'><div class='dim-name'>// {dim.name.upper()}</div><div><span class='dim-score' style='color:{c};'>{dim.raw_score}</span><span class='dim-range'>/ 100</span></div><div class='dim-label' style='color:{c};'>{dim.label}</div><div class='dim-expl'>{dim.explanation}</div><div style='margin-top:8px;'>{ev_html}{pen_html}</div></div>""",
                        unsafe_allow_html=True)

        with tab2:
            da1, da2 = st.columns([1, 1])
            with da1:
                st.markdown("<p class='lbl'>// full_evidence_breakdown</p>", unsafe_allow_html=True)
                for dim in scoring.dimensions:
                    with st.expander(f"{dim.name}  ·  {dim.raw_score}/100"):
                        if dim.evidence:
                            st.markdown("**Evidence (what worked):**")
                            for e in dim.evidence: st.markdown(f"<div class='ev-item'>✓ {e}</div>",
                                                               unsafe_allow_html=True)
                        if dim.penalties:
                            st.markdown("**Penalties (what dragged score down):**")
                            for p in dim.penalties: st.markdown(f"<div class='pen-item'>✗ {p}</div>",
                                                                unsafe_allow_html=True)
                        st.caption(dim.explanation)
            with da2:
                st.markdown("<p class='lbl'>// anti_gaming_report</p>", unsafe_allow_html=True)
                risk_cls = _risk_class(ag.overall_risk)
                st.markdown(
                    f"<div class='dim-card'><div class='dim-name'>AUTHENTICITY SCORE</div><div class='dim-score' style='color:{_color(ag.authenticity_score)};'>{ag.authenticity_score}</div><div style='margin-top:8px;'><span class='score-badge {risk_cls}'>OVERALL RISK: {ag.overall_risk}</span></div></div>",
                    unsafe_allow_html=True)
                if ag.top_buzzwords:
                    bw_html = "".join(f"<span class='pill pill-warn'>{b}</span>" for b in ag.top_buzzwords)
                    st.markdown(f"<div style='margin-bottom:10px;'>{bw_html}</div>", unsafe_allow_html=True)
                if ag.unsupported_claims:
                    for claim in ag.unsupported_claims:
                        st.markdown(
                            f"<div style='background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid #111111;padding:6px 10px;margin-bottom:6px;font-size:0.75rem;font-family:DM Mono,monospace;color:#111111;'>{claim.claim}<br><span style='color:#666666;'>{claim.reason}</span></div>",
                            unsafe_allow_html=True)

                st.markdown("<hr class='s'>", unsafe_allow_html=True)
                st.markdown("<p class='lbl'>// skill_evidence_matching</p>", unsafe_allow_html=True)
                if emb.matches:
                    for m in emb.matches[:15]:
                        icon, css_cls = ("✓", "skill-row-ok") if m.supported else ("✗", "skill-row-warn")
                        ev_txt = f" → {m.evidence[:60]}..." if m.evidence else " → no evidence found"
                        st.markdown(
                            f"<div class='{css_cls}'>{icon} {m.skill} <span style='color:#666666;font-size:0.68rem;'>{ev_txt}</span></div>",
                            unsafe_allow_html=True)

        with tab3:
            if suggestions.ai_used and suggestions.ai_summary:
                st.markdown(
                    f"<div style='background:#FFFFFF;border-left:4px solid #111111;padding:1rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#333333;line-height:1.8;'><span style='font-family:DM Mono,monospace;font-size:0.62rem;color:#666666;letter-spacing:2px;display:block;margin-bottom:8px;'>// AI COACHING (Gemini)</span>{suggestions.ai_summary}</div>",
                    unsafe_allow_html=True)
            if suggestions.suggestions:
                for sug in suggestions.suggestions:
                    ex_html = f"<div class='sug-example'>Example: {sug.example}</div>" if sug.example else ""
                    st.markdown(
                        f"""<div class='sug-card'><div class='sug-cat'>#{sug.priority} · {sug.category.upper()}</div><div class='sug-issue'>{sug.issue}</div><div class='sug-action'>→ {sug.action}</div>{ex_html}</div>""",
                        unsafe_allow_html=True)
            else:
                st.success("No major issues found.")

        with tab4:
            with st.expander("📋 Parsed Resume JSON"): st.json(parsed.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2: ABOUT DEVELOPER (PORTFOLIO BRANDING)
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "👨‍💻 About Developer Portfolio":
    st.markdown("""
    <div class="main-title-container">
        <p class="main-title">Aravind Kumar Yedida</p>
        <p class="main-sub">Product Architect & System Designer</p>
    </div>
    """, unsafe_allow_html=True)

    col_dev1, col_dev2 = st.columns([2, 1])
    with col_dev1:
        st.markdown("<p class='lbl'>// Project Manifesto</p>", unsafe_allow_html=True)
        st.markdown(
            "#### \"Architected with AI, Directed by Strategy\"\n"
            "AuditCV was not created by simply asking an LLM to write code. "
            "It was built by mapping out a complete functional breakdown structure first. "
            "The engineering layout, the token verification strategy, the adaptive AI parsing layers, "
            "and the domain classification parameters were fully architected by me and generated using modular engineering prompts.\n\n"
            "This project proves that the modern developer is no longer just a coder, but an **Architect of Systems**."
        )
        st.markdown("<p class='lbl'>// Engineering Stack Ecosystem</p>", unsafe_allow_html=True)
        st.markdown(
            "- **Orchestration Layer:** Python 3.12 & Streamlit Micro-UI framework\n- **Adaptive Ingestion:** Dynamic LLM context processing (Gemini Extraction Architecture)\n- **Strategic Framework:** Rule-based Scoring Matrix (Tiered curves matching entry/expert levels)\n- **Vetting Layer:** Structural parsing algorithms checking for anomalies and keyword stuffing")
    with col_dev2:
        st.markdown("<p class='lbl'>// Digital Portals</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='profile-card'>
            <p style='font-weight:600; font-size:1.1rem; margin-top:0;'>Aravind Yedida</p>
            <p style='font-size:0.8rem; color:#666666; font-family:DM Mono,monospace; margin-bottom:1rem;'>[Data & Marketing Analyst & Builds AI Systems that really help]</p>
            <a href="http://www.aravindyedida.com" target="_blank" class="social-link">🌐 Personal Website</a><br><br>
            <a href="https://github.com/yaravind007" target="_blank" class="social-link">🐙 GitHub Repo</a><br><br>
            <a href="https://linkedin.com/in/yaravindkumar" target="_blank" class="social-link">💼 LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)


# ── VIEW 3: AUTOMATED STREAMS FEEDBACK ENGINE ─────────────────────────────────
elif app_mode == "💬 Strategic Feedback Hub":
    st.markdown("""
    <div class="main-title-container">
        <p class="main-title">Open Ecosystem Feedback</p>
        <p class="main-sub">Simplified Audit Tracker</p>
    </div>
    """, unsafe_allow_html=True)

    WEBHOOK_URL = st.secrets.get("WEBHOOK_URL",
                                 "https://script.google.com/macros/s/AKfycbxxJaDlFFq3DxI7U0tlwt_14O1fIyBB0VpQ1vHRZtyTCrVtCCDn2TTlpKw9vJ2X4jRB/exec")

    st.markdown("<p class='lbl'>// Quantitative Evaluation</p>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        project_rating = st.select_slider("Rate this project architecture:",
                                          options=["Needs Work", "Good Prototype", "Highly Impressive",
                                                   "Industry Standard Build"])
    with f_col2:
        scoring_satisfisaction = st.radio("Are you satisfied with the explainable, deterministic scoring system?",
                                          ["Yes, much better than traditional ATS", "Unsure, needs deeper testing",
                                           "No, prefer generic percentage matches"])

    st.markdown("<p class='lbl'>// Ecosystem Expansion Ideation</p>", unsafe_allow_html=True)
    future_improvements = st.text_area(
        "What feature improvements, suggestions, or brand new projects would you like to see next?",
        placeholder="e.g., Transparent screening engine...")

    if st.button("🚀 Log Feedback Session"):
        payload = {"rating": project_rating, "satisfisaction": scoring_satisfisaction,
                   "improvements": future_improvements}
        with st.spinner("Capturing metrics .."):
            try:
                response = requests.post(WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"},
                                         timeout=10)
                if response.status_code == 200 or response.ok:
                    st.success("🎯 Success! Feedback recorded.")
                else:
                    st.info("Payload compiled.")
            except Exception as e:
                st.error("System pipeline is holding. Error.")
        st.json(payload)

# ── WEBSITE BOTTOM FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="web-footer">
    <div class="footer-left">
        © 2026 AuditCV · Built Using AI | Tailored for the Global CS Community · Fully Reproducible System Metrics.
    </div>
    <div class="footer-right">
        <a href="http://www.aravindyedida.com" target="_blank" style="color:#111111; font-family:'DM Mono', monospace; font-size:0.72rem; font-weight:bold; text-decoration:none;">🌐 portfolio</a>
        <a href="https://linkedin.com/in/yaravindkumar" target="_blank" style="color:#111111; font-family:'DM Mono', monospace; font-size:0.72rem; font-weight:bold; text-decoration:none;">💼 linkedin</a>
    </div>
</div>
""", unsafe_allow_html=True)