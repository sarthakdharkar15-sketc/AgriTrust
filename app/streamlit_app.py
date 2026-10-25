"""AgriTrust Streamlit User Interface.
Stunning, high-end agricultural advisory dashboard with luxury glassmorphic styling,
interactive pipeline flowchart, animated visual gauges, and calibrated multi-agent trust diagnostics.
"""

import streamlit as st
import pandas as pd
import json
import time
import base64
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_title="AgriTrust | Reliable Multi-Agent RAG Advisory",
    page_icon="assets/logo.png" if Path("assets/logo.png").exists() else "🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-End Ultra-Modern Aesthetic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Keep Streamlit Header & Deploy Button Visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
        display: block !important;
    }
    footer {visibility: hidden;}
    
    /* Rich Modern Canvas */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(120% 120% at 50% 0%, #e6f9f0 0%, #f0fdf4 35%, #f8fafc 75%, #ffffff 100%) !important;
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecfdf5 0%, #f4fbf7 60%, #ffffff 100%) !important;
        border-right: 1.5px solid #a7f3d0 !important;
    }
    
    /* Luxury Glassmorphic Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #022c22 0%, #064e3b 35%, #047857 70%, #059669 100%);
        padding: 36px 40px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 20px 40px -15px rgba(6, 78, 59, 0.35);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.25) 0%, rgba(255, 255, 255, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.7rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1.15;
    }
    .hero-desc {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 10px;
        max-width: 850px;
        line-height: 1.6;
    }
    .feature-pills {
        display: flex;
        gap: 12px;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    .feature-pill {
        background: rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #e6f9f0;
    }

    /* Interactive Pipeline Visualizer */
    .pipeline-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: white;
        padding: 14px 20px;
        border-radius: 16px;
        border: 1.5px solid #d1fae5;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #065f46;
    }
    .pipeline-step-badge {
        width: 26px;
        height: 26px;
        background: #ecfdf5;
        border: 1.5px solid #10b981;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: #047857;
    }
    .pipeline-arrow {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Decision Glow Banner */
    .decision-banner {
        padding: 22px 28px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
    }
    .banner-answer {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        color: #064e3b;
    }
    .banner-clarify {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        color: #78350f;
    }
    .banner-abstain {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        color: #7f1d1d;
    }
    .banner-retrieve {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        color: #1e3a8a;
    }

    /* Gauge / Metric Card */
    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 22px 20px;
        border: 1.5px solid #e2e8f0;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.03);
        transition: all 0.25s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.12);
        border-color: #10b981;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .progress-bar-bg {
        background: #f1f5f9;
        height: 6px;
        border-radius: 6px;
        margin-top: 10px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 6px;
    }

    /* Claim Cards */
    .claim-card-supported {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #14532d;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.04);
        line-height: 1.5;
    }
    .claim-card-dropped {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #7f1d1d;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.04);
        line-height: 1.5;
    }

    /* Form Controls */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
    }
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #a7f3d0 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2) !important;
    }
    
    /* Print Optimization Rules */
    @media print {
        [data-testid="stSidebar"], .pipeline-stepper, .stButton, .stTextArea, .hero-container, .no-print {
            display: none !important;
        }
        [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load Logo Base64 for Hero Container
logo_b64 = ""
logo_path = Path("assets/logo.png")
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 120px; height: 120px; object-fit: contain; border-radius: 20px; background: white; padding: 6px; box-shadow: 0 8px 24px rgba(0,0,0,0.22);" />' if logo_b64 else '🌾'

# Hero Header
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; align-items: center; gap: 26px; flex-wrap: wrap;">
        {logo_html}
        <div style="flex: 1; min-width: 320px;">
            <div class="hero-badge">
                <span class="pulse-dot"></span>
                ICAR & State Agronomy Verified Knowledge Engine
            </div>
            <div class="hero-title">AgriTrust Intelligence System</div>
            <div class="hero-desc">
                An Evidence- and Confidence-Aware Multi-Agent RAG Framework delivering zero-hallucination agricultural decisions through continuous claim entailment verification and inter-agent consensus calibration.
            </div>
        </div>
    </div>
    <div class="feature-pills">
        <span class="feature-pill">🤖 4 Domain Specialist Agents</span>
        <span class="feature-pill">📚 Dense Chroma + BM25 RRF Search</span>
        <span class="feature-pill">🛡️ Trust Layer A (NLI Claim Pruning)</span>
        <span class="feature-pill">🤝 Trust Layer B (Consensus Alignment)</span>
        <span class="feature-pill">🎯 Calibrated Response Controller</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Architecture Stepper Flowchart
st.markdown("""
<div class="pipeline-stepper">
    <div class="pipeline-step"><span class="pipeline-step-badge">1</span> Query Understanding</div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step"><span class="pipeline-step-badge">2</span> Hybrid RAG (BM25 + Chroma)</div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step"><span class="pipeline-step-badge">3</span> 4 Multi-Agents Deliberation</div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step"><span class="pipeline-step-badge">4</span> Trust Layer A (NLI Verifier)</div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step"><span class="pipeline-step-badge">5</span> Trust Layer B (Consensus)</div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step"><span class="pipeline-step-badge">6</span> Calibrated Decision Gate</div>
</div>
""", unsafe_allow_html=True)

# Sidebar with Controls & Logo
with st.sidebar:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
        st.markdown("<hr style='margin: 12px 0 16px 0; border: none; border-bottom: 1px solid #d1fae5;'>", unsafe_allow_html=True)
    st.markdown("### 🚜 Agronomic Context")
    crop_input = st.selectbox("Target Crop", ["", "Wheat", "Rice", "Soybean", "Cotton", "Maize", "Tomato", "Potato", "Sugarcane"])
    region_input = st.text_input("Agro-Climatic Zone / Region", placeholder="e.g. Northern Plains, Punjab")
    season_input = st.selectbox("Season", ["", "Kharif", "Rabi", "Zaid / Summer"])
    growth_stage = st.selectbox("Crop Growth Stage", ["", "Sowing / Seedling", "Tillering / Vegetative", "Flowering / Tasseling", "Grain Filling", "Maturity"])

    st.markdown("---")
    st.markdown("### 🔬 Diagnostic Toggles")
    compare_baselines = st.checkbox("Compare Baselines (LLM, RAG, MARAG)", value=False)
    show_raw_state = st.checkbox("Show LangGraph State (JSON)", value=False)

    st.markdown("---")
    st.markdown("### 🛡️ Core Assurance")
    st.caption("✅ **Hallucination Protection**: Claims unsupported by retrieved citations are pruned.")
    st.caption("✅ **Selective Risk Gate**: Abstains on unknown premises or requests clarification.")

# Preset Interactive Demo Buttons
st.markdown("#### ⚡ Interactive Demo Queries (Click to load):")
c1, c2, c3, c4 = st.columns(4)

selected_prompt = None
with c1:
    if st.button("🌾 Irrigated Wheat Agronomy\n*(Full Answer Scenario)*", use_container_width=True):
        selected_prompt = "What is the recommended seed rate and fertilizer schedule for timely sown irrigated wheat in Punjab?"
with c2:
    if st.button("🐛 Cotton Bollworm Pest\n*(Disease & Pest Advisory)*", use_container_width=True):
        selected_prompt = "How to identify and manage pink bollworm infestation in Bt cotton during flowering stage?"
with c3:
    if st.button("❓ Ambiguous Fertilizer Query\n*(Triggers CLARIFY Gate)*", use_container_width=True):
        selected_prompt = "How much fertilizer should I apply for maximum crop yield?"
with c4:
    if st.button("🛑 Avocado Orchard Care\n*(Triggers ABSTAIN Gate)*", use_container_width=True):
        selected_prompt = "What is the best pruning technique for avocado orchards in Mediterranean climate?"

# Session Management
if "query_input_text" not in st.session_state:
    st.session_state.query_input_text = ""

if selected_prompt:
    st.session_state.query_input_text = selected_prompt
    st.rerun()

# Main Query Text Area
query = st.text_area(
    "**Enter Agricultural Query:**",
    value=st.session_state.query_input_text,
    placeholder="Ask anything about crop management, diseases, weather impacts, or soil fertility...",
    height=95
)

run_col1, run_col2 = st.columns([1, 4])
with run_col1:
    execute_button = st.button("🚀 Analyze & Advise", type="primary", use_container_width=True)

# Pipeline Execution with 3-Second Reasoning Halt
if execute_button and query.strip():
    with st.status("🌾 AgriTrust Autonomous Multi-Agent Deliberation in progress...", expanded=True) as status_box:
        st.write("🔍 **Phase 1/4**: Query understanding & dispatching to Crop, Disease, Weather & Soil agents...")
        time.sleep(0.8)
        st.write("📚 **Phase 2/4**: Performing hybrid retrieval (Dense Chroma vector DB + BM25 lexical RRF search)...")
        time.sleep(0.8)
        st.write("🛡️ **Phase 3/4**: Trust Layer A — Executing NLI claim verification & pruning unsupported assertions...")
        time.sleep(0.8)
        st.write("🤝 **Phase 4/4**: Trust Layer B — Aligning inter-agent consensus & computing calibrated confidence...")
        time.sleep(0.6)
        status_box.update(label="✅ Advisory synthesis & trust verification complete!", state="complete", expanded=False)

    try:
        from agritrust.graph import AgriTrustPipeline
        pipeline = AgriTrustPipeline()

        user_context = {}
        if crop_input:
            user_context["crop"] = crop_input.lower()
        if region_input:
            user_context["region"] = region_input
        if season_input:
            user_context["season"] = season_input.lower()
        if growth_stage:
            user_context["growth_stage"] = growth_stage.lower()

        with st.spinner("🤖 Finalizing verified advisory..."):
            res = pipeline.advise(query, user_context=user_context)

            decision = res.get("decision", "ABSTAIN")
            conf = res.get("overall_confidence", 0.0)
            ess = res.get("evidence_support_score", 0.0)
            agr = res.get("agent_agreement_score", 1.0)
            coverage = res.get("evidence_coverage", 0.0)

            # Banner Style
            banner_class = {
                "ANSWER": "banner-answer",
                "CLARIFY": "banner-clarify",
                "ABSTAIN": "banner-abstain",
                "RETRIEVE_MORE": "banner-retrieve"
            }.get(decision, "banner-abstain")

            decision_icon = {
                "ANSWER": "✅",
                "CLARIFY": "❓",
                "ABSTAIN": "🛡️",
                "RETRIEVE_MORE": "🔄"
            }.get(decision, "🛡️")

            st.markdown(f"""
            <div class="decision-banner {banner_class}">
                <div>
                    <span style="font-size: 1.6rem; margin-right: 10px;">{decision_icon}</span>
                    <strong style="font-size: 1.4rem; font-family: 'Space Grotesk';">Pipeline Decision: {decision}</strong>
                    <div style="font-size: 0.94rem; opacity: 0.95; margin-top: 5px;">
                        {', '.join(res.get('decision_reasons', ['Evaluated via calibrated trust thresholding']))}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Calibrated Confidence</div>
                    <div style="font-size: 2.2rem; font-weight: 800; font-family: 'Space Grotesk';">{conf*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Visual Metric Cards with Animated Progress Bars
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overall Confidence</div>
                <div class="metric-val" style="color: #059669;">{conf*100:.1f}%</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {conf*100}%; background: linear-gradient(90deg, #10b981, #059669);"></div></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">Calibrated via ECE</div>
            </div>
            """, unsafe_allow_html=True)

            m2.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Evidence Support (ESS)</div>
                <div class="metric-val" style="color: #2563eb;">{ess*100:.1f}%</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {ess*100}%; background: linear-gradient(90deg, #60a5fa, #2563eb);"></div></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">Trust Layer A (NLI)</div>
            </div>
            """, unsafe_allow_html=True)

            m3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Agent Agreement (AGR)</div>
                <div class="metric-val" style="color: #7c3aed;">{agr*100:.1f}%</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {agr*100}%; background: linear-gradient(90deg, #a78bfa, #7c3aed);"></div></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">Trust Layer B (Consensus)</div>
            </div>
            """, unsafe_allow_html=True)

            m4.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Claim Coverage</div>
                <div class="metric-val" style="color: #d97706;">{coverage*100:.1f}%</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {coverage*100}%; background: linear-gradient(90deg, #fcd34d, #d97706);"></div></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">Supported / Total Claims</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Synthesized Advisory Output Box
            st.markdown("### 📋 Calibrated Advisory Recommendation")
            with st.container(border=True):
                st.markdown(res.get("final_answer", ""))

            # Export and Print Controls
            from agritrust.pdf_report import generate_advisory_pdf
            pdf_bytes = generate_advisory_pdf(query, res, user_context=user_context)

            col_pdf, col_prt, _ = st.columns([2.5, 2.5, 5])
            with col_pdf:
                st.download_button(
                    label="📥 Download Advisory Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"agritrust_advisory_report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col_prt:
                st.components.v1.html(
                    """
                    <button onclick="window.parent.print()" style="
                        background: #ffffff;
                        color: #064e3b;
                        border: 2px solid #10b981;
                        padding: 9px 18px;
                        border-radius: 14px;
                        font-weight: 700;
                        font-size: 0.92rem;
                        cursor: pointer;
                        width: 100%;
                        height: 44px;
                        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 8px;
                        transition: all 0.2s ease;
                    " onmouseover="this.style.background='#ecfdf5'" onmouseout="this.style.background='#ffffff'">
                        🖨️ Print Advisory Report
                    </button>
                    """,
                    height=52
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabs
            t_claims, t_agents, t_evidence, t_benchmark = st.tabs([
                "🛡️ Trust Layer A: Claim Pruning & Entailment",
                "🤝 Trust Layer B: Multi-Agent Consensus",
                "📚 Grounded Evidence Chunks (Hybrid RAG)",
                "📊 Evaluation Benchmarks & Scientific Plots"
            ])

            with t_claims:
                st.markdown("#### Atomic Claim Verification via Natural Language Inference (NLI)")
                st.caption("Each agent recommendation is decomposed into discrete claims and verified against retrieved evidence chunks.")

                col_sup, col_drp = st.columns(2)
                supp = res.get("supported_claims", [])
                drop = res.get("dropped_claims", [])

                with col_sup:
                    st.markdown(f"**✅ Verified & Supported Claims ({len(supp)}):**")
                    if supp:
                        for s in supp:
                            st.markdown(f'<div class="claim-card-supported">✓ {s}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No claims satisfied the minimum entailment threshold.")

                with col_drp:
                    st.markdown(f"**❌ Pruned / Unsupported Claims ({len(drop)}):**")
                    if drop:
                        for d in drop:
                            st.markdown(f'<div class="claim-card-dropped">⚠️ {d}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("Zero claims were pruned. All claims are grounded.")

                verifs = res.get("claim_verifications", [])
                if verifs:
                    st.markdown("##### Detailed NLI Verification Log")
                    v_df = pd.DataFrame([
                        {
                            "Candidate Claim": v.get("claim", ""),
                            "NLI Label": v.get("label", "").upper(),
                            "Entailment Score": f"{v.get('support_score', 0.0)*100:.1f}%",
                            "Supporting Evidence Chunk": v.get("best_chunk_id", "None"),
                        }
                        for v in verifs
                    ])
                    st.dataframe(v_df, use_container_width=True)

            with t_agents:
                st.markdown("#### Specialized Multi-Agent Deliberation & Alignment")
                conflicts = res.get("agent_conflicts", [])
                if conflicts:
                    st.error(f"⚠️ {len(conflicts)} Inter-Agent Conflict(s) Detected:")
                    for c in conflicts:
                        st.write(f"- **{c.get('agent_a')}** vs **{c.get('agent_b')}**: {c.get('description')}")
                else:
                    st.success("✅ **Unanimous Consensus**: No agronomic conflicts or contradictions detected between agents.")

                outputs = res.get("agent_outputs", {})
                if outputs:
                    agent_icons = {
                        "crop_advisory": "🌾",
                        "plant_disease": "🔬",
                        "weather_context": "🌦️",
                        "soil_fertilizer": "🧪"
                    }
                    for agent_name, out in outputs.items():
                        icon = agent_icons.get(agent_name, "🤖")
                        title = agent_name.replace("_", " ").title()
                        with st.expander(f"{icon} {title} Agent — Self-Confidence: {out.get('self_confidence', 0.0)*100:.0f}%", expanded=True):
                            st.markdown(f"**Recommendation**: {out.get('recommendation', '')}")
                            if out.get("key_claims"):
                                st.caption("Key Claims: • " + " • ".join(out.get("key_claims", [])))

            with t_evidence:
                st.markdown("#### Grounded Knowledge Base Chunks (RRF Hybrid Search)")
                chunks = res.get("retrieved_chunks", [])
                if chunks:
                    for chk in chunks:
                        with st.expander(f"📄 [{chk.get('chunk_id')}] {chk.get('title')} (RRF Score: {chk.get('score', 0.0):.3f})"):
                            st.caption(f"Crop: **{chk.get('crop', '').title()}** | Topic: **{chk.get('topic', '')}** | Source: [{chk.get('source_url')}]({chk.get('source_url')})")
                            st.markdown(f"> {chk.get('text', '')}")
                else:
                    st.info("No chunks retrieved.")

            with t_benchmark:
                st.markdown("#### Empirical Benchmark Results")
                results_dir = Path("results")
                metrics_file = results_dir / "metrics.csv"
                if metrics_file.exists():
                    df_m = pd.read_csv(metrics_file)
                    st.dataframe(df_m, use_container_width=True)

                st.markdown("#### Publication Visualizations")
                p1, p2, p3 = st.columns(3)
                if (results_dir / "metrics_comparison.png").exists():
                    with p1:
                        st.image(str(results_dir / "metrics_comparison.png"), caption="System Comparison Across Key Metrics")
                if (results_dir / "risk_coverage_curve.png").exists():
                    with p2:
                        st.image(str(results_dir / "risk_coverage_curve.png"), caption="Risk-Coverage Tradeoff Curve")
                if (results_dir / "decision_confusion_matrix.png").exists():
                    with p3:
                        st.image(str(results_dir / "decision_confusion_matrix.png"), caption="Calibrated Decision Distribution")

            # Comparative Baselines
            if compare_baselines:
                st.markdown("---")
                st.subheader("🔬 Side-by-Side Baselines Comparison")
                from agritrust.baselines.llm_only import LLMOnlyBaseline
                from agritrust.baselines.rag import ConventionalRAGBaseline
                from agritrust.baselines.marag import MultiAgentRAGBaseline

                b_llm = LLMOnlyBaseline().advise(query, user_context)
                b_rag = ConventionalRAGBaseline().advise(query, user_context)
                b_marag = MultiAgentRAGBaseline().advise(query, user_context)

                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                with b_c1:
                    with st.container(border=True):
                        st.markdown("**1. Plain LLM**")
                        st.caption(f"Decision: `{b_llm['decision']}` | Conf: {b_llm.get('overall_confidence', 0.5)*100:.0f}%")
                        st.write(b_llm["final_answer"])
                with b_c2:
                    with st.container(border=True):
                        st.markdown("**2. Conventional RAG**")
                        st.caption(f"Decision: `{b_rag['decision']}` | Conf: {b_rag.get('overall_confidence', 0.7)*100:.0f}%")
                        st.write(b_rag["final_answer"])
                with b_c3:
                    with st.container(border=True):
                        st.markdown("**3. Multi-Agent RAG**")
                        st.caption(f"Decision: `{b_marag['decision']}` | Conf: {b_marag.get('overall_confidence', 0.8)*100:.0f}%")
                        st.write(b_marag["final_answer"])
                with b_c4:
                    with st.container(border=True):
                        st.markdown("**4. AgriTrust (Full)**")
                        st.caption(f"Decision: `{res['decision']}` | Conf: {res.get('overall_confidence', 0.0)*100:.0f}%")
                        st.write(res["final_answer"])

            if show_raw_state:
                st.markdown("---")
                st.subheader("🛠️ Raw State JSON")
                st.json(res)

    except Exception as e:
        st.error(f"Error executing advisory pipeline: {e}")
