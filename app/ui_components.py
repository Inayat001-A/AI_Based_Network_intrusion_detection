"""
UI Components, Custom Cyber-Dark Glassmorphic Styling, and Plotly Visualizations for Streamlit SOC Dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any


def inject_custom_css():
    """Injects futuristic cybersecurity styling, glassmorphism, and neon glowing accents."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Dark Theme Settings */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1322 0%, #070a12 90%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Futuristic Header Banner */
    .soc-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.65) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .soc-title {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #00FF9D 0%, #00E5FF 50%, #7928CA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .soc-subtitle {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Glowing Pulsing Status Beacon */
    .pulse-beacon {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #00FF9D;
        box-shadow: 0 0 12px #00FF9D;
        animation: pulse-animation 2s infinite;
        margin-right: 8px;
    }

    @keyframes pulse-animation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 157, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 255, 157, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 157, 0); }
    }

    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.3);
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0 0 0;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Severity Badges */
    .badge-critical {
        background: rgba(255, 0, 85, 0.15);
        color: #FF0055;
        border: 1px solid rgba(255, 0, 85, 0.3);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-high {
        background: rgba(255, 136, 0, 0.15);
        color: #FF8800;
        border: 1px solid rgba(255, 136, 0, 0.3);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-medium {
        background: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-info {
        background: rgba(0, 255, 157, 0.15);
        color: #00FF9D;
        border: 1px solid rgba(0, 255, 157, 0.3);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Renders the top SOC operations banner."""
    st.markdown("""
    <div class="soc-header">
        <div>
            <h1 class="soc-title">🛡️ AI-NIDS // SECURITY OPERATIONS CENTER</h1>
            <div class="soc-subtitle">REAL-TIME INTRUSION DETECTION, ZERO-DAY RADAR & WIRE-SPEED AI TELEMETRY</div>
        </div>
        <div style="text-align: right;">
            <div style="display: flex; align-items: center; justify-content: flex-end;">
                <span class="pulse-beacon"></span>
                <span style="font-weight: 700; color: #00FF9D; font-size: 13px; font-family: 'JetBrains Mono', monospace;">LIVE DEFENSE: ACTIVE</span>
            </div>
            <div style="color: #64748b; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">ENGINE: XGBOOST + DEEP AUTOENCODER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#00E5FF"):
    """Renders a modern glassmorphic KPI tile."""
    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="color: #64748b; font-size: 12px; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def create_attack_gauge(attack_rate_pct: float) -> go.Figure:
    """Creates a futuristic cyberpunk radial gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=attack_rate_pct,
        number={"suffix": "%", "font": {"color": "#f8fafc", "family": "Outfit", "size": 32}},
        title={"text": "THREAT RATIO GAUGE", "font": {"color": "#94a3b8", "size": 12, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickwidth": 1},
            "bar": {"color": "#FF0055" if attack_rate_pct > 25 else "#00FF9D", "thickness": 0.3},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "borderwidth": 1,
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "steps": [
                {"range": [0, 15], "color": "rgba(0, 255, 157, 0.15)"},
                {"range": [15, 40], "color": "rgba(255, 136, 0, 0.15)"},
                {"range": [40, 100], "color": "rgba(255, 0, 85, 0.25)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=15),
        height=210
    )
    return fig


def create_threat_donut(threat_counts: Dict[str, int]) -> go.Figure:
    """Creates a glowing dark-themed threat category donut chart."""
    labels = list(threat_counts.keys())
    values = list(threat_counts.values())
    
    colors = {
        "BENIGN": "#00FF9D",
        "DoS Hulk": "#FF0055",
        "PortScan": "#00E5FF",
        "SSH-Patator": "#FF8800",
        "Bot": "#7928CA",
        "Zero-Day Anomaly": "#E000FF"
    }
    color_seq = [colors.get(k.split()[0], "#38bdf8") for k in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        marker=dict(colors=color_seq, line=dict(color="#0B0F19", width=2)),
        textinfo="percent",
        hoverinfo="label+value+percent",
        textfont=dict(color="#ffffff", family="Inter")
    )])
    
    fig.update_layout(
        title=dict(text="ATTACK TAXONOMY BREAKDOWN", font=dict(color="#94a3b8", size=12, family="JetBrains Mono")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0", size=11, family="Inter"), orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=35, b=25),
        height=240
    )
    return fig


def create_port_target_chart(port_counts: Dict[int, int]) -> go.Figure:
    """Creates a targeted network ports bar chart."""
    ports = [f"Port {p}" for p in port_counts.keys()]
    counts = list(port_counts.values())
    
    fig = go.Figure(go.Bar(
        x=counts,
        y=ports,
        orientation="h",
        marker=dict(
            color=counts,
            colorscale=[[0, "#00E5FF"], [1, "#7928CA"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        )
    ))
    fig.update_layout(
        title=dict(text="TARGETED DESTINATION PORTS", font=dict(color="#94a3b8", size=12, family="JetBrains Mono")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Flow Count", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#e2e8f0")),
        margin=dict(l=10, r=10, t=35, b=20),
        height=230
    )
    return fig


def create_loss_histogram(loss_array: np.ndarray, threshold: float) -> go.Figure:
    """Creates an Autoencoder reconstruction error distribution plot."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=loss_array,
        nbinsx=40,
        marker=dict(color="rgba(0, 229, 255, 0.6)", line=dict(color="#00E5FF", width=1)),
        name="Flow Loss (MSE)"
    ))
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#FF0055",
        annotation_text=f"Zero-Day Threshold: {threshold:.4f}",
        annotation_font=dict(color="#FF0055", family="JetBrains Mono")
    )
    fig.update_layout(
        title=dict(text="AUTOENCODER RECONSTRUCTION ERROR (ZERO-DAY RADAR)", font=dict(color="#94a3b8", size=12, family="JetBrains Mono")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Reconstruction MSE Loss", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Frequency", tickfont=dict(color="#94a3b8")),
        margin=dict(l=10, r=10, t=35, b=20),
        height=250
    )
    return fig
