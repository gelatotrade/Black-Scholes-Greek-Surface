#!/usr/bin/env python3
"""
Black-Scholes Greek Surface - Streamlit Dashboard

An interactive web application for exploring Black-Scholes Greeks.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.black_scholes import BlackScholes
from src.visualizations import GreekSurfaceVisualizer


# Page configuration
st.set_page_config(
    page_title="Black-Scholes Greek Surfaces",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888888;
        text-align: center;
        margin-top: 0;
    }
    .formula {
        font-size: 1.1rem;
        color: #ff69b4;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Black-Scholes Greek Surfaces</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive 3D Visualization of Option Greeks</p>', unsafe_allow_html=True)
st.markdown('<p class="formula">Θ + ½σ²S²Γ + rSΔ = rV</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Parameters
st.sidebar.header("📊 Option Parameters")

option_type = st.sidebar.selectbox(
    "Option Type",
    ["Call", "Put"],
    index=0
).lower()

col1, col2 = st.sidebar.columns(2)
with col1:
    S = st.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0)
with col2:
    K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0)

col1, col2 = st.sidebar.columns(2)
with col1:
    T = st.slider("Time to Expiry (Years)", 0.01, 3.0, 0.5, 0.01)
with col2:
    sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.2, 0.01)

r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, 0.005)
q = st.sidebar.slider("Dividend Yield (q)", 0.0, 0.1, 0.0, 0.005)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization")

greek_options = {
    "Delta (Δ)": "delta",
    "Gamma (Γ)": "gamma",
    "Theta (Θ)": "theta",
    "Vega (ν)": "vega",
    "Rho (ρ)": "rho",
    "Option Price": "price"
}
selected_greek = st.sidebar.selectbox(
    "Greek to Display",
    list(greek_options.keys()),
    index=0
)
greek = greek_options[selected_greek]

show_atm = st.sidebar.checkbox("Show ATM Line", value=True)
show_peak = st.sidebar.checkbox("Show Peak Zone", value=True)
show_position = st.sidebar.checkbox("Show Current Position", value=True)

# Calculate current values
bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

# Main layout
tab1, tab2, tab3 = st.tabs(["📈 3D Surface", "📊 Greeks Analysis", "📚 Formulas"])

with tab1:
    # Create visualization
    viz = GreekSurfaceVisualizer(K=K, r=r, sigma=sigma, q=q)

    fig = viz.plot_greek_surface(
        greek=greek,
        option_type=option_type,
        current_S=S if show_position else None,
        current_T=T if show_position else None,
        show_atm_line=show_atm,
        show_peak=show_peak
    )

    # Adjust layout for Streamlit
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"{option_type.capitalize()} Option Analysis")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Option Price",
            f"${bs.price(option_type):.4f}",
            delta=None
        )

    with col2:
        st.metric(
            "Moneyness (S/K)",
            f"{S/K:.4f}",
            delta="ITM" if (S > K and option_type == 'call') or (S < K and option_type == 'put') else "OTM"
        )

    with col3:
        st.metric(
            "Days to Expiry",
            f"{T * 365:.0f}",
            delta=None
        )

    with col4:
        iv_equiv = sigma * 100
        st.metric(
            "Implied Vol",
            f"{iv_equiv:.1f}%",
            delta=None
        )

    st.markdown("---")

    # Greeks table
    st.subheader("First-Order Greeks")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### Δ Delta")
        delta_val = bs.delta(option_type)
        st.markdown(f"**{delta_val:.6f}**")
        st.caption("Price sensitivity")

    with col2:
        st.markdown("### Γ Gamma")
        gamma_val = bs.gamma()
        st.markdown(f"**{gamma_val:.6f}**")
        st.caption("Delta sensitivity")

    with col3:
        st.markdown("### Θ Theta")
        theta_val = bs.theta_per_day(option_type)
        st.markdown(f"**{theta_val:.6f}**")
        st.caption("Per day decay")

    with col4:
        st.markdown("### ν Vega")
        vega_val = bs.vega_pct()
        st.markdown(f"**{vega_val:.6f}**")
        st.caption("Per 1% vol change")

    with col5:
        st.markdown("### ρ Rho")
        rho_val = bs.rho_pct(option_type)
        st.markdown(f"**{rho_val:.6f}**")
        st.caption("Per 1% rate change")

    st.markdown("---")

    # Second-order Greeks
    st.subheader("Second-Order Greeks")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Vanna**")
        st.write(f"{bs.vanna():.6f}")

    with col2:
        st.markdown("**Volga**")
        st.write(f"{bs.volga():.6f}")

    with col3:
        st.markdown("**Charm**")
        st.write(f"{bs.charm(option_type):.6f}")

    with col4:
        st.markdown("**Speed**")
        st.write(f"{bs.speed():.6f}")

with tab3:
    st.subheader("Black-Scholes Formulas")

    st.markdown("""
    ### Option Pricing

    **Call Option:**
    $$C = Se^{-qτ}Φ(d_1) - Ke^{-rτ}Φ(d_2)$$

    **Put Option:**
    $$P = Ke^{-rτ}Φ(-d_2) - Se^{-qτ}Φ(-d_1)$$

    ### Key Parameters

    $$d_1 = \\frac{\\ln(S/K) + (r - q + \\frac{1}{2}σ^2)τ}{σ\\sqrt{τ}}$$

    $$d_2 = d_1 - σ\\sqrt{τ}$$

    ### The Greeks

    | Greek | Formula | Description |
    |-------|---------|-------------|
    | Delta (Δ) | $Φ(d_1)$ for calls | Price sensitivity |
    | Gamma (Γ) | $\\frac{φ(d_1)}{Sσ\\sqrt{τ}}$ | Delta sensitivity |
    | Theta (Θ) | $-\\frac{Sσφ(d_1)}{2\\sqrt{τ}} - rKe^{-rτ}Φ(d_2)$ | Time decay |
    | Vega (ν) | $S\\sqrt{τ}φ(d_1)$ | Volatility sensitivity |
    | Rho (ρ) | $Kτe^{-rτ}Φ(d_2)$ for calls | Interest rate sensitivity |

    ### Black-Scholes PDE

    $$\\boxed{Θ + \\frac{1}{2}σ^2S^2Γ + rSΔ = rV}$$

    This fundamental equation connects all the major Greeks and forms the basis
    for delta hedging and risk-neutral option pricing.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Black-Scholes Greek Surface Visualizer | "
    "Built with Streamlit and Plotly"
    "</p>",
    unsafe_allow_html=True
)
