import streamlit as st

def show_header():
    st.title("Pakistan Revenue Forecasting System (PRFS)")
    st.markdown("### A Structural Macro-Fiscal Forecasting Platform")
    st.markdown("---")

def show_executive_summary(results=None):
    with st.container():
        st.markdown(
            """
            <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;border-left:5px solid #2c3e50;">
                <h2 style="margin-top:0;color:#2c3e50;">Executive Summary</h2>
                <p>This platform provides structural econometric forecasts for Pakistan's federal tax revenue. 
                Unlike simple buoyancy models, the PRFS accounts for sub-sectoral transmission channels (LSM, Imports, Consumption) 
                and dynamic adjustment paths.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

def show_methodology():
    with st.expander("Definitions & Methodology"):
        st.markdown("""
        ### Econometric Framework
        The system utilizes a Two-Stage Least Squares (2SLS) inspired recursive architecture:
        1. **Stage 1 (Transmission Channels):** Estimates the relationship between global macro drivers (Nominal GDP, Exchange Rate) 
           and sectoral bases (LSM, Imports, Consumption).
        2. **Stage 2 (Tax Revenue):** Regresses major tax heads against their respective sectoral bases.
        
        ### Elasticities vs Semi-Elasticities
        * **Elasticities:** Variables like GDP, LSM, and Imports are log-transformed. The coefficients represent 
          percentage changes in revenue for a 1% change in the base.
        * **Semi-Elasticities:** Level variables like Inflation and Policy Rate represent the percentage change 
          in revenue for a 1 unit (percentage point) change in the variable.
        """)
