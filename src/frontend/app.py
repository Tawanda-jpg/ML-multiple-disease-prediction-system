import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="DR.ML-Multi-Disease Predictor",
    page_icon= "📈",
    layout= "centered"
)

st.title("🧠Dr. ML-Multi-Disease Predictor")

st.write(
    """
Use the left side to navigate:\n
_ 📈 Diabetes Risk Predictor\n
_ ❤️ Heart Disease Risk Predictor
"""
)

st.info("Make sure the FastAPI Backend is running before making predictions")
         
         
         
         