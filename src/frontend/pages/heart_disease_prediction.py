import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import requests
import streamlit as st

from src.frontend.config.fr_settings import Settings

settings = Settings()
API_URL = settings.api_url

st.set_page_config(
    page_title= "DR. ML- Heart Disease Risk Prediction",
    page_icon= "❤️",
    layout= "centered"
)

st.title("Heart Disease Risk Predictor")
st.write("Enter Patient Details and click **Predict**")

st.subheader("Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age",1,120,52)
    sex = st.selectbox("Sex (1-Male, 0-Female)",[0,1])
    cp = st.number_input("Chest Pain Type (cp)",0,3,0)
    trestbps = st.number_input("Resting Blood Pressure",0,250,125)
    chol = st.number_input("Cholesterol",0,600,212)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
    restecg = st.number_input("Resting ECG (restecg)",0,2,1)
    thalach = st.number_input("Max Heart Rate (thalach)",0, 250,168)
    exang = st.selectbox("Exercise Induced Angina (exang)",[0,1])

with col3:
    oldpeak = st.number_input("Oldpeak (ST Depression)",0.0,10.0,1.0)
    slope = st.number_input("Slope",0,2,2)
    ca = st.number_input("Number of Major vessels (ca)",0,4,0)
    thal = st.selectbox("Thal",[0,1])

if st.button("🔍Predict", use_container_width=True):
    payload = {
        "disease":"heart_disease",
        "features": {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs":fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal

            
        },
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error("Unable To Reach Prediction Service")
        st.caption(str(e))
        st.stop()
    
    result = response.json()
    prediction = int(result["prediction"])
    probability = float(result["probability"])

    st.divider()
    st.metric("Heart Disease Probability", f"{probability:.2f}")

    if prediction == 1:
        st.error("🚩Model Prediction: Heart Disease Present")
    else:
        st.success("🟢Model Prediction: No Heart Disease Detected")

