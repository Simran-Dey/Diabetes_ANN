import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="GlucoGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME TOGGLE --------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Toggle buttons in sidebar
st.sidebar.markdown("### 🌗 Theme Mode")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("☀️ Light", use_container_width=True,
                 type="primary" if not st.session_state.dark_mode else "secondary"):
        st.session_state.dark_mode = False
        st.rerun()

with col2:
    if st.button("🌙 Dark", use_container_width=True,
                 type="primary" if st.session_state.dark_mode else "secondary"):
        st.session_state.dark_mode = True
        st.rerun()

# Status message
if st.session_state.dark_mode:
    st.sidebar.success("🌙 Dark mode activated")
else:
    st.sidebar.info("☀️ Light mode activated")

st.sidebar.caption("Also try native switch: ⋮ → Settings → Theme")

# -------------------- LIGHT BLUE BUTTONS CSS --------------------
blue_normal = "#42a5f5"     # Lighter blue (main)
blue_hover  = "#2196f3"     # Mid-tone for hover
blue_active = "#1976d2"     # Darker/original for active/pressed

# Text colors — white works great on these lighter blues in both modes
text_color = "#ffffff"      # Pure white — excellent contrast

st.markdown(f"""
<style>
    /* Force ALL buttons to use the new light blue family */
    button {{
        background-color: {blue_normal} !important;
        color: {text_color} !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }}

    button:hover {{
        background-color: {blue_hover} !important;
        color: {text_color} !important;
    }}

    button:active,
    button:focus {{
        background-color: {blue_active} !important;
        color: {text_color} !important;
        box-shadow: 0 0 0 3px rgba(66, 165, 245, 0.4) !important;  /* lighter blue shadow */
    }}

    /* Target Streamlit button kinds explicitly */
    button[kind="primary"],
    button[kind="secondary"],
    button[kind="tertiary"] {{
        background-color: {blue_normal} !important;
        color: {text_color} !important;
    }}

    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    button[kind="tertiary"]:hover {{
        background-color: {blue_hover} !important;
        color: {text_color} !important;
    }}

    /* Make primary buttons (like Analyze Risk) slightly more prominent */
    button[kind="primary"] {{
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}

    /* Basic dark/light mode background & text */
    .stApp {{
        background-color: {'#121212' if st.session_state.dark_mode else '#ffffff'} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {'#0d1a0d' if st.session_state.dark_mode else '#f5f9f2'} !important;
    }}
    .stMarkdown, h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: {'#e8f5e9' if st.session_state.dark_mode else '#1a3c34'} !important;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD LOGO --------------------
logo = Image.open("logo.png")

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=120)

with col2:
    st.title("GlucoGuard AI 🛡️")
    st.markdown("### Intelligent Diabetes Risk Prediction System")
    st.markdown("Deep Learning Powered | Accurate | Reliable")

st.markdown("---")

# -------------------- LOAD MODEL + SCALER --------------------
@st.cache_resource
def load_resources():
    model = load_model("diabetes_ann_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# -------------------- INPUT SECTION --------------------
st.sidebar.header("🧾 Enter Patient Details")

pregnancies    = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose        = st.sidebar.slider("Glucose (mg/dL)", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin        = st.sidebar.slider("Insulin", 0, 900, 80)
bmi            = st.sidebar.slider("BMI", 0.0, 60.0, 25.0, step=0.1)
dpf            = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
age            = st.sidebar.slider("Age", 10, 100, 30)

input_data = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])

input_data_scaled = scaler.transform(input_data)

# -------------------- PREDICTION --------------------
if st.button("🔍 Analyze Risk", type="primary"):

    probability = model.predict(input_data_scaled)[0][0]
    prediction = 1 if probability > 0.5 else 0
    risk_pct = float(probability) * 100

    st.subheader("📊 Risk Assessment Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes ({risk_pct:.2f}%)")
        level = "High Risk"
    else:
        st.success(f"✅ Low Risk of Diabetes ({risk_pct:.2f}%)")
        level = "Low Risk"

    st.markdown("### 🔍 Key Risk Contributors")
    risk_factors = []
    if glucose >= 126: risk_factors.append("High Glucose")
    if bmi >= 30:      risk_factors.append("High BMI")
    if age > 45:       risk_factors.append("Higher Age")
    if dpf > 1.0:      risk_factors.append("Strong Family History")

    if risk_factors:
        st.write(", ".join(risk_factors))
    else:
        st.write("No major high-risk indicators detected.")

    st.markdown("### 📈 Risk Gauge")

    # Risk-dependent bar color (red-orange-green)
    gauge_color = "#4caf50" if risk_pct < 50 else "#ff9800" if risk_pct < 75 else "#f44336"

    # Theme-dependent gauge background
    gauge_bgcolor = "#1e1e1e" if st.session_state.dark_mode else "#ffffff"
    gauge_border  = "#333333" if st.session_state.dark_mode else "#e0e0e0"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        title={'text': "Diabetes Risk (%)"},
        number={'font': {'size': 40}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#ffffff" if st.session_state.dark_mode else "#000000"
            },
            'bar': {'color': gauge_color},
            'bgcolor': gauge_bgcolor,
            'bordercolor': gauge_border,
            'borderwidth': 1,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    # Better contrast for dark mode
    if st.session_state.dark_mode:
        fig_gauge.update_layout(
            paper_bgcolor="#121212",
            font_color="#e8f5e9"
        )
    else:
        fig_gauge.update_layout(
            paper_bgcolor="#ffffff",
            font_color="#1a1a1a"
        )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### 📄 Download Detailed Report")

    report_text = f"""
===============================
        GLUCOGUARD AI REPORT
===============================

Patient Details:
----------------
Pregnancies              : {pregnancies}
Glucose (mg/dL)          : {glucose}
Blood Pressure           : {blood_pressure}
Skin Thickness           : {skin_thickness}
Insulin                  : {insulin}
BMI                      : {bmi}
Diabetes Pedigree Func   : {dpf}
Age                      : {age}

--------------------------------
Risk Assessment Result:
--------------------------------
Risk Probability         : {risk_pct:.2f}%
Risk Level               : {level}

--------------------------------
Generated by GlucoGuard AI
Deep Learning Based Prediction
===============================
"""

    st.download_button(
        label="📥 Download Report (TXT)",
        data=report_text,
        file_name="glucoguard_ai_report.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("© 2026 GlucoGuard AI | Built with TensorFlow & Streamlit")