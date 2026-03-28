
import streamlit as st
import pandas as pd
import joblib
import requests
import os
import tempfile
import numpy as np

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Predictive Maintenance – Engine Condition Classifier")
st.markdown("""
This app predicts whether an engine requires **maintenance** or is **operating normally**
based on real-time sensor readings.
Model: **XGBoost Classifier** trained on engine sensor data from Hugging Face Hub.
""")

@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/Divyap90/predictivemaint/resolve/main/best_xgboost_model.joblib"
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name
    model = joblib.load(tmp_path)
    os.remove(tmp_path)
    return model

with st.spinner("Loading model from Hugging Face Hub..."):
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

FEATURE_RANGES = {
    "Engine RPM":          (624.79,  2239.74, 1250.0,   "rpm"),
    "Lub Oil Pressure":    (0.003,   7.265,   3.5,      "kPa"),
    "Fuel Pressure":       (0.002,   21.138,  10.0,     "kPa"),
    "Coolant Pressure":    (0.002,   7.478,   3.5,      "kPa"),
    "Lub Oil Temperature": (71.0,    89.0,    80.0,     "C"),
    "Coolant Temperature": (61.673,  195.527, 120.0,    "C"),
}

FEATURE_COLS = list(FEATURE_RANGES.keys())

st.sidebar.header("Sensor Input Values")
user_inputs = {}
for feat, (lo, hi, default, unit) in FEATURE_RANGES.items():
    user_inputs[feat] = st.sidebar.slider(
        f"{feat} ({unit})",
        min_value=float(lo), max_value=float(hi),
        value=float(default), step=round((hi - lo) / 200, 4),
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Sensor Readings")
    input_df = pd.DataFrame([user_inputs])
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    if st.button("Predict Engine Condition", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        with col2:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("### ENGINE REQUIRES MAINTENANCE")
                st.metric("Failure Probability", f"{probability[1]*100:.1f}%")
                st.markdown("""
**Recommended Actions:**
- Schedule immediate inspection
- Check oil pressure and temperature systems
- Verify fuel delivery components
- Inspect coolant circulation
                """)
            else:
                st.success("### ENGINE OPERATING NORMALLY")
                st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
                st.markdown("All sensor readings within normal operating range.")

            prob_df = pd.DataFrame({
                "Condition": ["Healthy", "Failing"],
                "Probability": [probability[0], probability[1]],
            })
            st.bar_chart(prob_df.set_index("Condition"))

st.markdown("---")
st.subheader("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload sensor data CSV", type=["csv"])
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURE_COLS if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(batch_df[FEATURE_COLS])
            probs = model.predict_proba(batch_df[FEATURE_COLS])
            batch_df["Prediction"] = ["Failing" if p == 1 else "Healthy" for p in preds]
            batch_df["Confidence (%)"] = (np.max(probs, axis=1) * 100).round(1)
            st.dataframe(batch_df, use_container_width=True)
            st.metric("Fleet Failure Rate", f"{(preds==1).mean()*100:.1f}%")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Predictive Maintenance System | XGBoost | Divyap90 | Hugging Face Spaces")
