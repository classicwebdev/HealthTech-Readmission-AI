import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="HealthTech AI Predictor", layout="centered")


# --- 1. ROBUST ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load Model
    model = xgb.XGBClassifier()
    model.load_model('hospital_readmission_model.json')

    # Load Feature Names from our JSON file (The Fail-Safe)
    with open('feature_names.json', 'r') as f:
        features = json.load(f)

    return model, features


try:
    model, expected_features = load_assets()
except FileNotFoundError:
    st.error("Missing files! Please run main.py first to generate model and feature names.")
    st.stop()

# --- 2. USER INTERFACE ---
st.title("🏥 Patient Readmission Risk Predictor")
st.markdown("This tool predicts the 30-day readmission risk for diabetic patients.")

st.sidebar.header("Clinical Inputs")


def get_user_inputs():
    # We collect values for the 4 most important features from our SHAP analysis
    inputs = {
        'number_inpatient': st.sidebar.slider("Inpatient Visits (Past Year)", 0, 15, 1),
        'num_medications': st.sidebar.slider("Number of Medications", 1, 80, 10),
        'time_in_hospital': st.sidebar.slider("Days in Hospital", 1, 14, 3),
        'num_lab_procedures': st.sidebar.number_input("Number of Lab Procedures", 1, 130, 40)
    }
    return inputs


user_values = get_user_inputs()

# --- 3. DATA ALIGNMENT (The 101-Feature Matrix) ---
# Create a matrix of zeros for all 101 features
input_matrix = np.zeros((1, len(expected_features)))
input_df = pd.DataFrame(input_matrix, columns=expected_features)

# Fill the user-provided values into the correct columns
for key, value in user_values.items():
    if key in input_df.columns:
        input_df.at[0, key] = value

# --- 4. PREDICTION ---
if st.button("Generate Risk Assessment"):
    # Calculate probability
    prob = model.predict_proba(input_df)[0][1]

    st.divider()

    # Professional Display
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Readmission Probability", value=f"{prob:.1%}")

    with col2:
        if prob > 0.5:
            st.error("Status: HIGH RISK")
            st.write("Clinical Note: Prioritize for transitional care support.")
        else:
            st.success("Status: LOW RISK")
            st.write("Clinical Note: Follow standard discharge protocol.")

    # Show a small hint about the most important feature
    st.info(f"Insight: Prior inpatient visits ({user_values['number_inpatient']}) heavily influenced this score.")