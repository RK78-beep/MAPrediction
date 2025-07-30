
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="M&A Deal Success Predictor", layout="centered")

st.title("🤖 M&A Deal Success Predictor")
st.markdown("Predict the likelihood of a successful M&A deal using a trained AI model.")

# Load model and scaler
with open("final_mna_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("final_mna_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# User Inputs
col1, col2 = st.columns(2)
with col1:
    target_val = st.number_input("Target Company Valuation ($M)", min_value=0.0, step=10.0)
    deal_val = st.number_input("Deal Value ($M)", min_value=0.0, step=10.0)
    past_deals = st.number_input("Previous M&A Deals by Acquirer", min_value=0, step=1)

with col2:
    acquirer_val = st.number_input("Acquirer Company Valuation ($M)", min_value=0.0, step=10.0)
    same_sector = st.selectbox("Are Both Companies in the Same Sector?", ["Yes", "No"])

# Convert to model input
X = np.array([[target_val, acquirer_val, deal_val, 1 if same_sector == "Yes" else 0, past_deals]])
X_scaled = scaler.transform(X)

# Prediction
if st.button("Predict Success"):
    prediction = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)[0][1]
    if prediction[0] == 1:
        st.success(f"✅ Deal Likely to Succeed — Confidence: {prob*100:.2f}%")
    else:
        st.error(f"❌ Deal Likely to Fail — Confidence: {prob*100:.2f}%")
