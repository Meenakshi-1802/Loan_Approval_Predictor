# ---------------------- app.py ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="💰 Loan Approval Predictor", page_icon="💰", layout="wide")

# ---------------------- Brightness Toggle ----------------------
st.markdown("""
<div style='position:fixed; top:10px; right:10px; z-index:100;'>
    <button id='theme-toggle' style='font-size:20px;'>🔆</button>
</div>
<script>
const button = window.parent.document.querySelector('#theme-toggle');
button.onclick = () => {
    document.body.classList.toggle('dark-mode');
}
</script>
<style>
body.dark-mode {
    background-color: #111 !important;
    color: #EEE !important;
}
body.dark-mode .stButton>button {
    background-color: #333 !important;
    color: #EEE !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Load Model, Scaler, Encoders ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from app folder
MODEL_PATH = os.path.join(BASE_DIR, "Scripts", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Scripts", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "Scripts", "encoders.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# ---------------------- App Title ----------------------
st.title("💰 Loan Approval Predictor")

# ---------------------- Input Form ----------------------
with st.form(key="loan_form"):
    Loan_ID = st.text_input("Loan ID")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
    Credit_History = st.selectbox("Credit History", ["0", "1"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submit = st.form_submit_button("Predict Loan Approval")

# ---------------------- Prediction ----------------------
if submit:
    input_df = pd.DataFrame({
        "Gender": [Gender],
        "Married": [Married],
        "Dependents": [Dependents],
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area": [Property_Area]
    })

    # Fix Dependents
    input_df['Dependents'] = input_df['Dependents'].replace('3+', 3).astype(int)

    # ---------------------- Feature Engineering ----------------------
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['Loan_Income_Ratio'] = input_df['LoanAmount'] / (input_df['Total_Income'] + 1e-5)
    input_df['Has_Dependents'] = input_df['Dependents'].apply(lambda x: 1 if x > 0 else 0)
    input_df['LoanAmount_log'] = np.log(input_df['LoanAmount'] + 1)
    input_df['Total_Income_log'] = np.log(input_df['Total_Income'] + 1)

    # ---------------------- Encode Categorical ----------------------
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

    # ---------------------- Scale ----------------------
    scaled_input = scaler.transform(input_df)

    # ---------------------- Predict ----------------------
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] * 100

    if prediction == 1:
        st.success(f"Loan Approved ✅\nApproval Probability: {proba:.2f}%")
    else:
        st.error(f"Loan Denied ❌\nApproval Probability: {proba:.2f}%")

# ---------------------- GitHub DP ----------------------
st.markdown(f"""
<div style='position:fixed; bottom:10px; right:10px; display:flex; align-items:center; gap:10px;'>
    <img src="https://github.com/Meenakshi-1802.png" alt="GitHub DP" style="border-radius:50%; width:40px; height:40px;">
    <span style='font-weight:bold; color:#333;'>Meenakshi-1802</span>
</div>
""", unsafe_allow_html=True)
