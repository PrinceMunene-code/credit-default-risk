import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Credit Default Risk", layout="centered")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_credit_default.pkl"

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# App title
# --------------------------------------------------
st.title("Credit Default Risk Prediction")
st.write(
    "This application predicts the probability of loan default "
    "using a trained LightGBM model."
)

# --------------------------------------------------
# User inputs
# --------------------------------------------------
st.header("Applicant Information")

person_age = st.number_input("Age", min_value=18, max_value=100)
person_income = st.number_input("Annual Income", min_value=0)
person_emp_length = st.number_input("Employment Length (years)", min_value=0.0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)

cb_person_default_on_file = st.selectbox(
    "Previous Default on Record?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

# --------------------------------------------------
# Build input dataframe
# --------------------------------------------------
input_data = pd.DataFrame(
    [
        {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "cb_person_default_on_file": cb_person_default_on_file,
        }
    ]
)

# --------------------------------------------------
# Align features to training schema
# --------------------------------------------------
expected_features = model.feature_name_

for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_features]

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Default Risk"):
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Probability of Default:** {probability:.2%}")

    if probability >= 0.5:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")
