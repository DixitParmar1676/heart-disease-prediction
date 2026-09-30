import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Load model files
@st.cache_resource
def load_files():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_files()

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient details below and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting BP (mm Hg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    # One-hot encode and match training columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale (only the columns the scaler was trained on, if available)
    if hasattr(scaler, "feature_names_in_"):
        cols_to_scale = list(scaler.feature_names_in_)
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    else:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=columns)

    pred = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        st.write(f"Risk probability: **{prob:.1%}**")

    if pred == 1:
        st.error("⚠️ High risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease.")

st.caption("This is a student ML project, not medical advice.")