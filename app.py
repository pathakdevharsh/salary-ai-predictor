
import streamlit as st
import pandas as pd
import joblib
import json

# Load model
model = joblib.load("salary_model.pkl")

# Load options
with open("options.json", "r") as f:
    options = json.load(f)

# Page settings
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Employee Salary Prediction")
st.write("Predict Salary Category using Logistic Regression")

st.divider()

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

experience = st.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=50,
    value=2
)

education = st.selectbox(
    "Education Level",
    options["Education Level"]
)

job_role = st.selectbox(
    "Job Role",
    options["Job Title"]
)

st.divider()

# Prediction Button
if st.button("🔮 Predict Salary", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Experience (Years)": [experience],
        "Education Level": [education],
        "Job Title": [job_role]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "High Salary":
        st.success("💰 High Salary")
    else:
        st.warning("📉 Low Salary")
        
