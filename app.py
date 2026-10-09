
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("salary_model.pkl")

model = load_model()

with open("options.json", "r") as f:
    options = json.load(f)

st.title("💰 Employee Salary Prediction")
st.write("Predict an estimated salary based on employee details.")
st.divider()

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
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

if st.button("🔮 Predict Salary", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_role],
        "Years of Experience": [experience]
    })

    try:
        prediction = model.predict(input_data)[0]
        salary = float(prediction)

        st.subheader("Prediction Result")
        st.success(f"💰 Estimated Salary: ${salary:,.2f}")

        st.caption(
            "This is a model-based estimate, not a guaranteed salary."
        )

    except Exception as e:
        st.error("Prediction failed. Error details:")
        st.code(repr(e))
