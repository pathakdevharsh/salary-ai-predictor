import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Load model without caching
def load_model():
    return joblib.load("salary_model.pkl")

@st.cache_data
def load_options():
    with open("options.json", "r") as f:
        return json.load(f)

model = load_model()
options = load_options()

st.title("💰 Employee Salary Prediction")
st.write("Predict an estimated salary based on employee details.")
st.divider()

expected_columns = list(model.feature_names_in_)

# Show the actual loaded model details
st.write("Model type:", type(model).__name__)
st.write("Expected columns:", expected_columns)

age = st.number_input("Age", min_value=18, max_value=70, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
experience = st.number_input(
    "Experience (Years)", min_value=0, max_value=50, value=2
)
education = st.selectbox("Education Level", options["Education Level"])
job_role = st.selectbox("Job Role", options["Job Title"])

if st.button("🔮 Predict Salary", use_container_width=True):
    try:
        values = {
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_role,
            "Years of Experience": experience,
            "Experience (Years)": experience
        }

        input_data = pd.DataFrame([{
            col: values[col] for col in expected_columns
        }])

        prediction = model.predict(input_data)[0]

        st.write("Raw model output:", repr(prediction))

        if isinstance(prediction, str):
            st.error(
                "The loaded file is a classification model, not a salary "
                "regression model. Please replace salary_model.pkl with "
                "the verified model saved from Colab."
            )
        else:
            salary = float(prediction)
            st.subheader("Prediction Result")
            st.success(f"💰 Estimated Salary: ${salary:,.2f}")
            st.caption(
                "This is a model-based estimate, not a guaranteed salary."
            )

    except Exception as e:
        st.error("Prediction failed:")
        st.code(repr(e))
