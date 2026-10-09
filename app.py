```python
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

@st.cache_data
def load_options():
    with open("options.json", "r") as f:
        return json.load(f)

model = load_model()
options = load_options()

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

    try:
        # Get the exact feature names expected by the loaded model
        expected_columns = list(model.feature_names_in_)

        values = {
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_role,
            "Years of Experience": experience,
            "Experience (Years)": experience
        }

        # Build input using the model's exact column names
        missing = [col for col in expected_columns if col not in values]

        if missing:
            st.error(f"Cannot match model input columns: {missing}")
        else:
            input_data = pd.DataFrame([
                {col: values[col] for col in expected_columns}
            ])

            prediction = model.predict(input_data)[0]

            if isinstance(prediction, str):
                st.error(
                    f"The loaded model predicts a category: {prediction}. "
                    "Upload the trained numerical salary regression model."
                )
            else:
                salary = float(prediction)
                st.subheader("Prediction Result")
                st.success(f"💰 Estimated Salary: ${salary:,.2f}")
                st.caption(
                    "This is a model-based estimate, not a guaranteed salary."
                )

    except Exception as e:
        st.error("Prediction failed. Error details:")
        st.code(repr(e))
```
