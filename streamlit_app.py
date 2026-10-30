
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("group6_random_forest.joblib")

st.set_page_config(
    page_title="Hypertension Research Demo",
    page_icon="💙",
    layout="centered"
)

st.title("💙 Hypertension Classification Demo")

st.write(
    "This research project explores how age, BMI and "
    "lifestyle factors relate to measured blood pressure."
)

st.warning(
    "Educational research prototype only. This application "
    "does not diagnose hypertension or predict future disease."
)

st.subheader("Participant Information")

with st.form("participant_form"):

    age = st.number_input(
        "Age (years)",
        min_value=20,
        max_value=100,
        value=40
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    st.subheader("Lifestyle Questions")

    vigorous = st.selectbox(
        "Do you do vigorous recreational activities?",
        ["Yes", "No"]
    )

    moderate = st.selectbox(
        "Do you do moderate recreational activities?",
        ["Yes", "No"]
    )

    smoking = st.selectbox(
        "Have you smoked at least 100 cigarettes in your lifetime?",
        ["Yes", "No"]
    )

    alcohol_history = st.selectbox(
        "Have you ever had at least one alcoholic drink?",
        ["Yes", "No"]
    )

    alcohol_frequency = st.selectbox(
        "How often did you drink alcohol in the previous 12 months?",
        [
            "Never in the last year",
            "Every day",
            "Nearly every day",
            "3–4 times a week",
            "Twice a week",
            "Once a week",
            "2–3 times a month",
            "Once a month",
            "7–11 times in the last year",
            "3–6 times in the last year",
            "1–2 times in the last year"
        ]
    )

    submitted = st.form_submit_button(
        "Check Model Result"
    )


if submitted:

    yes_no_mapping = {
        "Yes": 1,
        "No": 2
    }

    frequency_mapping = {
        "Never in the last year": 0,
        "Every day": 1,
        "Nearly every day": 2,
        "3–4 times a week": 3,
        "Twice a week": 4,
        "Once a week": 5,
        "2–3 times a month": 6,
        "Once a month": 7,
        "7–11 times in the last year": 8,
        "3–6 times in the last year": 9,
        "1–2 times in the last year": 10
    }

    input_data = pd.DataFrame([{
        "RIDAGEYR": age,
        "BMXBMI": bmi,
        "PAQ650": yes_no_mapping[vigorous],
        "PAQ665": yes_no_mapping[moderate],
        "SMQ020": yes_no_mapping[smoking],
        "ALQ111": yes_no_mapping[alcohol_history],
        "ALQ121": frequency_mapping[alcohol_frequency]
    }])

    try:
        expected_features = list(model.feature_names_in_)

        if set(input_data.columns) != set(expected_features):
            st.error(
                "Input features do not match the trained model. "
                "Please check the training feature list."
            )
        else:
            input_data = input_data[expected_features]

            prediction = int(model.predict(input_data)[0])

            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            st.subheader("Model Result")

            if prediction == 1:
                st.error(
                    "The model classified this input as "
                    "at or above the study's blood-pressure threshold."
                )
            else:
                st.success(
                    "The model classified this input as "
                    "below the study's blood-pressure threshold."
                )

            st.metric(
                "Estimated positive-class probability",
                f"{probability:.1%}"
            )

            st.caption(
                "This is a model-generated estimate, not a "
                "medical diagnosis or an individual's future "
                "hypertension risk."
            )

    except Exception as error:
        st.error(f"Prediction error: {error}")
