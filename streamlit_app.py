import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="Group 6 | Hypertension Research",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- THEME ----------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f8fc; }
      [data-testid="stSidebar"] { background: #102b46; }
      [data-testid="stSidebar"] * { color: #f5f8fc !important; }
      .hero { background: linear-gradient(115deg,#102b46,#176b87); padding: 2.2rem 2.4rem;
              border-radius: 18px; color: white; margin-bottom: 1.2rem; }
      .hero h1 { color: white; font-size: 2.15rem; margin: 0 0 .45rem 0; }
      .hero p { color: #e3f1f8; font-size: 1.02rem; margin: 0; }
      .section-card { background: white; border: 1px solid #e5ebf2; border-radius: 14px;
                      padding: 1.15rem 1.3rem; margin: .5rem 0 1rem 0; }
      .eyebrow { color: #24748d; font-size: .78rem; font-weight: 700; letter-spacing: .08em;
                 text-transform: uppercase; }
      .small-note { color: #5b6b7c; font-size: .88rem; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e5ebf2;
                                      padding: 1rem; border-radius: 12px; }
      .stButton button, .stFormSubmitButton button { border-radius: 9px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- MODEL ----------------------------
MODEL_PATH = Path(__file__).with_name("group6_random_forest.joblib")
model = None
model_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_error = str(exc)

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.markdown("## 💙 Group 6")
    st.caption("Data Science & Digital Health")
    st.markdown("---")
    page = st.radio(
        "PROJECT MENU",
        ["Home", "Prediction Demo", "Model Performance", "Methodology", "About the Project"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Research prototype • Educational use only")

# ---------------------------- HELPERS ----------------------------
def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def disclaimer():
    st.warning(
        "Educational research prototype only. This model does not diagnose hypertension, "
        "replace professional medical assessment, or predict future disease."
    )

# ---------------------------- HOME ----------------------------
if page == "Home":
    hero(
        "Hypertension Classification Research",
        "Exploring how selected demographic and lifestyle factors relate to measured blood-pressure status.",
    )
    disclaimer()
    st.subheader("Project overview")
    st.write(
        "This Group 6 project develops and evaluates a machine-learning classification model "
        "using selected variables from the 2017–2018 National Health and Nutrition Examination Survey (NHANES). "
        "The prototype demonstrates how a trained model processes input variables; it is not a clinical tool."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-card"><div class="eyebrow">Research area</div><h3>Digital Health</h3><p class="small-note">Data science and machine learning</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card"><div class="eyebrow">Dataset</div><h3>NHANES 2017–2018</h3><p class="small-note">United States survey data</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="section-card"><div class="eyebrow">Model</div><h3>Random Forest</h3><p class="small-note">Classification prototype</p></div>', unsafe_allow_html=True)
    st.subheader("Explore the project")
    st.write("Use the menu on the left to try the prediction demo, review the reported model metrics, and read the study methodology.")

# ---------------------------- PREDICTION ----------------------------
elif page == "Prediction Demo":
    hero("Prediction Demo", "Enter the requested variables to see how the research model classifies the supplied input.")
    disclaimer()

    if model is None:
        st.error("The trained model could not be loaded. Confirm that `group6_random_forest.joblib` is in the same GitHub repository folder as this app.")
        with st.expander("Technical details"):
            st.code(model_error or "Unknown model-loading error")
    else:
        st.markdown("### Participant information")
        st.caption("The questions and response codes are based on the variables selected for the study.")
        with st.form("participant_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age (years)", min_value=20, max_value=100, value=40, step=1)
            with col2:
                bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=80.0, value=25.0, step=0.1)

            st.markdown("### Lifestyle questions")
            col1, col2 = st.columns(2)
            with col1:
                vigorous = st.selectbox("Do you do vigorous recreational activities?", ["Yes", "No"])
                smoking = st.selectbox("Have you smoked at least 100 cigarettes in your lifetime?", ["Yes", "No"])
                alcohol_history = st.selectbox("Have you ever had at least one alcoholic drink?", ["Yes", "No"])
            with col2:
                moderate = st.selectbox("Do you do moderate recreational activities?", ["Yes", "No"])
                alcohol_frequency = st.selectbox(
                    "How often did you drink alcohol in the previous 12 months?",
                    [
                        "Never in the last year", "Every day", "Nearly every day",
                        "3–4 times a week", "Twice a week", "Once a week",
                        "2–3 times a month", "Once a month",
                        "7–11 times in the last year", "3–6 times in the last year",
                        "1–2 times in the last year",
                    ],
                )
            submitted = st.form_submit_button("Run model classification", type="primary", use_container_width=True)

        if submitted:
            yes_no_mapping = {"Yes": 1, "No": 2}
            frequency_mapping = {
                "Never in the last year": 0, "Every day": 1, "Nearly every day": 2,
                "3–4 times a week": 3, "Twice a week": 4, "Once a week": 5,
                "2–3 times a month": 6, "Once a month": 7,
                "7–11 times in the last year": 8, "3–6 times in the last year": 9,
                "1–2 times in the last year": 10,
            }
            input_data = pd.DataFrame([{
                "RIDAGEYR": age,
                "BMXBMI": bmi,
                "PAQ650": yes_no_mapping[vigorous],
                "PAQ665": yes_no_mapping[moderate],
                "SMQ020": yes_no_mapping[smoking],
                "ALQ111": yes_no_mapping[alcohol_history],
                "ALQ121": frequency_mapping[alcohol_frequency],
            }])
            try:
                expected_features = list(model.feature_names_in_)
                if set(input_data.columns) != set(expected_features):
                    st.error("Input features do not match the features recorded by the trained model. Please check the training feature list.")
                    st.write("App inputs:", list(input_data.columns))
                    st.write("Model expects:", expected_features)
                else:
                    input_data = input_data[expected_features]
                    prediction = int(model.predict(input_data)[0])
                    probability = float(model.predict_proba(input_data)[0][1])
                    st.markdown("---")
                    st.subheader("Model output")
                    if prediction == 1:
                        st.error("The model classified this input as at or above the study's blood-pressure threshold.")
                    else:
                        st.success("The model classified this input as below the study's blood-pressure threshold.")
                    st.metric("Estimated positive-class probability", f"{probability:.1%}")
                    st.caption("This percentage is the model's estimated probability for its positive class, not a medical diagnosis or an individual's future hypertension risk.")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

# ---------------------------- MODEL PERFORMANCE ----------------------------
elif page == "Model Performance":
    hero("Model Performance", "A summary of the preliminary evaluation figures reported during model development.")
    disclaimer()
    st.info("These are the previously reported experiment results. Before final submission, verify them against the final evaluation notebook and saved model version.")
    metrics = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.647, "Precision": 0.6066, "Recall": 0.6760, "F1-score": 0.6394, "ROC-AUC": 0.7102},
        {"Model": "Random Forest", "Accuracy": 0.655, "Precision": 0.6046, "Recall": 0.7365, "F1-score": 0.6641, "ROC-AUC": 0.7110},
    ])
    st.subheader("Reported comparison")
    display_metrics = metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.1%}")
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    st.subheader("What the metrics mean")
    st.markdown(
        "- **Accuracy:** proportion of all classifications that were correct.\n"
        "- **Precision:** proportion of positive classifications that were correct.\n"
        "- **Recall:** proportion of actual positive cases identified by the model.\n"
        "- **F1-score:** balance between precision and recall.\n"
        "- **ROC-AUC:** how well the model separates the two classes across thresholds."
    )
    st.caption("The reported Random Forest scores are only slightly higher on accuracy, F1-score and ROC-AUC; this comparison alone does not establish statistically significant superiority.")

# ---------------------------- METHODOLOGY ----------------------------
elif page == "Methodology":
    hero("Study Methodology", "A concise overview of the dataset, target definition and modelling approach.")
    disclaimer()
    st.markdown("### Dataset")
    st.write("The project uses the CDC National Health and Nutrition Examination Survey (NHANES) 2017–2018 dataset. The source data files used in the project include demographic, body-measure, blood-pressure, physical-activity, smoking and alcohol questionnaire records, merged using the participant identifier (SEQN).")
    st.markdown("### Study target")
    st.write("For adults aged 20 years and above, the study target was defined from the available mean measured blood-pressure values: positive when mean systolic pressure was at least 130 mmHg or mean diastolic pressure was at least 80 mmHg. Records without both required mean measurements were not assigned a target.")
    st.markdown("### Input variables")
    st.write("Age, BMI, vigorous recreational activity, moderate recreational activity, lifetime smoking history, alcohol-use history and alcohol-use frequency were selected as model inputs.")
    st.markdown("### Model development")
    st.write("The project compared Logistic Regression and Random Forest classifiers using a stratified 80/20 train-test split with random_state=42. Preprocessing included imputation, scaling for numeric variables, one-hot encoding for categorical variables, and balanced class weights, as implemented in the training pipeline.")
    st.markdown("### Important interpretation")
    st.warning("This is a classification of measured blood-pressure status in the survey data. It is not a model that predicts who will develop hypertension in the future. The dataset is from the United States, so results may not generalise to the Nigerian population without local validation.")

# ---------------------------- ABOUT ----------------------------
elif page == "About the Project":
    hero("About Group 6", "Development and evaluation of a simple machine-learning model to classify hypertension using lifestyle factors.")
    st.markdown("### Thematic area")
    st.write("Data Science & Digital Health")
    st.markdown("### Project purpose")
    st.write("The project explores the use of selected demographic and lifestyle variables in a supervised machine-learning workflow and presents the output through an educational web application.")
    st.markdown("### Data source")
    st.write("CDC National Health and Nutrition Examination Survey (NHANES), 2017–2018.")
    st.markdown("### Research limitations")
    st.markdown(
        "- The dataset represents a United States survey population.\n"
        "- The model classifies a measured blood-pressure threshold; it does not forecast future disease.\n"
        "- Self-reported lifestyle variables may contain recall or reporting errors.\n"
        "- Model outputs should not be used to make healthcare decisions.\n"
        "- External and local validation would be required before any real-world clinical use. \n"
        "- The data is from 2017–2018, so it may not match how people live and what their health looks like today.\n"
        "- Things like diet, salt, sleep, family history, kidney disease and diabetes are not in our data. So the model can only show what goes along with high blood pressure, not what causes it. /n"
    )
    st.markdown("### Team")
    st.write("Group 6 — Yusuf Yunusa - AI/ML Engineer, Gabriel Olatunji Olaogun, Sheba simeon, Gift Kuyeti Victor, Ubaidat Adeola Olasunkanmi")

# ---------------------------- FOOTER ----------------------------
st.markdown("---")
st.caption("Group 6 Research Project • Data Science & Digital Health • Educational prototype only")
