from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ARTIFACTS = {
    "model": "heart_disease_model.pkl",
    "scaler": "scaler.pkl",
    "numerical_columns": "numerical_columns.pkl",
    "training_columns": "training_columns.pkl",
}


@st.cache_resource
def load_artifacts():
    """Load the model and preprocessing artifacts once per Streamlit session."""
    missing = [filename for filename in ARTIFACTS.values() if not Path(filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required project file(s) are missing: " + ", ".join(missing)
        )

    model = joblib.load(ARTIFACTS["model"])
    scaler = joblib.load(ARTIFACTS["scaler"])
    numerical_columns = list(joblib.load(ARTIFACTS["numerical_columns"]))
    training_columns = list(joblib.load(ARTIFACTS["training_columns"]))

    if len(training_columns) != len(set(training_columns)):
        raise ValueError("training_columns.pkl contains duplicate feature names.")
    if not set(numerical_columns).issubset(training_columns):
        raise ValueError(
            "The numerical feature list contains columns not present in training_columns.pkl."
        )
    if getattr(scaler, "n_features_in_", len(numerical_columns)) != len(numerical_columns):
        raise ValueError(
            "The saved scaler's feature count does not match numerical_columns.pkl."
        )
    if getattr(model, "n_features_in_", len(training_columns)) != len(training_columns):
        raise ValueError(
            "The model's expected feature count does not match training_columns.pkl."
        )

    return model, scaler, numerical_columns, training_columns


def build_model_input(values, training_columns):
    """Create one row using the fixed feature schema from model training."""
    numeric_values = {
        "age": values["age"],
        "trestbps": values["trestbps"],
        "chol": values["chol"],
        "thalch": values["thalch"],
        "oldpeak": values["oldpeak"],
        "ca": values["ca"],
        "age_per_trestbps": values["age"] / (values["trestbps"] + 1),
        "chol_per_thalch": values["chol"] / (values["thalch"] + 1),
        "chol_per_age": values["chol"] / (values["age"] + 1),
        "trest_per_age": values["trestbps"] / (values["age"] + 1),
        # Retained for compatibility with the saved model. Verify this formula
        # against the original training pipeline before using this model for research.
        "pulse_pressure": values["trestbps"] - 80,
        "high_bp": int(values["trestbps"] >= 140),
        "high_chol": int(values["chol"] >= 240),
    }

    # Preserve the exact feature order used when the model was trained.
    row = pd.DataFrame(0.0, index=[0], columns=training_columns)
    for column, value in numeric_values.items():
        if column in row.columns:
            row.at[0, column] = float(value)

    categorical_values = {
        "sex_Male": values["sex"] == "Male",
        "dataset_Hungary": values["dataset"] == "Hungary",
        "dataset_Switzerland": values["dataset"] == "Switzerland",
        "dataset_VA Long Beach": values["dataset"] == "VA Long Beach",
        "cp_atypical angina": values["cp"] == "atypical angina",
        "cp_non-anginal": values["cp"] == "non-anginal",
        "cp_typical angina": values["cp"] == "typical angina",
        "fbs_True": values["fbs"],
        "restecg_normal": values["restecg"] == "normal",
        "restecg_st-t abnormality": values["restecg"] == "st-t abnormality",
        "exang_True": values["exang"],
        "slope_flat": values["slope"] == "flat",
        "slope_upsloping": values["slope"] == "upsloping",
        "thal_normal": values["thal"] == "normal",
        "thal_reversable defect": values["thal"] == "reversable defect",
    }
    for column, selected in categorical_values.items():
        if column in row.columns:
            row.at[0, column] = float(selected)

    return row


st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter the available measurements and click **Predict**. "
    "This educational demo is not medical advice and must not be used for diagnosis."
)

try:
    model, scaler, numerical_columns, training_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info(
        "Keep app.py, heart_disease_model.pkl, scaler.pkl, "
        "numerical_columns.pkl, and training_columns.pkl in the same project folder."
    )
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox(
            "Chest pain type",
            ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=120
        )
        chol = st.number_input(
            "Cholesterol (mg/dL)", min_value=50, max_value=600, value=200
        )
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", [False, True])
        thalch = st.number_input(
            "Maximum heart rate achieved", min_value=50, max_value=250, value=150
        )

    with col2:
        restecg = st.selectbox(
            "Resting ECG",
            ["lv hypertrophy", "normal", "st-t abnormality"],
        )
        exang = st.selectbox("Exercise-induced angina", [False, True])
        oldpeak = st.number_input(
            "ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
        )
        slope = st.selectbox(
            "Slope of peak exercise ST", ["downsloping", "flat", "upsloping"]
        )
        thal = st.selectbox(
            "Thalassemia", ["fixed defect", "normal", "reversable defect"]
        )
        dataset = st.selectbox(
            "Dataset source",
            ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
        )
        ca = st.number_input(
            "Major vessels colored by fluoroscopy (0–3)",
            min_value=0,
            max_value=3,
            value=0,
            help="Enter a value from the relevant medical test; do not guess if unknown.",
        )

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalch": thalch,
        "restecg": restecg,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "thal": thal,
        "dataset": dataset,
        "ca": ca,
    }

    try:
        model_input = build_model_input(values, training_columns)
        model_input.loc[:, numerical_columns] = scaler.transform(
            model_input[numerical_columns]
        )
        prediction = model.predict(model_input)[0]

        st.subheader("Model output")
        if prediction == 1:
            st.error("The model classified this input as the positive class (1).")
        elif prediction == 0:
            st.success("The model classified this input as the negative class (0).")
        else:
            st.warning(f"The model returned class: {prediction!r}")

        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                positive_index = classes.index(1)
                probability = float(model.predict_proba(model_input)[0][positive_index])
                st.write(f"Model-estimated probability for class 1: **{probability:.1%}**")
            else:
                st.caption("Probability not shown because class 1 is not in the model's class labels.")
        else:
            st.caption("This model does not provide probability estimates.")

        st.warning(
            "Educational demonstration only. Class 0/1 meanings and clinical validity "
            "must be confirmed from the original training labels and evaluation. "
            "This result is not a diagnosis or medical advice."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption(
            "If this mentions feature names or dimensions, verify that the saved model, "
            "scaler, numerical_columns.pkl, and training_columns.pkl came from the same training run."
        )
