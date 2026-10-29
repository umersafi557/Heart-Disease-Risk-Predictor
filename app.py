from pathlib import Path
import warnings

import joblib
import streamlit as st

from preprocessing import build_model_input


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = {
    "model": "heart_disease_model.pkl",
    "scaler": "scaler.pkl",
    "numerical_columns": "numerical_columns.pkl",
    "training_columns": "training_columns.pkl",
}


@st.cache_resource
def load_artifacts():
    """Load and validate the model and preprocessing artifacts."""
    artifact_paths = {name: PROJECT_DIR / filename for name, filename in ARTIFACTS.items()}
    missing = [path.name for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required project file(s) are missing: " + ", ".join(missing)
        )

    # Joblib/pickle files must only be loaded from a trusted source.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model = joblib.load(artifact_paths["model"])
        scaler = joblib.load(artifact_paths["scaler"])
        numerical_columns = list(joblib.load(artifact_paths["numerical_columns"]))
        training_columns = list(joblib.load(artifact_paths["training_columns"]))

    if len(training_columns) != len(set(training_columns)):
        raise ValueError("training_columns.pkl contains duplicate feature names.")
    if len(numerical_columns) != len(set(numerical_columns)):
        raise ValueError("numerical_columns.pkl contains duplicate feature names.")
    if not set(numerical_columns).issubset(training_columns):
        raise ValueError(
            "numerical_columns.pkl contains columns not present in training_columns.pkl."
        )
    if getattr(scaler, "n_features_in_", len(numerical_columns)) != len(numerical_columns):
        raise ValueError(
            "The saved scaler's feature count does not match numerical_columns.pkl."
        )
    if getattr(model, "n_features_in_", len(training_columns)) != len(training_columns):
        raise ValueError(
            "The model's expected feature count does not match training_columns.pkl."
        )

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != training_columns:
        raise ValueError(
            "The saved model feature names/order do not match training_columns.pkl."
        )
    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != numerical_columns:
        raise ValueError(
            "The saved scaler feature names/order do not match numerical_columns.pkl."
        )

    return model, scaler, numerical_columns, training_columns


st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter the available measurements and click **Predict**. "
    "This educational demo is not medical advice and must not be used for diagnosis."
)
st.info(
    "The saved model's class labels and clinical performance have not been verified here. "
    "Treat the output as an unvalidated model class, not a diagnosis or a personal risk estimate."
)

try:
    model, scaler, numerical_columns, training_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info(
        "Keep app.py, preprocessing.py, heart_disease_model.pkl, scaler.pkl, "
        "numerical_columns.pkl, and training_columns.pkl in the project folder."
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
            "ST depression (oldpeak)",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Enter the value reported by the test; the model's original training range is not documented here.",
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
            help="Dataset source is a feature used by this saved model; it is not a patient measurement.",
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
        st.metric("Predicted model class", str(prediction))
        st.caption(
            "Class 0/1 meanings must be confirmed from the original target labels. "
            "This class is not a calibrated probability or a diagnosis."
        )

        # Some estimators expose predict_proba only when explicitly enabled.
        supports_probability = hasattr(model, "predict_proba") and getattr(
            model, "probability", True
        ) is not False
        if supports_probability:
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                positive_index = classes.index(1)
                probability = float(model.predict_proba(model_input)[0][positive_index])
                st.write(f"Model-estimated probability for class 1: **{probability:.1%}**")
            else:
                st.caption(
                    "Probability not shown because class 1 is not in the model's class labels."
                )
        else:
            st.caption(
                "This estimator does not expose probability estimates with its current settings."
            )

        st.warning(
            "Educational demonstration only. Confirm the labels, original preprocessing, "
            "validation procedure, and performance metrics before any research use. "
            "Do not use this result to make healthcare decisions."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption(
            "Check that the model, scaler, numerical_columns.pkl, training_columns.pkl, "
            "and preprocessing.py all come from the same training pipeline."
        )
