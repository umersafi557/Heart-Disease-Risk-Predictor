"""Feature engineering for the saved Heart Disease Risk Predictor model.

Keep these transforms synchronized with the original model-training pipeline.
The repository does not currently include that original training pipeline.
"""
import pandas as pd


def build_model_input(values: dict, training_columns: list[str]) -> pd.DataFrame:
    """Build one model row, preserving the saved training feature order."""
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
        # Preserved only to match the current saved feature schema. Verify against
        # the original training pipeline before using this model for research.
        "pulse_pressure": values["trestbps"] - 80,
        "high_bp": int(values["trestbps"] >= 140),
        "high_chol": int(values["chol"] >= 240),
    }

    categorical_values = {
        "sex_Male": values["sex"] == "Male",
        "dataset_Hungary": values["dataset"] == "Hungary",
        "dataset_Switzerland": values["dataset"] == "Switzerland",
        "dataset_VA Long Beach": values["dataset"] == "VA Long Beach",
        "cp_atypical angina": values["cp"] == "atypical angina",
        "cp_non-anginal": values["cp"] == "non-anginal",
        "cp_typical angina": values["cp"] == "typical angina",
        "fbs_True": bool(values["fbs"]),
        "restecg_normal": values["restecg"] == "normal",
        "restecg_st-t abnormality": values["restecg"] == "st-t abnormality",
        "exang_True": bool(values["exang"]),
        "slope_flat": values["slope"] == "flat",
        "slope_upsloping": values["slope"] == "upsloping",
        "thal_normal": values["thal"] == "normal",
        "thal_reversable defect": values["thal"] == "reversable defect",
    }

    if len(training_columns) != len(set(training_columns)):
        raise ValueError("The saved training feature list contains duplicate columns.")

    supported_columns = set(numeric_values) | set(categorical_values)
    supplied_columns = set(training_columns)
    missing = sorted(supported_columns - supplied_columns)
    unsupported = sorted(supplied_columns - supported_columns)
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing expected features: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported saved features: " + ", ".join(unsupported))
        raise ValueError(
            "The saved model feature schema does not match this app's preprocessing ("
            + "; ".join(details)
            + "). Use model artifacts trained with the matching preprocessing pipeline."
        )

    # Explicitly create every feature in the model's saved order; never use
    # get_dummies on a single row because it can omit unobserved categories.
    row = pd.DataFrame(0.0, index=[0], columns=training_columns)
    for column, value in numeric_values.items():
        row.at[0, column] = float(value)
    for column, selected in categorical_values.items():
        row.at[0, column] = float(selected)

    return row
