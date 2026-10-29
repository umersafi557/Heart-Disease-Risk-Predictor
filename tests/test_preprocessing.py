import pytest

from preprocessing import build_model_input


TRAINING_COLUMNS = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca",
    "sex_Male", "dataset_Hungary", "dataset_Switzerland",
    "dataset_VA Long Beach", "cp_atypical angina", "cp_non-anginal",
    "cp_typical angina", "fbs_True", "restecg_normal",
    "restecg_st-t abnormality", "exang_True", "slope_flat",
    "slope_upsloping", "thal_normal", "thal_reversable defect",
    "age_per_trestbps", "chol_per_thalch", "chol_per_age",
    "trest_per_age", "pulse_pressure", "high_bp", "high_chol",
]


def base_values():
    return {
        "age": 50,
        "sex": "Female",
        "cp": "asymptomatic",
        "trestbps": 120,
        "chol": 200,
        "fbs": False,
        "thalch": 150,
        "restecg": "lv hypertrophy",
        "exang": False,
        "oldpeak": 1.0,
        "slope": "downsloping",
        "thal": "fixed defect",
        "dataset": "Cleveland",
        "ca": 0,
    }


def test_builds_full_row_in_training_order():
    row = build_model_input(base_values(), TRAINING_COLUMNS)

    assert list(row.columns) == TRAINING_COLUMNS
    assert row.shape == (1, 28)
    assert row.loc[0, "age"] == 50.0
    assert row.loc[0, "pulse_pressure"] == 40.0
    assert row.loc[0, "high_bp"] == 0.0
    assert row.loc[0, "high_chol"] == 0.0
    # These are the omitted/reference categories in the saved one-hot schema.
    assert row.loc[0, "sex_Male"] == 0.0
    assert row.loc[0, "cp_typical angina"] == 0.0
    assert row.loc[0, "dataset_Hungary"] == 0.0


def test_selected_categorical_values_are_encoded():
    values = base_values()
    values.update({
        "sex": "Male",
        "cp": "typical angina",
        "dataset": "Switzerland",
        "fbs": True,
        "restecg": "normal",
        "exang": True,
        "slope": "flat",
        "thal": "normal",
    })
    row = build_model_input(values, TRAINING_COLUMNS)

    for column in (
        "sex_Male", "cp_typical angina", "dataset_Switzerland", "fbs_True",
        "restecg_normal", "exang_True", "slope_flat", "thal_normal",
    ):
        assert row.loc[0, column] == 1.0


def test_rejects_unknown_feature_schema():
    with pytest.raises(ValueError, match="unsupported saved features"):
        build_model_input(base_values(), TRAINING_COLUMNS + ["unknown_feature"])


def test_rejects_incomplete_feature_schema():
    with pytest.raises(ValueError, match="missing expected features"):
        build_model_input(base_values(), TRAINING_COLUMNS[:-1])


def test_rejects_duplicate_feature_names():
    with pytest.raises(ValueError, match="duplicate columns"):
        build_model_input(base_values(), TRAINING_COLUMNS + ["age"])
