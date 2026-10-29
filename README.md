# Heart Disease Risk Predictor

A small **Streamlit** demo that loads a saved scikit-learn classifier and formats user-entered heart-related measurements into the feature schema expected by the model.

> **Important:** This is an educational software project, not a medical device. It has not been clinically validated in this repository. Do not use its output to diagnose, screen for, or make treatment decisions about heart disease.

## Features

- Streamlit web form for entering the model's supported fields.
- Explicit 28-feature input schema that preserves the saved model's feature order.
- Input schema checks to catch mismatched model/scaler/preprocessing artifacts.
- Clear model-class output without inventing a probability score.
- Tests for one-hot encoding, feature values, and incompatible feature schemas.

## Requirements

- Windows, macOS, or Linux
- **Python 3.11 recommended**
- The four saved model/preprocessing artifacts in this repository

The model and scaler were serialized with scikit-learn 1.6.1, which is pinned in `requirements.txt`. Python 3.14 can make pip attempt a source build for this scikit-learn version. Use Python 3.11 for the simplest setup rather than changing the scikit-learn version without validating model compatibility.

## Run on Windows with VS Code

1. Download or clone the repository and open the extracted project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Run these commands one at a time:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, typically http://localhost:8501. Open that URL in your browser.

## Run on macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run tests

Install the development dependencies, then run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover feature-row construction and fail-fast behavior for feature-schema mismatches. They do not prove medical accuracy or validate the original training pipeline.

## Project structure

```text
.
├── app.py                       # Streamlit interface and model prediction
├── preprocessing.py             # Model input feature engineering
├── heart_disease_model.pkl      # Saved scikit-learn estimator
├── scaler.pkl                  # Saved numerical-feature scaler
├── numerical_columns.pkl       # Ordered numerical feature names
├── training_columns.pkl        # Ordered full model feature names
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Test dependencies
├── tests/
│   └── test_preprocessing.py
└── README.md
```

Keep `app.py`, `preprocessing.py`, and the four model/preprocessing artifacts together in the same project directory. The app resolves artifact paths relative to `app.py`, so it can be launched from another working directory.

## How the current model input is built

The app uses these existing numerical features: age, resting blood pressure, cholesterol, maximum heart rate, ST depression, fluoroscopy vessel count, four ratio features, `pulse_pressure`, `high_bp`, and `high_chol`. It also explicitly creates the saved one-hot categorical columns. The scaler is applied only to the saved numerical feature list.

**Do not retrain or change these formulas independently of the training pipeline.** In particular, `pulse_pressure = trestbps - 80` is retained only for compatibility with the current saved schema and is not the standard clinical pulse-pressure calculation. The original training code is not included, so that formula and the other engineered features still require review against the training pipeline.

## Known limitations

- The meaning of class labels 0 and 1, target definition, training data, train/test split, validation method, and performance metrics are not documented in the available artifacts. The UI therefore displays a model class, not a disease diagnosis or calibrated risk estimate.
- The dataset source is one of the model features. Dataset source describes the origin of data, not a patient measurement; its effect and appropriateness should be assessed in a proper model evaluation.
- The app's input limits are interface limits, not validated clinical reference ranges.
- Serialized joblib/pickle files can execute code during loading. Only use model artifacts from a trusted source.
- A local smoke test and feature-construction tests do not establish clinical validity.

## Troubleshooting

**`py -3.11` is not recognized:** install Python 3.11 and reopen the VS Code terminal.

**`streamlit` is not recognized:** use `python -m streamlit run app.py` after installing dependencies.

**scikit-learn installation fails under Python 3.14:** create a fresh virtual environment with Python 3.11 as shown above.

**Missing model file:** check that `heart_disease_model.pkl`, `scaler.pkl`, `numerical_columns.pkl`, and `training_columns.pkl` are present beside `app.py`.

**Feature schema mismatch:** keep the model, scaler, feature lists, and preprocessing code from the same training run. Do not fix the error by silently dropping columns.

## License

No license is currently specified. Contact the repository owner before reusing this project.
