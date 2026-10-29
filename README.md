# Heart Disease Risk Predictor

An educational Streamlit interface for a saved scikit-learn classifier. **It is not a medical device and must not be used to diagnose, screen, or guide treatment.**

## Requirements

- Windows, macOS, or Linux
- Python **3.11** (recommended for the saved scikit-learn 1.6.1 artifacts)
- The model artifacts in this repository

Python 3.14 may cause pip to try building scikit-learn 1.6.1 from source. Use Python 3.11 rather than changing the scikit-learn version arbitrarily, because serialized model compatibility depends on the version used to train it.

## Run on Windows with VS Code

Open the project folder in VS Code, then choose **Terminal → New Terminal**. Run these commands one at a time from the project directory:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks environment activation, you can skip activation and call the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

When Streamlit starts, open the local URL it prints, usually http://localhost:8501.

## Run on macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project files

```text
.
├── app.py
├── heart_disease_model.pkl
├── numerical_columns.pkl
├── scaler.pkl
├── training_columns.pkl
├── requirements.txt
└── README.md
```

Keep all four model/preprocessing files in the same folder as `app.py`. The application checks that these files exist and that the saved feature counts are consistent before allowing predictions.

## Known limitations

- The class meanings (0 and 1), training data, preprocessing pipeline, validation method, and clinical performance must be confirmed from the original training code and data. They cannot be established from the serialized files alone.
- The app retains the existing `pulse_pressure = trestbps - 80` feature formula to match the current saved-model schema. This is not the standard clinical definition of pulse pressure; verify it against the original training pipeline before using this model for research.
- The app displays a probability only if the loaded estimator supports `predict_proba`; it does not fabricate a confidence score.
- Only load pickle/joblib model files from sources you trust.

## Troubleshooting

- **`streamlit` is not recognized:** use `python -m streamlit run app.py` after installing requirements.
- **scikit-learn installation fails on Python 3.14:** install Python 3.11 and create the environment with `py -3.11 -m venv .venv`.
- **Missing artifact error:** check that the model, scaler, numerical columns, and training columns files are present beside `app.py`.

## License

No license is currently specified. Contact the repository owner before reusing this project.
