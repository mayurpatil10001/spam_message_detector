# SpamShield — Web-Based NLP Spam Message Detector

A Flask web app that classifies an SMS/message as **Likely Spam** or **Likely Legitimate** using TF-IDF text features and Logistic Regression.

## Features
- Responsive browser-based UI
- Message text input with character counter
- Spam/legitimate prediction and model confidence
- Basic language-signal hints
- Built-in example buttons
- About page describing the pipeline
- Model pipeline saved locally with `joblib`
- Health endpoint at `/health`
- No API key or internet connection required for model predictions

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Internet only for installing packages; the Google Fonts import is optional for styling

## Run on Windows PowerShell
Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open `http://127.0.0.1:5000` in your browser. Stop the server with `Ctrl+C`.

## Project structure
```text
spam_message_detector/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   └── about.html
├── static/
│   └── style.css
└── models/              # model file generated on first run
```

## Important dataset note
The project includes a **small, hand-written demonstration dataset** to make first run simple. It is not representative enough for a trustworthy real-world spam filter, and the displayed confidence is not calibrated. For a stronger academic submission, replace `TRAINING_DATA` in `app.py` with a larger labeled SMS dataset, use a stratified train/test split, and report precision, recall, F1-score, and a confusion matrix on held-out data. Do not report a test accuracy unless you actually evaluate on unseen test examples.

## Viva explanation
**Problem:** Unwanted SMS messages can contain promotional, deceptive, or suspicious language. The application demonstrates automatic text classification.

**NLP steps:**
1. Collect labeled example messages (`spam` and `ham`, where `ham` means legitimate).
2. TF-IDF vectorization turns words and word pairs into numeric features. Frequent terms within a message get more weight, while terms common across all messages get less weight.
3. Logistic Regression learns a decision boundary from these features and predicts a class.
4. Flask receives the submitted text, calls the saved scikit-learn pipeline, and renders the result in HTML.
5. Joblib saves the trained pipeline so it can be loaded on future starts.

**Why TF-IDF?** Machine-learning classifiers need numbers, not raw text. TF-IDF is lightweight and easy to explain for a beginner NLP project.

**Why Logistic Regression?** It is fast, interpretable as a baseline, and works well with sparse text features.

**Limitations:** Small training set; messages outside the examples may be misclassified; probability-like confidence values are not necessarily calibrated; this demo should not be used as a security product.

## Suggested improvements
- Use a larger public SMS Spam Collection dataset
- Add stratified train/test evaluation and a confusion matrix
- Let users upload a CSV and classify multiple messages
- Save prediction history to SQLite (avoid storing sensitive message content without consent)
- Add a confidence threshold and an “uncertain” outcome
