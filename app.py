from flask import Flask, render_template, request
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "spam_pipeline.joblib")
os.makedirs("models", exist_ok=True)

# Small built-in training set so the project runs immediately without downloading data.
# For stronger real-world performance, replace this with a larger labeled SMS dataset.
TRAINING_DATA = [
    ("Congratulations! You have won a free prize. Call now to claim", "spam"),
    ("URGENT! You have been selected for a cash reward. Click here", "spam"),
    ("Win a free iPhone today, limited offer, claim your prize now", "spam"),
    ("You have won 1000 dollars. Reply WIN to collect your cash", "spam"),
    ("Exclusive offer! Get cheap loans instantly. Apply now", "spam"),
    ("Claim your free vacation now. Call this number immediately", "spam"),
    ("Your account is selected for a special lottery reward", "spam"),
    ("Earn money from home quickly. Register now for guaranteed income", "spam"),
    ("You are a lucky winner. Send your details to receive the prize", "spam"),
    ("FREE entry in our weekly contest. Text WIN to 80000", "spam"),
    ("Get a discount now. Buy one get one free, offer ends today", "spam"),
    ("Your mobile number has won a gift voucher. Claim now", "spam"),
    ("Act now to receive a bonus. Limited time offer", "spam"),
    ("You have been approved for a loan. Call now", "spam"),
    ("Click this link to verify your account and claim reward", "spam"),
    ("Reminder: your appointment is scheduled for tomorrow at 10 AM", "ham"),
    ("Hey, are we still meeting for lunch today?", "ham"),
    ("Please call me when you reach the station", "ham"),
    ("I will be late by ten minutes. Please wait for me", "ham"),
    ("Can you send me the notes from today's lecture?", "ham"),
    ("Your package has been delivered to the reception desk", "ham"),
    ("Dinner is ready. Come home when you finish work", "ham"),
    ("Thanks for helping me with the assignment yesterday", "ham"),
    ("The meeting has been moved to 3 PM in room 204", "ham"),
    ("Don't forget to bring your notebook tomorrow", "ham"),
    ("I am on the train now. I will call you later", "ham"),
    ("Your electricity bill payment was received successfully", "ham"),
    ("Could you share the project file with me?", "ham"),
    ("Happy birthday! Have a wonderful day", "ham"),
    ("The class starts at 9 AM. Please be on time", "ham"),
    ("I reached home safely. Thanks for asking", "ham"),
    ("Please review the report and let me know your feedback", "ham"),
    ("Are you available for a quick call this evening?", "ham"),
    ("Your order is confirmed and will arrive on Friday", "ham"),
    ("I will send the address in a few minutes", "ham"),
    ("Let's study together for the exam this weekend", "ham"),
]

def train_model():
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return train_model()
    return train_model()

model = load_model()

def explain_message(message):
    lower = message.lower()
    signals = []
    if re.search(r"\b(free|winner|won|prize|claim|cash|bonus|reward)\b", lower):
        signals.append("promotional or prize-related wording")
    if re.search(r"\b(click|link|verify|urgent|act now|call now)\b", lower):
        signals.append("urgent action or link-related wording")
    if re.search(r"\b(loan|offer|discount|guaranteed|earn money)\b", lower):
        signals.append("offer or money-related wording")
    if re.search(r"\b(meeting|lecture|notes|home|train|tomorrow|thanks)\b", lower):
        signals.append("everyday conversational wording")
    return signals or ["overall wording patterns learned from the training examples"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)]) * 100
            result = {
                "label": prediction,
                "confidence": round(confidence, 1),
                "signals": explain_message(message),
                "message": message
            }
    return render_template("index.html", result=result, message=message)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    app.run(debug=True)
