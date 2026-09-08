from flask import Flask, request, render_template
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# ================== NLTK SETUP (only download once) ==================
nltk_data_path = "nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    stopwords.words("english")
except:
    nltk.download("stopwords", download_dir=nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet", download_dir=nltk_data_path)

# ================== FLASK APP ==================
app = Flask(__name__)

# ================== LOAD MODELS ==================
rf_model = joblib.load("saved_models/rf_model.pkl")
tfidf = joblib.load("saved_models/tfidf.pkl")
keras_tokenizer = joblib.load("saved_models/tokenizer.pkl")

# ✅ FIXED: correct format
lstm_model = load_model("saved_models/lstm_model.keras")

# ================== TEXT PROCESSING ==================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
MAX_LEN = 300

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    model_choice = request.form.get("model", "rf")

    cleaned = clean_text(news)

    if model_choice == "lstm":
        seq = keras_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        prob = lstm_model.predict(padded)[0][0]

        label = "REAL" if prob >= 0.5 else "FAKE"
        confidence = round(float(prob if prob >= 0.5 else 1 - prob) * 100, 2)
        model_used = "LSTM"

    else:
        vec = tfidf.transform([cleaned])

        pred = rf_model.predict(vec)[0]
        conf = rf_model.predict_proba(vec)[0]

        label = "REAL" if pred == 1 else "FAKE"
        confidence = round(max(conf) * 100, 2)
        model_used = "Random Forest"

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        model_used=model_used,
        news=news
    )

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True)
