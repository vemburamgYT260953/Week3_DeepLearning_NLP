import json
import re
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "customer_feedback.csv"
MODEL_PATH = ROOT / "task3_nlp" / "tfidf_model.joblib"
METRICS_PATH = ROOT / "results" / "task3_tfidf_metrics.json"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 1. Load dataset
df = pd.read_csv(DATA_PATH)

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])

# 4. Train model and measure training time
start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

# 5. Make predictions and evaluate
start_time = time.perf_counter()

predictions = model.predict(X_test)

inference_time = time.perf_counter() - start_time
inference_time_per_sample = inference_time / len(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("\n===== TF-IDF MODEL RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Training time:", round(training_time, 4), "seconds")
print("Inference time per sample:", round(inference_time_per_sample, 6), "seconds")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=["negative", "neutral", "positive"]))

# 6. Find top 10 important words for each class
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()

print("\n===== TOP 10 WORDS PER CLASS =====")

top_words = {}

for class_index, class_name in enumerate(classifier.classes_):
    weights = classifier.coef_[class_index]
    top_indices = weights.argsort()[-10:][::-1]

    words = [
        {
            "word": feature_names[index],
            "weight": round(float(weights[index]), 4),
        }
        for index in top_indices
    ]

    top_words[class_name] = words

    print(f"\n{class_name.upper()}:")

    for item in words:
        print(item["word"], "->", item["weight"])

# 7. Save trained model
joblib.dump(model, MODEL_PATH)

# 8. Save metrics for Task 4 comparison
metrics = {
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "training_time_seconds": float(training_time),
    "inference_time_per_sample_seconds": float(inference_time_per_sample),
    "top_words": top_words,
}

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)

