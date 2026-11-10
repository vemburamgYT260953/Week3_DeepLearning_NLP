import json
import re
import time
from pathlib import Path

from importlib import import_module
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


try:
    api = import_module("gensim.downloader")
except ModuleNotFoundError as exc:
    if exc.name == "gensim":
        raise ModuleNotFoundError(
            "Gensim is required. Install it with `python -m pip install gensim`."
        ) from exc
    raise


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "customer_feedback.csv"
METRICS_PATH = ROOT / "results" / "task3_embedding_metrics.json"


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z]+", text)


# 1. Load dataset
df = pd.read_csv(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

# 2. Load pre-trained GloVe vectors
print("Loading GloVe word embeddings...")
glove = api.load("glove-wiki-gigaword-50")
print("GloVe loaded successfully!")
print("Vector size:", glove.vector_size)

# 3. Find similar words
print("\n===== MOST SIMILAR WORDS =====")

for word in ["king", "computer"]:
    if word in glove:
        print(f"\nWords similar to '{word}':")
        print(glove.most_similar(word, topn=5))

# 4. Word analogy: king - man + woman = queen
print("\n===== WORD ANALOGY =====")

analogy = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5,
)

print("king - man + woman:")
print(analogy)

# 5. Convert each sentence into an average word vector
def sentence_vector(text):
    words = tokenize(text)

    vectors = [
        glove[word]
        for word in words
        if word in glove
    ]

    if not vectors:
        return np.zeros(glove.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0)


print("\nConverting feedback text into word vectors...")

X_train_vectors = np.vstack([
    sentence_vector(text) for text in X_train
])

X_test_vectors = np.vstack([
    sentence_vector(text) for text in X_test
])

print("Training vector shape:", X_train_vectors.shape)
print("Testing vector shape:", X_test_vectors.shape)

# 6. Train Logistic Regression using sentence embeddings
embedding_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

start_time = time.perf_counter()

embedding_model.fit(X_train_vectors, y_train)

training_time = time.perf_counter() - start_time

# 7. Evaluate
start_time = time.perf_counter()

predictions = embedding_model.predict(X_test_vectors)

inference_time = time.perf_counter() - start_time
inference_time_per_sample = inference_time / len(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("\n===== GLOVE EMBEDDING CLASSIFICATION RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Training time:", round(training_time, 4), "seconds")
print(
    "Inference time per sample:",
    round(inference_time_per_sample, 6),
    "seconds",
)

# 8. Save metrics
metrics = {
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "training_time_seconds": float(training_time),
    "inference_time_per_sample_seconds": float(inference_time_per_sample),
    "embedding": "GloVe 50-dimensional",
}

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nEmbedding metrics saved:", METRICS_PATH)