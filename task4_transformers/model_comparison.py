
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import pipeline


# 1. Load dataset
df = pd.read_csv("datasets/customer_feedback.csv")
df = df[["text", "label"]].dropna()

df["text"] = df["text"].astype(str)
df["label"] = df["label"].str.lower().str.strip()

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 3. Create traditional ML models
models = {
    "TF-IDF + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "TF-IDF + Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", MultinomialNB())
    ])
}

results = []

# 4. Train and evaluate traditional models
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    })


# 5. Load fine-tuned DistilBERT
print("\nLoading fine-tuned DistilBERT...")

classifier = pipeline(
    "text-classification",
    model="task4_transformers/saved_model",
    tokenizer="task4_transformers/saved_model",
    device=0 if torch.cuda.is_available() else -1
)

# 6. Predict using DistilBERT
bert_predictions = []

for text in X_test.tolist():
    output = classifier(
        text,
        truncation=True,
        max_length=128
    )[0]

    bert_predictions.append(output["label"].lower())

# 7. Evaluate DistilBERT
results.append({
    "Model": "Fine-tuned DistilBERT",
    "Accuracy": accuracy_score(y_test, bert_predictions),
    "Precision": precision_score(
        y_test, bert_predictions, average="weighted", zero_division=0
    ),
    "Recall": recall_score(
        y_test, bert_predictions, average="weighted", zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, bert_predictions, average="weighted", zero_division=0
    )
})

# 8. Display comparison
comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# 9. Identify best model
best_model = comparison.loc[
    comparison["F1 Score"].idxmax()
]

print("\nBest Model:", best_model["Model"])
print("Best F1 Score:", round(best_model["F1 Score"], 4))

# 10. Save results
comparison.to_csv(
    "datasets/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")

