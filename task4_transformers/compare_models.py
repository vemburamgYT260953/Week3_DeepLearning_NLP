import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TFIDF_PATH = ROOT / "results" / "task3_tfidf_metrics.json"
BERT_PATH = ROOT / "results" / "task4_distilbert_metrics.json"
OUTPUT_PATH = ROOT / "results" / "task4_model_comparison.csv"


with open(TFIDF_PATH, encoding="utf-8") as file:
    tfidf = json.load(file)

with open(BERT_PATH, encoding="utf-8") as file:
    bert = json.load(file)


comparison = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": tfidf["accuracy"],
        "Macro F1": tfidf["macro_f1"],
        "Training Time (seconds)": tfidf["training_time_seconds"],
        "Inference Time per Sample (seconds)": (
            tfidf["inference_time_per_sample_seconds"]
        ),
    },
    {
        "Model": "DistilBERT",
        "Accuracy": bert["accuracy"],
        "Macro F1": bert["macro_f1"],
        "Training Time (seconds)": bert["training_time_seconds"],
        "Inference Time per Sample (seconds)": (
            bert["inference_time_per_sample_seconds"]
        ),
    },
])

print("\n===== TF-IDF vs DISTILBERT =====")
print(comparison.to_string(index=False))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_PATH, index=False)

print("\nComparison saved:", OUTPUT_PATH)

