import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "datasets" / "customer_feedback.csv"
MODEL_PATH = ROOT / "task4_transformers" / "saved_model"
METRICS_PATH = ROOT / "results" / "task4_distilbert_metrics.json"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

label_names = ["negative", "neutral", "positive"]

label2id = {
    label: index
    for index, label in enumerate(label_names)
}

id2label = {
    index: label
    for label, index in label2id.items()
}

df["label_id"] = df["label"].map(label2id)

if df["label_id"].isna().any():
    raise ValueError("Dataset contains an unknown label.")

print("Dataset size:", len(df))
print("Label mapping:", label2id)


# --------------------------------------------------
# 2. Split train, validation and test
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_id"],
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    stratify=train_df["label_id"],
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. Load tokenizer
# --------------------------------------------------

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 4. Create custom PyTorch Dataset
# --------------------------------------------------

class FeedbackDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label_id"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        encoding["labels"] = self.labels[index]

        return encoding


train_dataset = FeedbackDataset(train_df)
val_dataset = FeedbackDataset(val_df)
test_dataset = FeedbackDataset(test_df)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# --------------------------------------------------
# 5. Load pre-trained DistilBERT classifier
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Training device:", device)


# --------------------------------------------------
# 6. Define evaluation metrics
# --------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "accuracy": accuracy,
        "f1": macro_f1,
    }


# --------------------------------------------------
# 7. Configure Trainer
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=str(ROOT / "task4_transformers" / "checkpoints"),
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    save_total_limit=1,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# --------------------------------------------------
# 8. Fine-tune model
# --------------------------------------------------

print("\n===== DISTILBERT TRAINING STARTED =====")

training_start = time.perf_counter()

trainer.train()

training_time = time.perf_counter() - training_start

print("\nTraining completed!")
print("Training time:", round(training_time, 2), "seconds")


# --------------------------------------------------
# 9. Evaluate on untouched test dataset
# --------------------------------------------------

print("\n===== TEST DATA EVALUATION =====")

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

print(test_results)


# --------------------------------------------------
# 10. Measure inference time
# --------------------------------------------------

model.eval()

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)

all_predictions = []
all_labels = []

inference_start = time.perf_counter()

with torch.no_grad():

    for batch in test_loader:

        labels = batch.pop("labels")

        batch = {
            key: value.to(trainer.model.device)
            for key, value in batch.items()
        }

        outputs = trainer.model(**batch)

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

inference_time = time.perf_counter() - inference_start

inference_per_sample = inference_time / len(test_dataset)

test_accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

test_macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
)

print("Test accuracy:", round(test_accuracy, 4))
print("Test macro F1:", round(test_macro_f1, 4))
print(
    "Inference time per sample:",
    round(inference_per_sample, 6),
    "seconds",
)


# --------------------------------------------------
# 11. Save fine-tuned model and tokenizer
# --------------------------------------------------

MODEL_PATH.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(MODEL_PATH))
tokenizer.save_pretrained(str(MODEL_PATH))

print("\nFine-tuned model saved:", MODEL_PATH)


# --------------------------------------------------
# 12. Save metrics for comparison
# --------------------------------------------------

metrics = {
    "model": "DistilBERT",
    "accuracy": float(test_accuracy),
    "macro_f1": float(test_macro_f1),
    "training_time_seconds": float(training_time),
    "inference_time_per_sample_seconds": float(inference_per_sample),
    "test_samples": len(test_dataset),
    "labels": label_names,
}

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved:", METRICS_PATH)
print("\n===== TASK 4 MODEL TRAINING COMPLETE =====")

