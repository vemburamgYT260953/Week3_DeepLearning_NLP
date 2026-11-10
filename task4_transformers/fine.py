
import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# 1. Configuration
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "datasets/customer_feedback.csv"
SAVE_PATH = "task4_transformers/saved_model"

labels = ["negative", "neutral", "positive"]
label2id = {label: i for i, label in enumerate(labels)}

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# 3. Validate and prepare data
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].str.lower().str.strip()

if not set(df["label"]).issubset(label2id):
    raise ValueError("Dataset contains unexpected labels!")

df["label"] = df["label"].map(label2id)

# 4. Split dataset
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True), preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df.reset_index(drop=True), preserve_index=False
)

# 5. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# 6. Load pretrained DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={i: label for label, i in label2id.items()},
    label2id=label2id
)

# 7. Evaluation metrics
def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1": f1_score(
            true_labels,
            predictions,
            average="weighted"
        )
    }

# 8. Training configuration
training_args = TrainingArguments(
    output_dir="task4_transformers/checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=20,
    report_to="none",
    seed=42
)

# 9. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer
    ),
    compute_metrics=compute_metrics
)

# 10. Fine-tune model
print("\nStarting DistilBERT fine-tuning...")

trainer.train()

# 11. Evaluate model
metrics = trainer.evaluate()

print("\nFine-tuning evaluation:")
print(metrics)

# 12. Save trained model
os.makedirs(SAVE_PATH, exist_ok=True)

trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\nFine-tuned model saved successfully!")
print("Saved location:", SAVE_PATH)
