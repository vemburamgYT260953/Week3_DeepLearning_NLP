from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Model directory
MODEL_PATH = Path(__file__).resolve().parent / "saved_model"

# Create API
app = FastAPI(
    title="Customer Feedback Classifier",
    description="Predicts whether customer feedback is positive, neutral or negative.",
    version="1.0",
)

# Load model ONCE when the application starts
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
)

model.to(device)
model.eval()

print("DistilBERT model loaded successfully!")


# Request format
class PredictionRequest(BaseModel):

    text: str = Field(
        min_length=1,
        description="Customer feedback text",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be empty or whitespace.")
        return value.strip()


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Feedback Classifier API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1,
        )

    label_id = predicted_class.item()

    label = model.config.id2label[label_id]

    return {
        "text": request.text,
        "label": label,
        "confidence": round(confidence.item(), 4),
    }