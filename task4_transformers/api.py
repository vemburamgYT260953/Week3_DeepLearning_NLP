
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI(
    title="Customer Feedback NLP API",
    description="Sentiment classification using fine-tuned DistilBERT",
    version="1.0"
)

MODEL_PATH = "task4_transformers/saved_model"

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1
)


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {
        "message": "Customer Feedback NLP API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "DistilBERT"
    }


@app.post("/predict")
def predict_feedback(request: FeedbackRequest):
    try:
        result = classifier(
            request.text,
            truncation=True,
            max_length=128
        )[0]

        return {
            "input_text": request.text,
            "prediction": result["label"],
            "confidence": round(float(result["score"]), 4)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

