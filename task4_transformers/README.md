# Task 4 – Transformers and FastAPI Deployment

## Objective
Use Hugging Face Transformers to demonstrate tokenization,
pre-trained NLP pipelines, DistilBERT fine-tuning and API deployment.

## 1. DistilBERT Tokenizer
- Model: distilbert-base-uncased
- Display tokens
- Display input_ids
- Display attention_mask
- Test two sentences

## 2. Hugging Face Pipelines
- Sentiment analysis on three sentences
- Named Entity Recognition on three sentences

## 3. Fine-tuning
- Model: distilbert-base-uncased
- Dataset: customer_feedback.csv
- Classes: negative, neutral, positive
- Train/validation/test split
- Training API: Hugging Face Trainer
- Epochs: 2
- Optimizer: AdamW (Trainer default)
- Learning rate: 2e-5
- Evaluate using accuracy and macro F1

## 4. Model Comparison
Compare TF-IDF + Logistic Regression and DistilBERT using:
- Accuracy
- Macro F1
- Training time
- Inference time per sample

## 5. FastAPI
- GET /
- POST /predict
- Returns predicted label and confidence
- Loads model once at application startup
- Empty text returns HTTP 422

## 6. BERT vs GPT
BERT uses a bidirectional encoder for understanding text.
GPT uses a causal decoder for next-token text generation.

## Run Instructions

From the project root:

python task4_transformers/tokenizer_demo.py
python task4_transformers/pipelines_demo.py
python task4_transformers/train_distilbert.py
python task4_transformers/compare_models.py

Start API:

uvicorn task4_transformers.app:app --reload

Open:

http://127.0.0.1:8000/docs