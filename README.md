
# Week 3 - Deep Learning + NLP

## Project
Customer Support Ticket Classification

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- FastAPI

## Tasks Completed
1. Neural Network Fundamentals
2. PyTorch Training
3. NLP Text Preprocessing
4. TF-IDF Classification
5. Hugging Face Tokenization
6. Pipeline Demonstration
7. DistilBERT Fine-tuning
8. Model Comparison
9. FastAPI Prediction API

## How to Run

Install dependencies:

pip install -r requirements.txt

Run tokenizer:

python task4_transformers/tokenizer_demo.py

Run pipeline:

python task4_transformers/pipeline_demo.py

Run API:

python -m uvicorn task4_transformers.api:app --reload

API documentation:

http://127.0.0.1:8000/docs

## Evaluation

See task4_transformers/model_comparison.csv
for actual model comparison results.

## Author
Vembu Ram G
