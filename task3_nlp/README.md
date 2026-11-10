# Task 3 – Classical NLP Baseline and Word Embeddings

## Objective
Build a customer feedback classification system using TF-IDF,
Logistic Regression, and pre-trained GloVe word embeddings.

## Dataset
- File: datasets/customer_feedback.csv
- Total samples: 900
- Classes: positive, neutral, negative
- Samples per class: 300
- Split: 80% training and 20% testing

## Part 1 – TF-IDF + Logistic Regression
- Clean and lowercase text.
- Convert text into TF-IDF features.
- Train Logistic Regression.
- Evaluate accuracy and macro F1.
- Display classification report and confusion matrix.
- Display the top 10 important words for each class.

## Part 2 – GloVe Word Embeddings
- Load glove-wiki-gigaword-50.
- Find similar words for king and computer.
- Test the analogy king - man + woman.
- Convert each sentence into an average word vector.
- Train Logistic Regression using sentence vectors.
- Compare accuracy and macro F1 with TF-IDF.

## Generated Files
- ../datasets/customer_feedback.csv
- tfidf_model.joblib
- ../results/task3_tfidf_metrics.json
- ../results/task3_embedding_metrics.json

## Run Instructions

From the project root:

python task3_nlp/create_dataset.py
python task3_nlp/nlp_baseline.py
python task3_nlp/embeddings.py
