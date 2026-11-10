
from transformers import pipeline

# Create sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Sample sentences
texts = [
    "I love learning Deep Learning and NLP!",
    "This project is very difficult and frustrating.",
    "Python is an amazing programming language."
]

# Run prediction
results = classifier(texts)

# Display results
for text, result in zip(texts, results):
    print("\nText:", text)
    print("Label:", result["label"])
    print("Confidence:", round(result["score"], 4))
