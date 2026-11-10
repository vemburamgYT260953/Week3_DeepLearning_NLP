from transformers import pipeline

# Sentiment analysis pipeline
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

sentences = [
    "I love this product. The quality is excellent.",
    "The delivery was late and I am very disappointed.",
    "The product arrived yesterday. I have not used it yet.",
]

print("===== SENTIMENT ANALYSIS =====")

for sentence in sentences:
    result = sentiment(sentence)
    print("\nText:", sentence)
    print("Prediction:", result)


# Named Entity Recognition pipeline
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)

ner_sentences = [
    "Google opened a new office in London.",
    "Elon Musk works with Tesla in the United States.",
    "Microsoft was founded by Bill Gates in Washington.",
]

print("\n===== NAMED ENTITY RECOGNITION =====")

for sentence in ner_sentences:
    result = ner(sentence)
    print("\nText:", sentence)
    print("Entities:", result)
    