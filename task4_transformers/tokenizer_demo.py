from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentences = [
    "I am very happy with the customer service.",
    "The delivery was late and the product was damaged.",
]

for sentence in sentences:
    encoded = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    print("\nSentence:", sentence)

    print("Tokens:")
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    print("Input IDs:")
    print(encoded["input_ids"][0].tolist())

    print("Attention Mask:")
    print(encoded["attention_mask"][0].tolist())
    