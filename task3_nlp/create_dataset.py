import itertools
from pathlib import Path

import pandas as pd

# Three groups of customer feedback
feedback = {
    "positive": {
        "openers": [
            "I really like",
            "I am very happy with",
            "I absolutely love",
            "I am satisfied with",
            "I had a great experience with",
            "I strongly recommend",
            "I enjoyed using",
            "I am impressed by",
            "The quality of",
            "My experience with",
        ],
        "messages": [
            "the excellent product quality",
            "the fast delivery service",
            "the friendly customer support",
            "the simple ordering process",
            "the useful features",
            "the affordable price",
            "the smooth performance",
            "the reliable service",
            "the easy payment process",
            "the overall experience",
        ],
        "endings": [
            "and I would recommend it.",
            "and I will use it again.",
            "because it exceeded my expectations.",
        ],
    },
    "negative": {
        "openers": [
            "I really dislike",
            "I am very unhappy with",
            "I regret choosing",
            "I am disappointed with",
            "I had a terrible experience with",
            "I cannot recommend",
            "I struggled with",
            "I am frustrated by",
            "The poor quality of",
            "My bad experience with",
        ],
        "messages": [
            "the damaged product quality",
            "the delayed delivery service",
            "the rude customer support",
            "the confusing ordering process",
            "the missing features",
            "the expensive price",
            "the slow performance",
            "the unreliable service",
            "the failed payment process",
            "the overall experience",
        ],
        "endings": [
            "and I would not recommend it.",
            "and I will not use it again.",
            "because it did not meet my expectations.",
        ],
    },
    "neutral": {
        "openers": [
            "I have an average opinion of",
            "I have no strong opinion about",
            "I recently tried",
            "I would describe as average",
            "My experience with",
            "I noticed some things about",
            "I can give a neutral review of",
            "I have mixed feelings about",
            "I found",
            "I would say that",
        ],
        "messages": [
            "the product quality",
            "the delivery service",
            "the customer support",
            "the ordering process",
            "the available features",
            "the listed price",
            "the application performance",
            "the service reliability",
            "the payment process",
            "the overall experience",
        ],
        "endings": [
            "and it was neither good nor bad.",
            "and it was acceptable overall.",
            "but I do not have a strong preference.",
        ],
    },
}

rows = []

for label, groups in feedback.items():
    combinations = itertools.product(
        groups["openers"],
        groups["messages"],
        groups["endings"],
    )

    for opener, message, ending in combinations:
        text = f"{opener} {message} {ending}"
        rows.append({"text": text, "label": label})

df = pd.DataFrame(rows).drop_duplicates(subset="text")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = Path("datasets/customer_feedback.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSample records:")
print(df.head(5).to_string(index=False))
