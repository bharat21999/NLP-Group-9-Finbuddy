
import pandas as pd
import json
import random

# === Config ===
csv_path = "/home/ubuntu/finbuddy/Data/merged_financial_terms.csv"  # path to your input file
jsonl_output_path = "/home/ubuntu/finbuddy/Data/improved_financial_training_data.jsonl"

# === Load Data ===
df = pd.read_csv(csv_path)
df = df[['term', 'definition']].dropna()

# === Analogy Pool (Add more as you wish) ===
analogy_templates = [
    "Imagine you want to buy candy, and you have a dollar in your hand. You can use it right away—that’s liquid. But if your money is in a piggy bank that’s hard to open, it’s not very liquid.",
    "Think of it like sharing slices of pizza. Each slice you own is like owning a part of a company—that’s what a stock is.",
    "It’s like when you lend your friend $10, and they promise to give you $12 later. That’s how bonds work.",
    "It’s like keeping your lunchbox filled with snacks, fruits, and sandwiches. That’s diversification—if you don’t like one thing, you still have others.",
    "Imagine everyone puts money into a big classroom jar. Then a teacher decides how to spend it wisely for the class. That’s like a mutual fund.",
    "It’s like borrowing a toy from a friend and returning it with a thank-you candy—that’s interest.",
    "Think of saving for a bicycle by putting coins in different jars. That’s like investing in different funds to manage risk.",
]

# === Format Examples ===
formatted_data = []
for _, row in df.iterrows():
    term = row['term']
    definition = row['definition'].strip().split(".")[0]  # Use only first sentence for clarity
    example = random.choice(analogy_templates)

    instruction = (
        "You are the best teacher to teach finance."
        f"Explain the financial term '{term}' in the simplest way possible, like teaching a curious 10-year-old. "
        "Use a real-world example. Avoid technical words. Do not repeat the question. "
        "Never include Q&A labels or instructions. Just give a helpful and clear answer."
    )

    output = f"{definition}. {example}"

    formatted_data.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })

# === Save as JSONL ===
with open(jsonl_output_path, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved {len(formatted_data)} examples to {jsonl_output_path}")

