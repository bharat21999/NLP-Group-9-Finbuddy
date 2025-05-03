import json
import re

# === Input/Output paths ===
input_path = "/home/ubuntu/finbuddy/Data/improved_financial_training_data.jsonl"
output_path = "/home/ubuntu/finbuddy/Data/final_curated_finetune_data.jsonl"

# === Step 1: High-quality curated analogies for select terms ===
analogy_map = {
    "stock": "Think of it like sharing slices of pizza. Each slice you own is like owning a part of a company—that’s what a stock is.",
    "bond": "It’s like when you lend your friend $10, and they promise to give you $12 later. That’s how bonds work.",
    "revenue": "Revenue is the total money a lemonade stand makes before buying lemons and sugar.",
    "credit": "A credit score is like a report card for your money habits. If you pay bills on time, you get a high score.",
    "ledger": "Think of a general ledger like a school notebook where every transaction is written down and totaled.",
    "mutual fund": "A mutual fund is like a classroom jar where everyone puts in money and a teacher decides how to use it.",
    "diversification": "It’s like packing snacks, fruits, and sandwiches. If you don’t like one thing, you still have others—just like spreading investments.",
    "interest": "Interest is like borrowing a toy and giving it back with an extra candy to say thank you.",
    "contract": "A contract is like promising to swap your lunch with a friend and both of you writing it down.",
    "hierarchy": "Corporate hierarchy is like a school where the principal is at the top, then teachers, then students. Everyone has a role."
}

# === Step 2: Load, match analogies, and clean ===
final_data = []

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        term_match = re.search(r"'(.+?)'", item["instruction"])
        term = term_match.group(1).lower() if term_match else ""

        output = item["output"].strip()
        matched = False

        for keyword, analogy in analogy_map.items():
            if keyword in term or keyword in output.lower():
                output = output.split(".")[0].strip() + f". {analogy}"
                matched = True
                break

        if not matched:
            output = output.split(".")[0].strip() + "."

        final_data.append({
            "instruction": item["instruction"],
            "input": "",
            "output": output
        })

# === Step 3: Save final dataset ===
with open(output_path, "w", encoding="utf-8") as outfile:
    for entry in final_data:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f" Saved {len(final_data)} curated entries to {output_path}")
