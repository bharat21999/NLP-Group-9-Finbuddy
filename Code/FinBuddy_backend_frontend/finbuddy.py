import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fetch_stock import fetch_stock_info
import re
from faiss_stock_retriever import retrieve_ticker

# === Load tokenizer and model ===
adapter_path = "/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter"
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === Inference function ===
def generate_answer(prompt):
    system_prompt = (
        "You are the best finance teacher in the world. Explain financial concepts in simple and clear terms "
        "with real-world examples that even a child can understand. Never echo instructions."
    )
    full_prompt = f"{system_prompt}\n\n{prompt.strip()}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace(full_prompt, "").strip()

# === Stock summarization function ===
def summarize_stock(company, ticker, stock_data):
    summary_prompt = (
        f"You are a financial teacher. The user asked about the stock of {company} ({ticker}). "
        f"Here's the latest stock data:\n\n{stock_data}\n\n"
        "Do not repeat the user's question. Just provide an easy explanation and summarization of the fetched stock data."
    )

    inputs = tokenizer(summary_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace(summary_prompt, "").strip()

# === Main chatbot loop ===
def start_chat():
    print("💬 FinBuddy: Ask me about finance or a stock (e.g., 'What is a bond?' or 'Tell me about TSLA')")
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Bye! Stay financially smart.")
            break

        # === Handle stock-related queries ===
        if "stock" in user_input.lower() or "price" in user_input.lower():
            try:
                result = retrieve_ticker(user_input)
                ticker = result['ticker']
                company = result['company']
                note = result.get('note', "")
                print(f"\n📈 Fetching info for {company} ({ticker})... {note}")

                stock_response = fetch_stock_info(ticker)
                print(f"\n📊 Raw Stock Data:\n{stock_response}")

                print("\n🧠 Explaining what this means...")
                summary = summarize_stock(company, ticker, stock_response)
                print(f"\nFinBuddy: {summary}")

            except Exception as e:
                print(f"⚠️ Could not find a valid stock: {e}")
        else:
            print("\n🧠 Thinking...")
            answer = generate_answer(user_input)
            print(f"\nFinBuddy: {answer}")

# === Run chatbot ===
if __name__ == "__main__":
    start_chat()
