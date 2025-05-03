
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score

# === Load evaluation set ===
with open("/home/ubuntu/finbuddy/Data/eval_set_bertscore.json", "r") as f:
    eval_data = json.load(f)

# === Config ===
adapter_path = "/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter"
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Load base model ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
base_model.eval()

# === Load fine-tuned model ===
ft_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
ft_model.eval()

def generate_response(model, prompt):
    system_prompt = (
        "You are the best finance teacher in the world. Explain in a simple way using real-world examples. "
        "Do not include Q/A labels or made-up facts."
    )
    full_prompt = system_prompt + "\n\n" + prompt.strip()
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(full_prompt, "").strip()

# === Collect generations ===
base_outputs, ft_outputs, references = [], [], []
# === Collect generations and save to file ===
comparison_outputs = []

for item in eval_data:
    prompt = item["prompt"]
    reference = item["reference"]

    base_out = generate_response(base_model, prompt)
    ft_out = generate_response(ft_model, prompt)

    base_outputs.append(base_out)
    ft_outputs.append(ft_out)
    references.append(reference)

    comparison_outputs.append({
        "prompt": prompt,
        "answer_a": base_out,
        "answer_b": ft_out,
        "reference": reference
    })

# Save comparison set for GPT-4 evaluation
with open("gpt4_comparison_batch.jsonl", "w") as f:
    for entry in comparison_outputs:
        f.write(json.dumps(entry) + "\n")

# === Compute BERTScores ===
P_base, R_base, F1_base = score(base_outputs, references, lang="en", verbose=True)
P_ft, R_ft, F1_ft = score(ft_outputs, references, lang="en", verbose=True)

# === Print Average Scores ===
print("\\n=== BERTScore Evaluation ===")
print(f"Base Model - F1: {F1_base.mean().item():.4f}")
print(f"Fine-Tuned Model - F1: {F1_ft.mean().item():.4f}")
