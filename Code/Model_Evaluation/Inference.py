import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

# === Paths ===
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter"

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

# === Base model with quantization ===
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

# === Load LoRA adapter ===
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === Prompt ===
prompt = "You are the best finance teacher in the world. Explain the term 'derivatives' in the simplest way possible with a real-world example that even a child can understand. Do not make up facts."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === Generate response ===
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

# === Decode output ===
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n Answer Only:\n", output_text.replace(prompt, "").strip())

