import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# === Load evaluation dataset ===
eval_path = "/home/ubuntu/finbuddy/Data/eval_set_bertscore.json"
with open(eval_path) as f:
    eval_data = json.load(f)

# === Model setup ===
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Load base and fine-tuned model ===
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
).eval()

ft_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
ft_model = PeftModel.from_pretrained(ft_model, adapter_path).eval()

# === Generation helper ===
def generate(model, prompt):
    system_prompt = "You are a finance teacher. Explain in simple terms using real-world examples. No labels or made-up facts."
    full_prompt = system_prompt + "\n\n" + prompt.strip()
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).replace(full_prompt, "").strip()

# === Evaluation setup ===
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method4

base_rouge, ft_rouge = [], []
base_bleu, ft_bleu = [], []

# === Evaluation loop ===
for ex in eval_data:
    prompt = ex["prompt"]
    reference = ex["reference"]

    base_out = generate(base_model, prompt)
    ft_out = generate(ft_model, prompt)

    # ROUGE
    base_rouge.append(scorer.score(reference, base_out))
    ft_rouge.append(scorer.score(reference, ft_out))

    # BLEU
    ref_tokens = [reference.split()]
    try:
        base_bleu.append(sentence_bleu(ref_tokens, base_out.split(), smoothing_function=smooth_fn))
    except TypeError:
        base_bleu.append(0.0)
    try:
        ft_bleu.append(sentence_bleu(ref_tokens, ft_out.split(), smoothing_function=smooth_fn))
    except TypeError:
        ft_bleu.append(0.0)

# === Average scores ===
def avg_rouge(scores, metric):
    return sum([s[metric].fmeasure for s in scores]) / len(scores)

print("\n=== ROUGE Comparison ===")
print(f"Base - ROUGE-1: {avg_rouge(base_rouge, 'rouge1'):.4f}")
print(f"FT   - ROUGE-1: {avg_rouge(ft_rouge, 'rouge1'):.4f}")
print(f"Base - ROUGE-2: {avg_rouge(base_rouge, 'rouge2'):.4f}")
print(f"FT   - ROUGE-2: {avg_rouge(ft_rouge, 'rouge2'):.4f}")
print(f"Base - ROUGE-L: {avg_rouge(base_rouge, 'rougeL'):.4f}")
print(f"FT   - ROUGE-L: {avg_rouge(ft_rouge, 'rougeL'):.4f}")

print("\n=== BLEU Comparison ===")
print(f"Base BLEU: {sum(base_bleu)/len(base_bleu):.4f}")
print(f"FT   BLEU: {sum(ft_bleu)/len(ft_bleu):.4f}")
