import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training



# Load full dataset
dataset_path = '/home/ubuntu/finbuddy/Data/final_curated_finetune_data_deduped.jsonl'
full_dataset = load_dataset('json', data_files=dataset_path, split='train')

# Split into train and validation sets (e.g., 90% train / 10% validation)
dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

#  Model & tokenizer setup
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

#  LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral uses these
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Format instruction dataset
def formatting_func(example):
    prompt = f"{example['instruction']}\n"
    inputs = tokenizer(prompt, truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(example['output'], truncation=True, padding='max_length', max_length=512)
    return {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "labels": outputs['input_ids']
    }

tokenized_train = train_dataset.map(formatting_func, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(formatting_func, remove_columns=eval_dataset.column_names)


#  Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#  Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_mistral_final",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    max_grad_norm=1.0,
    fp16=True,
    bf16=False,
    report_to="none",
    logging_dir="./logs",
    save_total_limit=2
)


#  Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator
)


#  Train
trainer.train()

#  Save final model
model.save_pretrained("/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter")
tokenizer.save_pretrained("/home/ubuntu/finbuddy/Models/final_finetuned_mistral_lora_adapter")
