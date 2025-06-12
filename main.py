import os
import json
import torch
import nltk
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoModelForSequenceClassification, AutoTokenizer,
    T5Tokenizer, T5ForConditionalGeneration,
    AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from tqdm import tqdm

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

# Set random seed
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download NLTK punkt
nltk.download("punkt")

# Load and tokenize dataset
def load_jsonl_dataset(file_path):
    data = {"prompt": [], "response": []}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("prompt") and entry.get("response"):
                        data["prompt"].append(entry["prompt"])
                        data["response"].append(entry["response"])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
        if not data["prompt"]:
            raise ValueError("Dataset is empty or contains no valid prompt-response pairs.")
        return Dataset.from_dict(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {file_path} not found.")

# Tokenization function for SFT
def tokenize_function(examples, max_length=512):
    inputs = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# SFT (Supervised Fine-Tuning)
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

raw_dataset = load_jsonl_dataset("/kaggle/input/lawdata/law_stackexchange_data.jsonl")
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
tokenized_datasets = DatasetDict({
    split: dataset[split].map(lambda x: tokenize_function(x), batched=True, remove_columns=["prompt", "response"])
    for split in ["train", "test"]
})

training_args = TrainingArguments(
    output_dir="./gpt2_sft",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=500,
    learning_rate=5e-5,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)
trainer.train()
model.save_pretrained("./gpt2_sft")
tokenizer.save_pretrained("./gpt2_sft")

# Reward Model Training
sft_model = GPT2LMHeadModel.from_pretrained("./gpt2_sft").to(device)
sft_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_sft")
sft_pipe = pipeline("text-generation", model=sft_model, tokenizer=sft_tokenizer, max_length=256, device=0 if torch.cuda.is_available() else -1)

scorer_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
scorer_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def score_response_t5(prompt, response):
    input_text = f"Rate the legal helpfulness of this answer to the question '{prompt}' on a scale from 1 to 10 (1 = unhelpful, 10 = very helpful): {response}"
    inputs = scorer_tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = scorer_model.generate(**inputs, max_new_tokens=10)
    decoded = scorer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Flan-T5 output: {decoded}")
    for token in decoded.split():
        if token.isdigit() and 1 <= int(token) <= 10:
            return int(token)
    return 1

ranked_data = []
for ex in raw_dataset.select(range(1500)):
    prompt = ex["prompt"]
    outputs = sft_pipe(f"Prompt: {prompt}\nResponse:", num_return_sequences=3, do_sample=True, temperature=0.9, top_p=0.9)
    responses = [o["generated_text"].split("Response:")[-1].strip() for o in outputs]
    scores = [score_response_t5(prompt, r) for r in responses]
    print(f"Scores for prompt '{prompt[:50]}...': {scores}")
    for i in range(len(responses)):
        for j in range(len(responses)):
            if scores[i] > scores[j]:
                ranked_data.append({"prompt": prompt, "chosen": responses[i], "rejected": responses[j]})

if not ranked_data:
    print("Error: ranked_data is empty. Using fallback scoring with reward_model.")
    reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def score_response_fallback(prompt, response):
        input_text = f"Prompt: {prompt}\nResponse: {response}"
        inputs = reward_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            return reward_model(**inputs).logits.item()
    for ex in raw_dataset.select(range(1500)):
        prompt = ex["prompt"]
        outputs = sft_pipe(f"Prompt: {prompt}\nResponse:", num_return_sequences=3, do_sample=True, temperature=0.9, top_p=0.9)
        responses = [o["generated_text"].split("Response:")[-1].strip() for o in outputs]
        scores = [score_response_fallback(prompt, r) for r in responses]
        print(f"Fallback scores for prompt '{prompt[:50]}...': {scores}")
        for i in range(len(responses)):
            for j in range(len(responses)):
                if scores[i] > scores[j]:
                    ranked_data.append({"prompt": prompt, "chosen": responses[i], "rejected": responses[j]})

if not ranked_data:
    raise ValueError("Fallback scoring also produced empty ranked_data. Check dataset or SFT model outputs.")

# Save ranked_data to JSONL for DPO
with open("./dpo_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in ranked_data:
        json.dump(item, f)
        f.write("\n")

reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class RewardDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        chosen = self.tokenizer(f"Prompt: {item['prompt']}\nResponse: {item['chosen']}",
                               truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        rejected = self.tokenizer(f"Prompt: {item['prompt']}\nResponse: {item['rejected']}",
                                 truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids_chosen": chosen["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected["attention_mask"].squeeze(0)
        }

reward_dataset = RewardDataset(ranked_data, reward_tokenizer)
reward_dataloader = DataLoader(reward_dataset, batch_size=2, shuffle=True)

optimizer = AdamW(reward_model.parameters(), lr=1e-5)
reward_model.train()
for epoch in range(5):
    total_loss = 0
    for batch in reward_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        chosen_outputs = reward_model(batch["input_ids_chosen"], attention_mask=batch["attention_mask_chosen"]).logits
        rejected_outputs = reward_model(batch["input_ids_rejected"], attention_mask=batch["attention_mask_rejected"]).logits
        loss = -torch.nn.functional.logsigmoid(chosen_outputs - rejected_outputs).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(reward_dataloader)}")

reward_model.save_pretrained("./reward_model")

# DPO Training
dpo_model = AutoModelForCausalLM.from_pretrained("./gpt2_sft").to(device)
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Enable gradient checkpointing
dpo_model.gradient_checkpointing_enable()
base_model.gradient_checkpointing_enable()

# Apply LoRA
dpo_model = get_peft_model(dpo_model, lora_config)

# Load DPO dataset
dpo_dataset = load_dataset("json", data_files={"train": "./dpo_dataset.jsonl"}, split="train")

# Tokenize DPO dataset
MAX_LENGTH = 1024

def tokenize_dpo(example):
    prompt = tokenizer(example["prompt"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    chosen = tokenizer(example["chosen"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    rejected = tokenizer(example["rejected"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    return {
        "prompt_input_ids": prompt["input_ids"],
        "prompt_attention_mask": prompt["attention_mask"],
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"]
    }

tokenized_dpo_dataset = dpo_dataset.map(tokenize_dpo, batched=False)
tokenized_dpo_dataset.set_format(
    type="torch",
    columns=[
        "prompt_input_ids", "prompt_attention_mask",
        "chosen_input_ids", "chosen_attention_mask",
        "rejected_input_ids", "rejected_attention_mask"
    ]
)

# DPO training config
dpo_config = DPOConfig(
    output_dir="./gpt2_dpo",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    logging_dir="./gpt2_dpo/logs",
    save_total_limit=2,
    report_to="none",
    beta=0.1,
    remove_unused_columns=False
)

# DPO Trainer
trainer = DPOTrainer(
    model=dpo_model,
    ref_model=base_model,
    args=dpo_config,
    train_dataset=tokenized_dpo_dataset,
    processing_class=tokenizer
)

# Train DPO
trainer.train()

# Save DPO model
dpo_model.save_pretrained("./gpt2_dpo")
tokenizer.save_pretrained("./gpt2_dpo")