!pip install -q transformers accelerate datasets sentencepiece

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

MODEL_NAME = "google/gemma-3-4b-it"
OUT_DIR    = "model_dir"

# === Загружаем dataset ===
ds = load_dataset("json", data_files="/content/gdrive/MyDrive/baq_clean.jsonl")
print(ds)

# === Токенизатор ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Tokenization + labels ===
def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # labels = input_ids (только так Gemma вернет loss!)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

# === Data collator (важно!) ===
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # Causal LM режим
)

# === Загружаем модель ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === Training arguments ===
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_steps=300,
    logging_steps=50,
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=collator
)

# === Train ===
trainer.train(resume_from_checkpoint=True)

# === Save model ===
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✔ Pretraining finished!")
