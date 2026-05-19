from transformers import TrainingArguments
from transformers import BitsAndBytesConfig

from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM,  DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
#
#########4


# Print the first training example



# Load tokenizer and model
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=False,token="")
tokenizer.padding_side = "right"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=""


)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,#16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],

)

model = get_peft_model(model, peft_config)

def preprocess(batch):
    input_ids_list = []
    attention_masks = []
    labels_list = []
    token_type_ids_list = []

    for doc, summ in zip(batch["document"], batch["summary"]):
        doc = str(doc)
        summ = str(summ)

        messages = [
            {"role": "user", "content": f"Write a concise summary in 1 sentence in Kazakh:\n\nArticle: {doc}"},
            {"role": "assistant", "content": summ}
        ]

        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 👉 токенизация ОДИН раз
        full = tokenizer(
            full_prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        
        assistant_start = full_prompt.find("<start_of_turn>assistant")

        prefix_text = full_prompt[:assistant_start]

        prefix_ids = tokenizer(
            prefix_text,
            add_special_tokens=False
        )["input_ids"]

        cut = len(prefix_ids)

        # защита от выхода за границы
        if cut >= len(input_ids):
            cut = len(input_ids) - 1

        labels = [-100] * cut + input_ids[cut:]
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        # token_type_ids (для gemma3 обязательно)
        token_type_ids = [0] * cut + [1] * (len(input_ids) - cut)

        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)
        token_type_ids_list.append(token_type_ids)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
        "token_type_ids": token_type_ids_list
    }


# Preprocessing
max_length = 512
def preprocess_old(example):
    doc = str(example.get("document", ""))
    summ = str(example.get("summary", ""))
    messages = [
        {
            "role": "user",
            "content": f"Write a concise summary in 1 sentence in Kazakh:\n\nArticle: {doc}"
        },
        {
            "role": "assistant",
            "content": summ
        }
    ]

    
    messages_user = [messages[0]]

    prompt_only = tokenizer.apply_chat_template(
        messages_user,
        tokenize=False,
        add_generation_prompt=True
    )

    # 👉 2. полный диалог
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # 👉 3. токенизация
    full = tokenizer(
        full_prompt,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    prompt = tokenizer(
        prompt_only,
        max_length=512,
        truncation=True,
        add_special_tokens=False
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # 👉 4. masking
    cut = min(len(prompt["input_ids"]), len(input_ids))

    labels = [-100] * cut + input_ids[cut:]
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]




    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = train_dataset.select(range(20000))
train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    batch_size=64,
    remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(#
    preprocess,
    batched=True,
    batch_size=64,
    remove_columns=val_dataset.column_names
)












from transformers import TrainingArguments,Trainer

# Training args
training_args = TrainingArguments(
    output_dir="/gemma_sft_v1",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    warmup_steps=50,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    eval_steps=500,
    save_steps=100,
    eval_strategy="steps",
    report_to="none",
    logging_first_step=True,
    

    optim="paged_adamw_8bit",
    remove_unused_columns=False

)

# Trainer

trainer = Trainer(#
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)
