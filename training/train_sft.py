from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from trl import (
    SFTTrainer,
    SFTConfig
)

from peft import LoraConfig

import torch
import os


# =====================
# Experiment config
# =====================

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

output_dir = "results/exp002_alpaca_sft/checkpoints"

adapter_dir = "results/exp002_alpaca_sft/adapter"


# =====================
# Tokenizer
# =====================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# =====================
# Model
# =====================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.use_cache = False

# =====================
# LoRA config
# =====================

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# =====================
# Dataset
# =====================

dataset = load_dataset(
    "json",
    data_files="data/processed/sft_train.json"
)


dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)


print(dataset)


# =====================
# Training config
# =====================

training_args = SFTConfig(

    output_dir="results/exp002_alpaca_sft/checkpoints",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=10,


    save_strategy="steps",

    save_steps=500,

    save_total_limit=3,


    eval_strategy="epoch",


    gradient_checkpointing=True,


    fp16=True,


    max_seq_length=512,


    dataset_text_field="text"
)


# =====================
# Trainer
# =====================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"],

    processing_class=tokenizer,

    peft_config=lora_config
)


# =====================
# Training
# =====================

checkpoint_exists = False


if os.path.exists(output_dir):

    checkpoints = [
        x for x in os.listdir(output_dir)
        if x.startswith("checkpoint")
    ]

    if len(checkpoints) > 0:
        checkpoint_exists = True



if checkpoint_exists:

    print("Resume from checkpoint")

    trainer.train(
        resume_from_checkpoint=True
    )

else:

    print("Start new training")

    trainer.train()



# =====================
# Save adapter
# =====================

trainer.model.save_pretrained(
    adapter_dir
)


tokenizer.save_pretrained(
    adapter_dir
)


print("EXP002 Alpaca SFT finished!")