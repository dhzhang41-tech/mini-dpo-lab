from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from trl import (
    DPOTrainer,
    DPOConfig
)

from peft import LoraConfig

import torch
import os


# =====================
# Config
# =====================

# 与EXP003保持一致
# 保证公平比较：都是从同一个SFT模型开始

model_path = (
    "results/exp003_sft_merged"
)


# EXP004输出目录

output_dir = (
    "results/exp004_reasoning_dpo_adapter"
)


# 使用Reasoning Preference数据

data_path = (
    "data/processed/reasoning_dpo_train.json"
)



# =====================
# Tokenizer
# =====================

print("Loading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    model_path
)


# =====================
# Model
# =====================

print("Loading SFT merged model...")


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


model.config.use_cache = False



# =====================
# Dataset
# =====================

print("Loading Reasoning-DPO dataset...")


dataset = load_dataset(
    "json",
    data_files=data_path,
    split="train"
)


print(dataset)



# =====================
# LoRA
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
# Training config
# =====================

training_args = DPOConfig(

    output_dir=output_dir,


    # 保持与EXP003一致

    num_train_epochs=1,


    per_device_train_batch_size=1,


    gradient_accumulation_steps=4,


    learning_rate=5e-5,


    logging_steps=10,


    save_strategy="steps",

    save_steps=200,

    save_total_limit=2,


    fp16=True,


    max_length=512,

    max_prompt_length=256,


    beta=0.1
)



# =====================
# Trainer
# =====================

trainer = DPOTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config

)



# =====================
# Train
# =====================

print(
    "Start EXP004 Reasoning-DPO training"
)


trainer.train()



# =====================
# Save adapter
# =====================

trainer.model.save_pretrained(
    output_dir
)


tokenizer.save_pretrained(
    output_dir
)


print(
    "EXP004 Reasoning-DPO finished!"
)