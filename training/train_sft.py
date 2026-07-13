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


# =====================
# Experiment config
# =====================

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

output_dir = "results/exp002_alpaca_sft_v3/checkpoints"

adapter_dir = "results/exp002_alpaca_sft_v3/adapter"


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

# disable cache during training
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

    output_dir=output_dir,


    # full training epochs
    num_train_epochs=3,


    # TEST ONLY
    # remove this line after validation
 


    per_device_train_batch_size=1,


    gradient_accumulation_steps=4,


    learning_rate=2e-4,


    logging_steps=10,


    # checkpoint
    save_strategy="steps",

    save_steps=500,

    save_total_limit=3,


    # evaluation
    eval_strategy="epoch",


    # fp16
    fp16=True,


    # avoid long sequence OOM
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
# Train
# =====================

print("Start new EXP002-v3 training")

trainer.train()


# =====================
# Save LoRA adapter
# =====================

import gc

gc.collect()

torch.cuda.empty_cache()

trainer.model.save_pretrained(
    adapter_dir
)


tokenizer.save_pretrained(
    adapter_dir
)


print("EXP002-v3 Alpaca SFT finished!")