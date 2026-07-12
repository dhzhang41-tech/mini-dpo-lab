from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from trl import (
    SFTTrainer,
    SFTConfig
)

from peft import (
    LoraConfig,
    get_peft_model
)

import torch


model_name = "Qwen/Qwen2.5-1.5B-Instruct"


# =====================
# tokenizer
# =====================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# =====================
# model
# =====================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


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



# LoRA will be injected by SFTTrainer


# =====================
# dataset
# =====================

dataset = load_dataset(
    "json",
    data_files="data/sft_chat.json"
)


# =====================
# training config
# =====================

training_args = SFTConfig(
    output_dir="results/sft",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    fp16=True,

    dataset_text_field="text"
)


# =====================
# trainer
# =====================

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    processing_class=tokenizer,

    peft_config=lora_config
)


trainer.train()


# =====================
# save
# =====================

trainer.model.save_pretrained(
    "results/sft_adapter"
)

tokenizer.save_pretrained(
    "results/sft_adapter"
)


print("SFT training finished!")