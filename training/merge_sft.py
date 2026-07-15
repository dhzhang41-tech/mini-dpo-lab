from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os


# =====================
# Config
# =====================

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"


sft_adapter_path = (
    "results/exp002_alpaca_sft_v3/adapter"
)


output_path = (
    "results/exp003_sft_merged"
)


# =====================
# Load tokenizer
# =====================

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name
)


# =====================
# Load base model
# =====================

print("Loading base model...")


model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


# =====================
# Load SFT LoRA
# =====================

print("Loading SFT adapter...")


model = PeftModel.from_pretrained(
    model,
    sft_adapter_path
)


# =====================
# Merge
# =====================

print("Merging LoRA adapter...")


model = model.merge_and_unload()


# =====================
# Save
# =====================

os.makedirs(
    output_path,
    exist_ok=True
)


print("Saving merged model...")


model.save_pretrained(
    output_path
)


tokenizer.save_pretrained(
    output_path
)


print(
    "SFT merge completed!"
)