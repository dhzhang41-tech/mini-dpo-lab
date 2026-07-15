from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

import torch
import json
import os


# =====================
# Config
# =====================

base_model_name = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


sft_adapter_path = (
    "results/exp002_alpaca_sft_v3/adapter"
)


sft_merged_path = (
    "results/exp003_sft_merged"
)


dpo_adapter_path = (
    "results/exp003_dpo_adapter"
)


prompt_file = (
    "evaluation/prompts.json"
)


output_file = (
    "evaluation/outputs/dpo_compare_results.json"
)


# =====================
# Tokenizer
# =====================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name
)


# =====================
# Load Base
# =====================

print("Loading base model...")


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


# =====================
# Load SFT
# =====================

print("Loading SFT model...")


sft_base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


sft_model = PeftModel.from_pretrained(
    sft_base,
    sft_adapter_path
)


# =====================
# Load DPO
# =====================

print("Loading DPO model...")


dpo_base = AutoModelForCausalLM.from_pretrained(
    sft_merged_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


dpo_model = PeftModel.from_pretrained(
    dpo_base,
    dpo_adapter_path
)


print("All models loaded!")


# =====================
# Generate
# =====================

def generate(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )


    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return text



# =====================
# Load prompts
# =====================

with open(
    prompt_file,
    "r",
    encoding="utf-8"
) as f:

    prompts = json.load(f)



# =====================
# Evaluation
# =====================

results = []


for item in prompts:

    print(
        "Testing:",
        item["category"]
    )


    prompt = item["prompt"]


    result = {

        "id": item["id"],

        "category": item["category"],

        "prompt": prompt,


        "base_output":
            generate(
                base_model,
                prompt
            ),


        "sft_output":
            generate(
                sft_model,
                prompt
            ),


        "dpo_output":
            generate(
                dpo_model,
                prompt
            )
    }


    results.append(result)



# =====================
# Save
# =====================

os.makedirs(
    "evaluation/outputs",
    exist_ok=True
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    "Evaluation finished!"
)

print(
    "Saved:",
    output_file
)