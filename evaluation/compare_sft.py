import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =================
# Config
# =================

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

adapter_path = (
    "results/exp002_alpaca_sft_v3/adapter"
)

prompt_file = (
    "evaluation/prompts.json"
)

output_file = (
    "evaluation/outputs/compare_results.json"
)


# =================
# Load prompts
# =================

with open(
    prompt_file,
    "r",
    encoding="utf-8"
) as f:

    prompts = json.load(f)



# =================
# Tokenizer
# =================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



# =================
# Base model
# =================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)



# =================
# SFT model
# =================

print("Loading SFT adapter...")

sft_model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)


sft_model.eval()
base_model.eval()


print("Models loaded successfully")



# =================
# Generate
# =================

def generate(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )


    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )



# =================
# Evaluation
# =================

results = []


for item in prompts:

    print(
        f"\nTesting {item['category']}"
    )


    prompt = item["prompt"]


    base_output = generate(
        base_model,
        prompt
    )


    sft_output = generate(
        sft_model,
        prompt
    )


    results.append(
        {
            "id": item["id"],
            "category": item["category"],
            "prompt": prompt,
            "base_output": base_output,
            "sft_output": sft_output
        }
    )



# =================
# Save
# =================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nEvaluation finished!"
)

print(
    f"Saved to {output_file}"
)