from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

adapter_path = "results/sft_adapter"


prompt = "Explain the difference between SFT and DPO."


# =================
# tokenizer
# =================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# =================
# Base model
# =================

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


# =================
# SFT model
# =================

sft_model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)


def generate(model):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )


    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


print("\n===== Base Model =====")
print(generate(base_model))


print("\n===== SFT Model =====")
print(generate(sft_model))