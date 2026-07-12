from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "Qwen/Qwen2.5-1.5B-Instruct"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


print("Model loaded!")


prompt = "Explain reinforcement learning in simple words."


inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=200
)


response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print("\n===== Response =====")
print(response)