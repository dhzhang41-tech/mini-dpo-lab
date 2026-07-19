import json
import os
import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =========================
# Config
# =========================

BASE_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


SFT_ADAPTER = (
    "results/exp002_alpaca_sft_v3/adapter"
)


SFT_MERGED = (
    "results/exp003_sft_merged"
)


DPO_ADAPTER = (
    "results/exp003_dpo_adapter"
)


REASONING_DPO_ADAPTER = (
    "results/exp004_reasoning_dpo_adapter"
)


TEST_FILE = (
    "data/raw/gsm8k/test.jsonl"
)


NUM_TEST = 100


OUTPUT_DIR = (
    "evaluation/outputs/gsm8k"
)



# =========================
# Load data
# =========================

def load_test_data():

    data = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(
                json.loads(line)
            )


    return data[:NUM_TEST]



# =========================
# Load model
# =========================

def load_base_model():

    print("Loading Base model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    return model, tokenizer



def load_sft_model():

    print("Loading SFT model...")


    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )


    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )


    model = PeftModel.from_pretrained(
        base,
        SFT_ADAPTER
    )


    model.eval()


    return model, tokenizer



def load_dpo_model():

    print("Loading DPO model...")


    tokenizer = AutoTokenizer.from_pretrained(
        SFT_MERGED
    )


    base = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED,
        torch_dtype=torch.float16,
        device_map="auto"
    )


    model = PeftModel.from_pretrained(
        base,
        DPO_ADAPTER
    )


    model.eval()


    return model, tokenizer



def load_reasoning_dpo_model():

    print("Loading Reasoning-DPO model...")


    tokenizer = AutoTokenizer.from_pretrained(
        SFT_MERGED
    )


    base = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED,
        torch_dtype=torch.float16,
        device_map="auto"
    )


    model = PeftModel.from_pretrained(
        base,
        REASONING_DPO_ADAPTER
    )


    model.eval()


    return model, tokenizer



# =========================
# Generate
# =========================

def generate(
    model,
    tokenizer,
    question
):

    prompt = f"""
Solve this math problem step by step.

Problem:

{question}

Answer:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(
        model.device
    )


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )


    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


    return answer



# =========================
# Evaluate one model
# =========================

def evaluate_model(
    name,
    loader
):

    model, tokenizer = loader()


    data = load_test_data()


    results = []


    for item in tqdm(
        data,
        desc=name
    ):

        output = generate(
            model,
            tokenizer,
            item["question"]
        )


        results.append(
            {
                "question":
                    item["question"],

                "ground_truth":
                    item["answer"],

                "output":
                    output
            }
        )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    path = os.path.join(
        OUTPUT_DIR,
        f"{name}.json"
    )


    with open(
        path,
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
        f"Saved {path}"
    )


    # release GPU

    del model

    torch.cuda.empty_cache()



# =========================
# Main
# =========================

if __name__ == "__main__":


    evaluate_model(
        "base",
        load_base_model
    )


    evaluate_model(
        "sft",
        load_sft_model
    )


    evaluate_model(
        "dpo",
        load_dpo_model
    )


    evaluate_model(
        "reasoning_dpo",
        load_reasoning_dpo_model
    )


    print(
        "GSM8K evaluation finished!"
    )