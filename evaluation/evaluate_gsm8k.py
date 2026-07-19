import json
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# ==========================
# Config
# ==========================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


SFT_MODEL = (
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



# ==========================
# Load data
# ==========================

def load_test_data(path):

    data=[]

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(json.loads(line))


    return data[:NUM_TEST]



# ==========================
# Load model
# ==========================

def load_model(model_path, adapter=None):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )


    if adapter:

        model = PeftModel.from_pretrained(
            model,
            adapter
        )


    model.eval()


    return model, tokenizer



# ==========================
# Generate
# ==========================

def generate_answer(
    model,
    tokenizer,
    question
):

    prompt = f"""
Solve the following math problem.

Explain your reasoning step by step.

Problem:

{question}
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False
        )


    result = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


    return result



# ==========================
# Main
# ==========================


if __name__ == "__main__":


    data = load_test_data(
        TEST_FILE
    )


    print(
        f"Loaded {len(data)} test samples"
    )


    print("Loading Base...")
    base_model, base_tokenizer = load_model(
        BASE_MODEL
    )


    print("Loading SFT...")
    sft_model, sft_tokenizer = load_model(
        SFT_MODEL
    )


    print("Loading DPO...")
    dpo_model, dpo_tokenizer = load_model(
        SFT_MODEL,
        DPO_ADAPTER
    )


    print("Loading Reasoning-DPO...")
    rdpo_model, rdpo_tokenizer = load_model(
        SFT_MODEL,
        REASONING_DPO_ADAPTER
    )


    results=[]


    for idx,item in enumerate(
        tqdm(data)
    ):

        question=item["question"]

        results.append(
            {
                "id":idx,

                "question":question,

                "base":
                generate_answer(
                    base_model,
                    base_tokenizer,
                    question
                ),

                "sft":
                generate_answer(
                    sft_model,
                    sft_tokenizer,
                    question
                ),

                "dpo":
                generate_answer(
                    dpo_model,
                    dpo_tokenizer,
                    question
                ),

                "reasoning_dpo":
                generate_answer(
                    rdpo_model,
                    rdpo_tokenizer,
                    question
                )
            }
        )


    with open(
        "evaluation/gsm8k_results.json",
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