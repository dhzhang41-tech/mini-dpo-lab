from datasets import load_dataset
import json
import os


output_file = "data/processed/dpo_train.json"

sample_size = 1000


print("Loading UltraFeedback...")


dataset = load_dataset(
    "openbmb/UltraFeedback",
    split="train"
)


print(dataset)


dpo_data = []


for item in dataset.select(range(sample_size)):

    prompt = item["instruction"]

    completions = item["completions"]


    # 按人工评分排序
    completions = sorted(
        completions,
        key=lambda x: float(x["overall_score"])
    )


    rejected = completions[0]["response"]

    chosen = completions[-1]["response"]


    dpo_data.append(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    )


print(
    "Generated samples:",
    len(dpo_data)
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dpo_data,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"Saved to {output_file}"
)