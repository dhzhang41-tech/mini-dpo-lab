from datasets import load_dataset
import json
from pathlib import Path


output_path = Path(
    "data/processed/sft_train.json"
)


output_path.parent.mkdir(
    exist_ok=True
)


dataset = load_dataset(
    "yahma/alpaca-cleaned"
)["train"]


formatted_data = []


for item in dataset:

    instruction = item["instruction"]

    input_text = item["input"]

    output = item["output"]


    if input_text.strip():

        user_text = (
            instruction
            + "\n"
            + input_text
        )

    else:

        user_text = instruction


    text = (
        "<|im_start|>user\n"
        + user_text
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + output
        + "\n<|im_end|>"
    )


    formatted_data.append(
        {
            "text": text
        }
    )


with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        formatted_data,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    "Saved:",
    len(formatted_data),
    "samples"
)