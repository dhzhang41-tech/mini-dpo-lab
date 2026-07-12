import json


input_file = "data/sft_dataset.json"


output_file = "data/sft_chat.json"


with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


formatted_data = []


for item in data:

    text = (
        "<|im_start|>user\n"
        + item["instruction"]
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + item["output"]
        + "\n<|im_end|>"
    )

    formatted_data.append(
        {
            "text": text
        }
    )


with open(output_file, "w", encoding="utf-8") as f:
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