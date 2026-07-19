import json
import re
import os


INPUT_FILE = (
    "data/intermediate/reasoning_pairs.json"
)

OUTPUT_FILE = (
    "data/processed/reasoning_dpo_train.json"
)



def extract_final_answer(text):

    pattern = r"Final Answer:\s*(.*)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None



def normalize_answer(answer):

    if answer is None:
        return None


    answer = answer.lower()

    answer = answer.replace(
        "$",
        ""
    )

    answer = answer.replace(
        ",",
        ""
    )

    answer = answer.strip()


    return answer



def validate_pair(item):


    chosen = item["chosen"]

    rejected = item["rejected"]

    ground_truth = normalize_answer(
        item["ground_truth"]
    )


    chosen_answer = normalize_answer(
        extract_final_answer(chosen)
    )


    rejected_answer = normalize_answer(
        extract_final_answer(rejected)
    )


    # 必须存在答案

    if chosen_answer is None:
        return False


    if rejected_answer is None:
        return False


    # chosen必须正确

    if chosen_answer != ground_truth:
        return False


    # rejected必须错误

    if rejected_answer == ground_truth:
        return False


    return True



if __name__ == "__main__":


    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    print(
        f"Loaded {len(data)} samples"
    )


    filtered = []


    for item in data:


        if validate_pair(item):

            filtered.append(
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                }
            )



    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            filtered,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"Kept {len(filtered)} samples"
    )


    print(
        f"Saved to {OUTPUT_FILE}"
    )