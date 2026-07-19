import json
import re
import os


# =========================
# Config
# =========================

RESULT_DIR = (
    "evaluation/outputs/gsm8k"
)


MODELS = [
    "base",
    "sft",
    "dpo",
    "reasoning_dpo"
]



# =========================
# Extract answer
# =========================

def extract_number(text):

    """
    Extract final numerical answer
    """

    # 优先找 Final Answer

    patterns = [
        r"Final Answer:\s*\$?([0-9]+(?:\.[0-9]+)?)",
        r"####\s*([0-9]+(?:\.[0-9]+)?)",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1)



    # fallback:
    # 找最后一个数字

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )


    if numbers:
        return numbers[-1]


    return None



# =========================
# Ground truth
# =========================

def extract_ground_truth(answer):

    """
    GSM8K answer format:

    ... reasoning
    #### 42

    """

    match = re.search(
        r"####\s*([0-9]+(?:\.[0-9]+)?)",
        answer
    )


    if match:

        return match.group(1)


    return None



# =========================
# Evaluate
# =========================

def evaluate_model(model_name):


    path = os.path.join(
        RESULT_DIR,
        f"{model_name}.json"
    )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)



    correct = 0


    total = len(data)


    wrong_cases = []



    for item in data:


        gt = extract_ground_truth(
            item["ground_truth"]
        )


        pred = extract_number(
            item["output"]
        )


        if pred == gt:

            correct += 1

        else:

            wrong_cases.append(
                {
                    "question":
                        item["question"],

                    "ground_truth":
                        gt,

                    "prediction":
                        pred,

                    "output":
                        item["output"]
                }
            )



    accuracy = correct / total



    return {
        "model": model_name,

        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "wrong_cases":
            wrong_cases
    }



# =========================
# Main
# =========================


if __name__ == "__main__":


    results=[]


    for model in MODELS:

        print(
            f"Evaluating {model}..."
        )


        result = evaluate_model(
            model
        )

        results.append(
            result
        )


    print("\n====================")
    print("GSM8K Accuracy")
    print("====================")


    for r in results:

        print(
            f"{r['model']:20s}"
            f"{r['correct']}/{r['total']} "
            f"{r['accuracy']:.2%}"
        )



    with open(
        "evaluation/outputs/gsm8k_accuracy.json",
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
        "\nSaved:"
        " evaluation/outputs/gsm8k_accuracy.json"
    )