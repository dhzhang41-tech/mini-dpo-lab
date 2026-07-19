import json
import re


DPO_FILE = (
    "evaluation/outputs/gsm8k/dpo.json"
)

REASONING_FILE = (
    "evaluation/outputs/gsm8k/reasoning_dpo.json"
)



def extract_number(text):

    patterns = [
        r"Final Answer:\s*\$?([0-9]+(?:\.[0-9]+)?)",
        r"####\s*([0-9]+(?:\.[0-9]+)?)"
    ]


    for p in patterns:

        m = re.search(
            p,
            text,
            re.IGNORECASE
        )

        if m:
            return m.group(1)


    nums = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )


    if nums:
        return nums[-1]


    return None



def extract_gt(text):

    m = re.search(
        r"####\s*([0-9]+(?:\.[0-9]+)?)",
        text
    )

    if m:
        return m.group(1)

    return None



with open(
    DPO_FILE,
    "r",
    encoding="utf-8"
) as f:

    dpo=json.load(f)



with open(
    REASONING_FILE,
    "r",
    encoding="utf-8"
) as f:

    rdpo=json.load(f)



gain_cases=[]

loss_cases=[]



for d,r in zip(dpo,rdpo):


    gt=extract_gt(
        d["ground_truth"]
    )


    dpo_ans=extract_number(
        d["output"]
    )


    r_ans=extract_number(
        r["output"]
    )



    dpo_correct = (
        dpo_ans == gt
    )


    r_correct = (
        r_ans == gt
    )


    if (
        not dpo_correct
        and r_correct
    ):

        gain_cases.append(
            {
                "question":
                    d["question"],

                "ground_truth":
                    gt,

                "dpo":
                    d["output"],

                "reasoning_dpo":
                    r["output"]
            }
        )



    if (
        dpo_correct
        and not r_correct
    ):

        loss_cases.append(
            {
                "question":
                    d["question"],

                "ground_truth":
                    gt,

                "dpo":
                    d["output"],

                "reasoning_dpo":
                    r["output"]
            }
        )



print("====================")
print("DPO -> Reasoning-DPO gains")
print(
    len(gain_cases)
)


print("====================")
print("DPO -> Reasoning-DPO losses")
print(
    len(loss_cases)
)



with open(
    "evaluation/outputs/gain_cases.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        gain_cases,
        f,
        ensure_ascii=False,
        indent=2
    )



with open(
    "evaluation/outputs/loss_cases.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        loss_cases,
        f,
        ensure_ascii=False,
        indent=2
    )



print(
    "Saved analysis files"
)