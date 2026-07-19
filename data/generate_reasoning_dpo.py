import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


# =========================
# Load API configuration
# =========================

load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# =========================
# Configuration
# =========================

NUM_SAMPLES = 500

DATA_FILE = "data/raw/gsm8k/train.jsonl"

OUTPUT_FILE = "data/intermediate/reasoning_pairs.json"


# =========================
# Load GSM8K
# =========================

def load_gsm8k(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(
                json.loads(line)
            )

    return data



# =========================
# Save checkpoint
# =========================

def save_results(results):

    os.makedirs(
        "data/intermediate",
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )



# =========================
# DeepSeek API with retry
# =========================

def call_deepseek(prompt):

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )


            return response.choices[0].message.content


        except Exception as e:

            print(
                f"\nAPI request failed ({attempt+1}/3)"
            )

            print(e)


            time.sleep(5)



    raise RuntimeError(
        "DeepSeek API failed after 3 retries"
    )



# =========================
# Generate chosen reasoning
# =========================

def generate_chosen(question):

    prompt = f"""
You are an expert mathematical reasoning teacher.

Solve the following problem carefully.

Requirements:

1. Provide a clear step-by-step reasoning process.
2. Explain important calculations.
3. Make sure the final answer is correct.
4. Do not introduce information that is not in the problem.
5. End with:

Final Answer: <answer>


Problem:

{question}
"""

    return call_deepseek(prompt)



# =========================
# Generate rejected reasoning
# =========================

def generate_rejected(chosen):

    prompt = f"""
You are creating a negative example for mathematical reasoning training.

Below is a correct solution:

{chosen}


Create a plausible but incorrect reasoning process.

Requirements:

1. The solution should look like a real student's mistake.
2. The error must come from the reasoning process.
3. The final answer must be wrong.
4. The mistake should be subtle.

Possible mistakes:

- arithmetic mistake
- wrong interpretation
- missing a step
- incorrect logical assumption


Do NOT:

- introduce new information
- change the problem conditions


End with:

Final Answer: <wrong answer>


Generate only the incorrect solution.
"""

    return call_deepseek(prompt)



# =========================
# Main pipeline
# =========================

if __name__ == "__main__":


    print("Loading GSM8K...")


    dataset = load_gsm8k(
        DATA_FILE
    )


    print(
        f"Loaded {len(dataset)} samples"
    )


    # Load existing checkpoint

    if os.path.exists(
        OUTPUT_FILE
    ):

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)


        print(
            f"Loaded checkpoint: {len(results)} samples"
        )


    else:

        results = []



    start_index = len(results)


    print(
        f"Continue from index {start_index}"
    )


    for item in tqdm(
        dataset[start_index:NUM_SAMPLES]
    ):

        question = item["question"]

        answer = item["answer"]


        ground_truth = (
            answer.split("####")[-1]
            .strip()
        )


        try:

            chosen = generate_chosen(
                question
            )


            rejected = generate_rejected(
                chosen
            )


            results.append(
                {
                    "prompt": question,
                    "ground_truth": ground_truth,
                    "chosen": chosen,
                    "rejected": rejected
                }
            )


            # save immediately

            save_results(
                results
            )


            print(
                "\nSaved checkpoint:",
                len(results)
            )


            print("\n" + "=" * 60)

            print("QUESTION:")
            print(question)

            print("\nCHOSEN:")
            print(chosen)

            print("\nREJECTED:")
            print(rejected)



        except Exception as e:

            print(
                "\nSample failed:"
            )

            print(e)

            print(
                "Skipping this sample..."
            )

            continue



    print(
        f"\nFinished. Total samples: {len(results)}"
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )