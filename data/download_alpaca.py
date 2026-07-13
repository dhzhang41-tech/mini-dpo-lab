from datasets import load_dataset


dataset = load_dataset(
    "yahma/alpaca-cleaned"
)


print(dataset)

print("\nFirst example:")
print(dataset["train"][0])