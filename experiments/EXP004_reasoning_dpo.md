# EXP004 Reasoning-DPO

## Goal

Test whether preference optimization on reasoning traces improves mathematical reasoning ability.

## Model

Base:

Qwen2.5-1.5B-Instruct

Training:

SFT:

Alpaca instruction tuning

DPO:

standard preference optimization

Reasoning-DPO:

GSM8K reasoning preference pairs

## Dataset

GSM8K

Generated pairs:

500

Validated:

404

## Training

Epoch:

1

LoRA:

r=8

beta:

0.1

## Results

|Model|Accuracy|

|-|-|

|Base|29%|

|SFT|32%|

|DPO|32%|

|Reasoning-DPO|33%|

## Observation

Reasoning-DPO slightly improves GSM8K accuracy.

Qualitative analysis shows improvements mainly come from:

- completion quality

- reduced incomplete generation

- limited reasoning correction

Main failure:

- verbose reasoning

- instruction leakage

- reasoning hallucination