# Experiment 002: Alpaca LoRA SFT Baseline

## Research Question

Can supervised fine-tuning improve the instruction-following ability of Qwen2.5-1.5B-Instruct?

## Objective

Establish a complete SFT baseline using a real instruction-following dataset.

This experiment serves as the foundation for future preference optimization experiments such as DPO.

## Model

Base Model:

Qwen2.5-1.5B-Instruct

Fine-tuning Method:

LoRA-based Supervised Fine-Tuning

## Dataset

Dataset:

yahma/alpaca-cleaned

Samples:

51,760 instruction-response pairs

Data Processing:

Raw Alpaca format is converted into Qwen chat template:

<|im_start|>user

instruction

<|im_end|>

<|im_start|>assistant

response

<|im_end|>

## Training Configuration

Method:

LoRA SFT

Epochs:

3

Learning rate:

TBD

Batch size:

TBD

Random seed:

TBD

## Evaluation

Compare:

1. Base Qwen2.5-1.5B-Instruct

2. SFT model

Evaluation dimensions:

- Instruction following

- Response relevance

- Factual consistency

## Expected Outcome

The SFT model should better follow user instructions compared with the base model.

## Future Extension

This experiment provides the supervised fine-tuned model for later DPO experiments.

Pipeline:

Base Model

↓

SFT

↓

Preference Optimization (DPO)

↓

Alignment Model