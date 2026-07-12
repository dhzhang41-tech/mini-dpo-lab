# Mini-DPO Lab

A lightweight framework for studying LLM alignment with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Motivation

Large Language Models (LLMs) can generate fluent responses but may not always follow human instructions or preferences.

This project aims to build a small-scale LLM alignment pipeline to study how post-training methods improve model behavior.

## Research Questions

1. How does Supervised Fine-Tuning improve instruction following ability?
2. Does DPO further improve human preference alignment compared with SFT?
3. How do DPO hyperparameters influence model behavior?



## Pipeline

Base Model

↓

Supervised Fine-Tuning (SFT)

↓

Direct Preference Optimization (DPO)

↓

Evaluation & Visualization

## Tech Stack

- PyTorch
- HuggingFace Transformers
- HuggingFace TRL
- PEFT (LoRA)
- Qwen2.5 Models



## Experiments

Planned experiments:

- Base model vs SFT model
- SFT model vs DPO model
- DPO hyperparameter analysis

