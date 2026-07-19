# mini-dpo-lab

A minimal but complete implementation of an LLM alignment pipeline, including:

 

- Supervised Fine-Tuning (SFT)

- LoRA parameter-efficient fine-tuning

- Direct Preference Optimization (DPO)

- Reasoning-aware Preference Optimization

- Model evaluation and behavior analysis

 

The goal of this project is to understand and reproduce modern LLM post-training workflows under limited computational resources.

# Overview

Modern LLM alignment usually consists of multiple post-training stages:

 

Qwen2.5-1.5B-Instruct

 

↓

 

Supervised Fine-Tuning (SFT)

 

↓

 

Instruction-following Model

 

↓

 

Direct Preference Optimization (DPO)

 

↓

 

Preference-aligned Model

 

↓

 

Reasoning-DPO Investigation

# Models

Base Model:

Qwen/Qwen2.5-1.5B-Instruct

 

Training Strategy:

- Freeze original model parameters

- Train LoRA adapters

- Reduce GPU memory requirements

 

Hardware:

RTX 5070 Laptop GPU (16GB VRAM)

# Experiments

EXP001: Toy SFT Pipeline

 

Objective:

Validate the supervised fine-tuning workflow.

 

EXP002: Alpaca-style SFT

 

Objective:

Improve instruction-following ability through supervised fine-tuning.

 

Observation:

SFT improves instruction following, response formatting, and task-oriented generation.

 

EXP003: Direct Preference Optimization (DPO)

 

Objective:

Apply preference optimization on top of the SFT model.

 

Dataset:

UltraFeedback preference dataset

 

Training format:

prompt, chosen, rejected

 

Configuration:

Epochs: 1

Batch size: 1

Gradient accumulation: 4

Learning rate: 5e-5

Beta: 0.1

 

Main findings:

- SFT improves instruction following.

- DPO improves preference alignment and response consistency.

- SFT and DPO alone do not automatically improve reasoning ability.

# EXP004: Reasoning-DPO

EXP004 investigates whether preference optimization on reasoning trajectories can improve mathematical reasoning ability.

 

Dataset:

GSM8K

 

Pipeline:

GSM8K → Generate reasoning pairs → Validate → DPO training

 

Statistics:

Generated pairs: 500

Validated pairs: 404

 

Training:

Base Model: Qwen2.5-1.5B-Instruct

Method: LoRA + DPO

 

Evaluation:

GSM8K test set (100 samples)

 

Results:

Base: 29%

SFT: 32%

DPO: 32%

Reasoning-DPO: 33%

 

Analysis:

Reasoning-DPO shows a small improvement compared with standard DPO.

# Future Work

- Higher quality reasoning preference datasets

- Better negative reasoning generation

- Larger evaluation

- Compare with GRPO and RL-based reasoning optimization

# References

- InstructGPT: Training language models to follow instructions with human feedback

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- Qwen2.5 Technical Report

- UltraFeedback Dataset

- GSM8K Dataset

