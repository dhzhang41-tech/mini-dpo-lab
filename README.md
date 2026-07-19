# **mini-dpo-lab**



A minimal but complete implementation of an LLM alignment pipeline.



This project reproduces modern LLM post-training workflows under limited computational resources, including:



- Supervised Fine-Tuning (SFT)

- LoRA parameter-efficient fine-tuning

- Direct Preference Optimization (DPO)

- Reasoning-DPO

- Model evaluation and error analysis



The goal of this project is to understand, reproduce, and analyze modern LLM post-training techniques.



---



**Pipeline**



The overall workflow:



```mermaid

flowchart TD



A[Qwen2.5-1.5B-Instruct]



B[LoRA Supervised Fine-Tuning]



C[Instruction-following Model]



D[Preference Dataset Construction]



E[Direct Preference Optimization]



F[Reasoning Preference Dataset]



G[Reasoning-DPO]



H[GSM8K Evaluation]



I[Error Analysis]



A --> B

B --> C

C --> D

D --> E

F --> G

E --> G

G --> H

H --> I

```



---



**Results Summary**



| Stage | Objective | Observation |

|---|---|---|

| Base Model | General language modeling | Original model capability |

| SFT | Instruction supervision | Improves instruction following and response structure |

| DPO | Preference optimization | Improves response alignment and consistency |

| Reasoning-DPO | Reasoning preference optimization | Small improvement on mathematical reasoning |



---



**Project Structure**



```

mini-dpo-lab



├── data

│   ├── prepare_sft_[data.py](http://data.py)

│   ├── prepare_dpo_[data.py](http://data.py)

│   ├── generate_reasoning_[dpo.py](http://dpo.py)

│   └── validate_reasoning_[pairs.py](http://pairs.py)

│

├── training

│   ├── train_[sft.py](http://sft.py)

│   ├── train_[dpo.py](http://dpo.py)

│   └── train_reasoning_[dpo.py](http://dpo.py)

│

├── evaluation

│   ├── compare_[sft.py](http://sft.py)

│   ├── compare_[gsm8k.py](http://gsm8k.py)

│   ├── calculate_gsm8k_[accuracy.py](http://accuracy.py)

│   ├── analyze_reasoning_[gain.py](http://gain.py)

│   └── error_[analysis.md](http://analysis.md)

│

├── experiments

│   ├── EXP003_[DPO.md](http://DPO.md)

│   └── EXP004_reasoning_[dpo.md](http://dpo.md)

│

├── docs

│   ├── EXP003_DPO_[result.md](http://result.md)

│   └── EXP004_reasoning_dataset_[analysis.md](http://analysis.md)

│

└── [README.md](http://README.md)

```



---



**Model**



**Base Model**



Qwen/Qwen2.5-1.5B-Instruct



**Training Strategy**



- Freeze original model parameters

- Train LoRA adapters

- Reduce GPU memory requirements



**Hardware**



RTX 5070 Laptop GPU (16GB VRAM)



---



**Experiments**



**EXP001: Toy SFT Pipeline**



### Objective



Validate the complete supervised fine-tuning workflow.



### Implemented



- Dataset formatting

- LoRA training

- Adapter saving

- Basic inference testing



---



**EXP002: Alpaca-style SFT**



### Objective



Improve instruction-following ability through supervised fine-tuning.



### Training



- Supervised Fine-Tuning

- LoRA parameter-efficient training



### Observation



SFT improves:



- Instruction following

- Response formatting

- Task-oriented generation



### Limitation



- Limited improvement in reasoning ability



---



**EXP003: Direct Preference Optimization (DPO)**



### Objective



Apply preference optimization on top of the SFT model.



### Dataset



UltraFeedback preference dataset



### Training Format



```json

{

  "prompt": "...",

  "chosen": "...",

  "rejected": "..."

}

```



### Configuration



- Epochs: 1

- Batch size: 1

- Gradient accumulation steps: 4

- Learning rate: 5e-5

- Beta: 0.1

- Max sequence length: 512



### Evaluation



Compared models:



- Base Qwen model

- SFT model

- DPO model



### Findings



- SFT improves instruction following and response formatting.

- DPO improves preference alignment and response consistency.

- Alignment optimization does not automatically improve reasoning ability.



---



**EXP004: Reasoning-DPO**



**Objective**



Investigate whether reasoning-aware preference optimization can improve mathematical reasoning ability.



Different from standard DPO, this experiment uses reasoning preference pairs:



- chosen: correct reasoning trajectory

- rejected: plausible but incorrect reasoning trajectory



---



**Dataset**



- GSM8K

- 500 generated reasoning preference pairs

- 404 validated samples



Pipeline:



```

GSM8K



↓



Reasoning generation



↓



Negative reasoning construction



↓



Validation



↓



Reasoning-DPO Training

```



---



**Evaluation**



Benchmark:



- GSM8K test set

- 100 samples



**Results**



| Model | Accuracy |

|---|---:|

| Base | 29% |

| SFT | 32% |

| DPO | 32% |

| Reasoning-DPO | 33% |



---



**Analysis**



Reasoning-DPO achieves a small improvement over standard DPO.



Improvements:



- More complete reasoning trajectories

- Better step-by-step problem solving



Failure Cases:



- Irrelevant continuation

- Hallucinated instructions

- Noise from synthetic preference data



---



**Future Improvements**



- Higher quality reasoning preference data

- Better negative reasoning generation

- Larger evaluation benchmark

- Comparison with GRPO and RL-based reasoning optimization



---



**References**



- InstructGPT: Training language models to follow instructions with human feedback

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- Qwen2.5 Technical Report

- UltraFeedback Dataset

- GSM8K Dataset