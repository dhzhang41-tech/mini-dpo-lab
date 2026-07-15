**mini-dpo-lab**

**Overview**

A minimal but complete implementation of an LLM alignment pipeline, including Supervised Fine-Tuning (SFT), LoRA parameter-efficient fine-tuning, Direct Preference Optimization (DPO), and model evaluation.  

The goal is to understand and reproduce modern LLM post-training workflows.  

## Results Summary

| Stage | Training Objective | Observed Improvement |

|:---:|:---|:---|

| Base Model | Pre-training | General language modeling capability |

| SFT | Instruction Supervision | Better instruction following and structured responses |

| DPO | Human Preference Optimization | Better alignment, helpfulness, and response consistency |

Pipeline:  
Base LLM (Qwen2.5-1.5B-Instruct) → SFT → DPO → Preference-aligned Model

**Project Structure**

data/  

- prepare_sft_[data.py](http://data.py)  
- prepare_dpo_[data.py](http://data.py)

training/  

- train_[sft.py](http://sft.py)  
- merge_[sft.py](http://sft.py)  
- train_[dpo.py](http://dpo.py)

evaluation/  

- compare_[sft.py](http://sft.py)  
- compare_[dpo.py](http://dpo.py)  
- prompts.json

docs/  

- EXP003_DPO_[result.md](http://result.md)

**Models**

Base model: Qwen2.5-1.5B-Instruct  

Training strategy:  

- Freeze original parameters  
- Train LoRA adapters  
- Reduce GPU memory requirements

Hardware: RTX 5070 Laptop GPU (8GB VRAM)

**EXP001: Toy SFT Pipeline**

Objective: Validate the supervised fine-tuning workflow.  

Implemented:  

- Dataset formatting  
- LoRA training  
- Adapter saving  
- Basic inference testing

**EXP002: Alpaca-style SFT**

Objective: Improve instruction-following ability through supervised fine-tuning.  

Configuration:  

- Epochs: 3  
- Batch size: 1  
- Gradient accumulation: 4  
- Learning rate: 2e-4  
- Max sequence length: 512

Observation:  
SFT mainly improved instruction following, response structure, and task-oriented generation.

**EXP003: Direct Preference Optimization (DPO)**

Objective: Apply preference optimization on top of the SFT model.  

Dataset: UltraFeedback preference dataset.  

Format:  
{prompt, chosen, rejected}  

Training samples: 1000 preference pairs.  

Configuration:  

- Epochs: 1  
- Batch size: 1  
- Gradient accumulation: 4  
- Learning rate: 5e-5  
- Beta: 0.1  
- Max sequence length: 512

**DPO Training Results**

Training completed successfully.  

Observed:  

- DPO loss decreased  
- Chosen response reward increased  
- Rejected response reward decreased  
- Reward margin improved

Final metrics:  

- Train loss: 0.6108  
- Reward margin: ~0.52  
- Reward accuracy: ~0.82

**Evaluation**

Compared models:  

1. Base Qwen model
2. SFT model
3. DPO model

Categories:  

- Instruction following  
- Format control  
- Role playing  
- Reasoning  
- Coding

Conclusion:  
SFT improves instruction following and formatting. DPO improves alignment-related behaviors such as helpfulness and response consistency.

**Key Findings**

SFT improves:  

- Instruction following  
- Response formatting  
- Task understanding

DPO improves:  

- Preference alignment  
- Helpfulness  
- Response consistency

However, general SFT and DPO do not automatically improve reasoning or coding abilities without specialized data.

**Future Work**

Possible extensions:  

1. Scale DPO dataset size.
2. Apply domain-specific alignment.
3. Combine DPO with LLM agents.

**References**

InstructGPT  
Direct Preference Optimization: Your Language Model is Secretly a Reward Model  
Qwen2.5 Technical Report  
UltraFeedback Dataset