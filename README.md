**mini-dpo-lab**

A minimal but complete implementation of an LLM alignment pipeline, including:  
  
- Supervised Fine-Tuning (SFT)  
- LoRA parameter-efficient fine-tuning  
- Direct Preference Optimization (DPO)  
- Model evaluation and behavior analysis  
  
The goal of this project is to understand and reproduce modern LLM post-training workflows.

**Results Summary**

Stage | Training Objective | Observed Improvement  
Base Model | Pre-training | General language modeling capability  
SFT | Instruction Supervision | Better instruction following and structured responses  
DPO | Human Preference Optimization | Better alignment, helpfulness, and response consistency

**Pipeline**

Qwen2.5-1.5B-Instruct  
  
        ↓  
  
Supervised Fine-Tuning (SFT)  
  
        ↓  
  
Instruction-following Model  
  
        ↓  
  
Direct Preference Optimization (DPO)  
  
        ↓  
  
Preference-aligned Model

**Project Structure**

mini-dpo-lab  
  
├── data  
│   ├── prepare_sft_[data.py](http://data.py)  
│   └── prepare_dpo_[data.py](http://data.py)  
│  
├── training  
│   ├── train_[sft.py](http://sft.py)  
│   ├── merge_[sft.py](http://sft.py)  
│   └── train_[dpo.py](http://dpo.py)  
│  
├── evaluation  
│   ├── compare_[sft.py](http://sft.py)  
│   ├── compare_[dpo.py](http://dpo.py)  
│   ├── prompts.json  
│   └── outputs  
│  
├── docs  
│   └── EXP003_DPO_[result.md](http://result.md)  
│  
└── [README.md](http://README.md)

**Models**

Base Model:  
Qwen/Qwen2.5-1.5B-Instruct  
  
Training Strategy:  
- Freeze original model parameters  
- Train LoRA adapters  
- Reduce GPU memory requirements  
  
Hardware:  
RTX 5070 Laptop GPU (8GB VRAM)

**Experiments**

**EXP001: Toy SFT Pipeline**

Objective:  
Validate the supervised fine-tuning workflow.  
  
Implemented:  
- Dataset formatting  
- LoRA training  
- Adapter saving  
- Basic inference testing

**EXP002: Alpaca-style SFT**

Objective:  
Improve instruction-following ability through supervised fine-tuning.  
  
Training:  
- Supervised Fine-Tuning  
- LoRA fine-tuning  
  
Observation:  
SFT mainly improves instruction following, response structure, and task-oriented generation.  
  
Limitations:  
- No significant factual knowledge improvement  
- Limited reasoning improvement

**EXP003: Direct Preference Optimization (DPO)**

Objective:  
Apply preference optimization on top of the SFT model.  
  
Dataset:  
UltraFeedback preference dataset.  
  
Training format:  
{prompt, chosen, rejected}  
  
Training samples:  
1000 preference pairs  
  
Configuration:  
- Epochs: 1  
- Batch size: 1  
- Gradient accumulation: 4  
- Learning rate: 5e-5  
- Beta: 0.1  
- Max sequence length: 512

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
  
Main findings:  
- SFT improves instruction following and response formatting.  
- DPO improves preference alignment, helpfulness, and response consistency.  
- General SFT and DPO do not automatically improve reasoning or coding ability.

**References**

- InstructGPT: Training language models to follow instructions with human feedback  
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model  
- Qwen2.5 Technical Report  
- UltraFeedback Dataset