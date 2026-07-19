# mini-dpo-lab



A minimal but complete implementation of an LLM alignment pipeline, including:



- Supervised Fine-Tuning (SFT)

- LoRA parameter-efficient fine-tuning

- Direct Preference Optimization (DPO)

- Reasoning-aware Preference Optimization

- Model evaluation and behavior analysis



The goal of this project is to understand and reproduce modern LLM post-training workflows under limited computational resources.





Overview



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





Project Structure



mini-dpo-lab



├── data

│   ├── prepare_sft_[data.py](http://data.py)

│   ├── prepare_dpo_[data.py](http://data.py)

│   ├── generate_reasoning_[dpo.py](http://dpo.py)

│   └── validate_reasoning_[pairs.py](http://pairs.py)

│

├── training

│   ├── train_[sft.py](http://sft.py)

│   ├── merge_[sft.py](http://sft.py)

│   ├── train_[dpo.py](http://dpo.py)

│   └── train_reasoning_[dpo.py](http://dpo.py)

│

├── evaluation

│   ├── compare_[sft.py](http://sft.py)

│   ├── compare_[dpo.py](http://dpo.py)

│   ├── compare_[gsm8k.py](http://gsm8k.py)

│   ├── calculate_gsm8k_[accuracy.py](http://accuracy.py)

│   └── analyze_reasoning_[gain.py](http://gain.py)

│

├── experiments

│   ├── EXP003_[DPO.md](http://DPO.md)

│   └── EXP004_reasoning_[dpo.md](http://dpo.md)

│

├── docs

│   └── experiment_[analysis.md](http://analysis.md)

│

└── [README.md](http://README.md)





Models



Base Model:



Qwen/Qwen2.5-1.5B-Instruct





Training Strategy:



- Freeze original model parameters

- Train LoRA adapters

- Reduce GPU memory requirements

- Keep the original model unchanged





Hardware:



RTX 5070 Laptop GPU (16GB VRAM)





Installation



Create conda environment:



conda create -n mini-dpo python=3.11



conda activate mini-dpo





Install dependencies:



pip install -r requirements.txt





Quick Start



1. Supervised Fine-Tuning (SFT)



Prepare SFT dataset:



python data/prepare_sft_[data.py](http://data.py)



Train SFT model:



python training/train_[sft.py](http://sft.py)





2. Direct Preference Optimization (DPO)



Prepare preference dataset:



python data/prepare_dpo_[data.py](http://data.py)



Train DPO model:



python training/train_[dpo.py](http://dpo.py)





3. Reasoning-DPO



Generate reasoning preference pairs:



python data/generate_reasoning_[dpo.py](http://dpo.py)



Validate preference pairs:



python data/validate_reasoning_[pairs.py](http://pairs.py)



Train Reasoning-DPO:



python training/train_reasoning_[dpo.py](http://dpo.py)





4. Evaluation



Run GSM8K evaluation:



python evaluation/compare_[gsm8k.py](http://gsm8k.py)



Calculate accuracy:



python evaluation/calculate_gsm8k_[accuracy.py](http://accuracy.py)





Experiments





EXP001: Toy SFT Pipeline



Objective:



Validate the supervised fine-tuning workflow.



Implemented:



- Dataset formatting

- Chat template preparation

- LoRA training

- Adapter saving

- Basic inference testing





EXP002: Alpaca-style SFT



Objective:



Improve instruction-following ability through supervised fine-tuning.





Observation:



SFT mainly improves:



- Instruction following

- Response formatting

- Task-oriented generation





Limitations:



- No significant factual knowledge improvement

- Limited reasoning improvement





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



Max sequence length: 512





Evaluation:



Compared models:



- Base Qwen model

- SFT model

- DPO model





Categories:



- Instruction following

- Format control

- Role playing

- Reasoning

- Coding





Main findings:



- SFT improves instruction following and response structure.

- DPO improves preference alignment and response consistency.

- SFT and DPO alone do not automatically improve mathematical reasoning ability.





EXP004: Reasoning-DPO





Overview:



EXP004 investigates whether preference optimization on reasoning trajectories can improve mathematical reasoning ability.





Different from standard DPO, this experiment introduces reasoning-aware preference pairs:



- chosen: correct reasoning trajectory

- rejected: plausible but incorrect reasoning trajectory





Dataset:



GSM8K





Pipeline:



GSM8K



↓



Generate reasoning preference pairs



↓



Validate dataset



↓



DPO training





Statistics:



Generated pairs: 500



Validated pairs: 404





Training:



Base Model:



Qwen2.5-1.5B-Instruct





Method:



- LoRA

- DPO





Evaluation:



Benchmark:



GSM8K test set





Evaluation samples:



100 problems





Results:



Base: 29%



SFT: 32%



DPO: 32%



Reasoning-DPO: 33%





Analysis:



Reasoning-DPO achieves a small improvement compared with standard DPO.





Improvements:



- More complete reasoning trajectories

- Clearer intermediate calculations

- Reduced incorrect reasoning patterns





Limitations:



- Generated preference data contains noise

- Negative reasoning generation quality affects performance

- Small-scale evaluation limits conclusions





Future Work:



- Higher quality reasoning preference datasets

- Better negative reasoning generation strategies

- Larger-scale evaluation

- Compare with GRPO and RL-based reasoning optimization

- Explore reasoning verification methods





References:



- InstructGPT: Training language models to follow instructions with human feedback

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- Qwen2.5 Technical Report

- UltraFeedback Dataset

- GSM8K Dataset

  


