# EXP004: Reasoning-DPO



**1. Overview**



Large Language Models (LLMs) have demonstrated strong instruction-following ability after supervised fine-tuning and preference optimization.



However, standard preference optimization methods such as Direct Preference Optimization (DPO) mainly optimize response preference and alignment, but do not explicitly focus on improving reasoning ability.



This experiment investigates whether preference optimization over reasoning trajectories can improve mathematical reasoning performance.



Research Question:



Can reasoning-aware preference optimization improve the mathematical reasoning ability of small language models?





**2. Motivation**



Standard DPO uses preference pairs:



(prompt, chosen response, rejected response)



where the model learns to prefer preferred answers.



In this experiment, we extend the preference signal to reasoning processes:



(prompt, correct reasoning trajectory, incorrect reasoning trajectory)



The hypothesis:



- Correct reasoning trajectories provide richer supervision.

- Learning to distinguish correct and incorrect reasoning may improve reasoning reliability.





**3. Method**



3.1 Dataset Construction



Pipeline:



GSM8K Questions



↓



Generate Correct Reasoning



↓



Generate Incorrect Reasoning



↓



Validate Preference Pairs



↓



Reasoning-DPO Dataset





Each preference sample contains:



{

    "prompt": "math problem",

    "chosen": "correct reasoning process",

    "rejected": "incorrect reasoning process"

}





3.2 Preference Pair Generation



Chosen Response:



- Correct final answer

- Complete reasoning steps

- Clear mathematical explanation



Rejected Response:



- Plausible reasoning process

- Subtle reasoning mistake

- Incorrect final answer





**4. Experimental Setup**



Model:



Base Model:

Qwen2.5-1.5B-Instruct



Training Method:

LoRA + DPO



GPU:

RTX 5070 Laptop GPU (16GB VRAM)





Dataset:



Original Dataset:

GSM8K



Generated Pairs:

500



Validated Pairs:

404





Training Configuration:



Epochs: 1



Batch Size: 1



Gradient Accumulation: 4



Learning Rate: 5e-5



Beta: 0.1



Max Length: 512





**5. Evaluation**



Benchmark:



GSM8K test set



Evaluation Size:



100 samples





Compared Models:



**1. Base Qwen2.5-1.5B-Instruct**



**2. SFT model**



**3. DPO model**



**4. Reasoning-DPO model**





**6. Results**



GSM8K Accuracy:



Model | Accuracy



Base | 29%



SFT | 32%



DPO | 32%



Reasoning-DPO | 33%





Observation:



Reasoning-DPO achieves a small improvement over standard DPO.



Although the improvement is limited, the experiment demonstrates that reasoning trajectories can provide additional preference signals beyond final-answer supervision.





**7. Error Analysis**



Improvement Cases:



- Producing more complete reasoning steps

- Reducing skipped intermediate calculations

- Following structured problem-solving procedures





Failure Cases:



- Synthetic reasoning noise

- Irrelevant continuation after answering

- Hallucinated instructions from generated data

- Limited model capacity





**8. Limitations**



**1. Small Dataset Scale**



Only 404 validated reasoning preference pairs are used.



**2. Synthetic Preference Data**



Reasoning trajectories are generated automatically.



**3. Small Model Scale**



Experiments are conducted on a 1.5B parameter model.



**4. Limited Evaluation**



Only GSM8K is evaluated.





**9. Future Work**



Better Reasoning Preference Data:



- Human verified reasoning trajectories

- More diverse reasoning errors

- Hard negative generation





More Advanced Alignment Methods:



- SimPO

- ORPO

- GRPO





Larger Evaluation:



- GSM8K

- MATH

- BBH





**10. Conclusion**



EXP004 explores reasoning-aware preference optimization through Reasoning-DPO.



The results show a small but measurable improvement over standard DPO on GSM8K.



This experiment demonstrates a complete workflow for investigating LLM post-training methods:



Dataset Construction



↓



Preference Optimization



↓



Evaluation



↓



Error Analysis

  
