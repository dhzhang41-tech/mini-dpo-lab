# Reasoning-DPO Error Analysis

## 1. Experiment Overview

Dataset:

- GSM8K

Models:

- Base

- SFT

- DPO

- Reasoning-DPO

Evaluation:

- 100 GSM8K test samples

Results:

| Model | Accuracy |

|---|---|

| Base | 29% |

| SFT | 32% |

| DPO | 32% |

| Reasoning-DPO | 33% |

---

# 2. Gain Analysis

## Summary

Reasoning-DPO improves DPO mainly through:

1. Reducing incomplete generation

2. Improving instruction following

3. Limited reasoning correction ability

## Gain Categories

| Category | Count |

|---|---|

| Truncation Fix | 4 |

| Instruction Leakage Fix | 2 |

| Reasoning Correction | 1 |

| No Significant Improvement | 3 |

## Example Cases

### Case 1: Truncation Fix

Question:

...

DPO:

...

Reasoning-DPO:

...

Analysis:

Reasoning-DPO completes the reasoning chain.

---

# 3. Loss Analysis

## Summary

Reasoning-DPO failures mainly come from:

1. Overly verbose generation

2. Instruction leakage

3. Incorrect reasoning expansion

## Loss Categories

| Category | Count |

|---|---|

| Instruction Leakage |4|

| Verbosity Regression |2|

| Overthinking Error |1|

| New Arithmetic Error |1|

## Example Cases

### Case 1: Arithmetic Regression

Question:

...

DPO:

...

Reasoning-DPO:

...

Analysis:

The model introduced an unnecessary reasoning step and caused an arithmetic error.

---

# 4. Findings

The current Reasoning-DPO improves output completeness,

but does not significantly improve mathematical reasoning ability.

The main limitation is that the training pairs teach

format imitation rather than deeper reasoning improvement.

---

# 5. Future Improvement

Potential directions:

1. Higher quality reasoning preference pairs

2. Error-aware rejected samples

3. Self-correction reasoning data

4. Hard reasoning benchmark evaluation