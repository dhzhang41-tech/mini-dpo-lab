# EXP004 Reasoning Preference Dataset Analysis

## 1. Overview

EXP004 investigates whether preference optimization can improve mathematical reasoning ability by constructing reasoning-aware preference pairs.

Different from standard DPO, which mainly optimizes the final answer preference, EXP004 focuses on the quality of intermediate reasoning trajectories.

The goal is to train a model to prefer:

- correct reasoning processes

- logically consistent solutions

- fewer common reasoning failures

---

## 2. Dataset Construction

Dataset:

- Source: GSM8K

- Task: Mathematical word problems

- Initial samples: 100

Generation pipeline:

GSM8K Problem

↓

DeepSeek Reasoning Generation

↓

Generate:

Chosen reasoning trajectory

Rejected reasoning trajectory

↓

Automatic validation

The chosen response represents a correct reasoning process.

The rejected response represents a plausible but incorrect reasoning process.

---

## 3. Filtering Results

| Metric | Value |

|---|---:|

| Generated samples | 100 |

| Valid preference pairs | 84 |

| Filtered samples | 16 |

| Valid ratio | 84% |

Filtering criteria:

1. Chosen answer matches ground truth.

2. Rejected answer differs from ground truth.

3. Both reasoning trajectories contain valid final answers.

---

## 4. Reasoning Error Taxonomy

We manually inspected 25 rejected reasoning trajectories and categorized common failure patterns.

| Error Type | Count | Description |

|---|---:|---|

| Semantic Interpretation Error | 9 | Misunderstanding natural language relationships |

| Arithmetic Error | 9 | Incorrect numerical calculation |

| Missing Information Error | 3 | Ignoring important problem conditions |

| Multi-step Logical Error | 3 | Failure in maintaining multi-step reasoning consistency |

---

## 5. Error Examples

### 5.1 Semantic Interpretation Error

Example:

"half as many"

Correct:

May = April / 2

Incorrect:

May = half of total sales

Failure:

The model misunderstands the semantic relationship between quantities.

---

### 5.2 Arithmetic Error

Example:

7 starfish × 5 arms

Correct:

7 × 5 = 35

Incorrect:

7 × 5 = 30

Failure:

The reasoning structure is correct but numerical calculation fails.

---

### 5.3 Missing Information Error

Example:

A problem contains:

- initial amount

- additional amount

- final amount

Incorrect reasoning ignores the initial amount.

Failure:

The model loses previously provided information.

---

### 5.4 Multi-step Logical Error

Example:

A model correctly calculates intermediate values but changes the reasoning objective in later steps.

Failure:

The model cannot maintain logical consistency throughout the entire reasoning chain.

---

## 6. Summary

The constructed reasoning preference dataset contains diverse reasoning failures rather than simple incorrect answers.

The dominant failure modes are:

1. Semantic misunderstanding

2. Arithmetic mistakes

These observations motivate EXP004:

Training a reasoning-aware DPO model that learns preferences over reasoning trajectories rather than only final answers.