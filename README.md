# Lottery Ticket Hypothesis & Iterative Magnitude Pruning (IMP) Demonstration

This repository demonstrates the **Lottery Ticket Hypothesis (LTH)** and **Iterative Magnitude Pruning (IMP)** using a simple regression task: fitting noisy `sin(x)` data with fully connected neural networks in PyTorch.

## 🧠 Core Ideas

- **LTH**: A randomly initialized dense neural network contains a small subnetwork (a "winning ticket") that, when trained in isolation from the same initialization, matches the full model's performance.
- **IMP**: By gradually pruning small weights and retraining after each round, we can preserve or even improve test accuracy while drastically reducing model size.

## 🧪 What This Code Does

### 1. Generate Synthetic Data
- Function: `y = sin(x) + noise`
- 100 points, split into train/test

### 2. Train a Fully Connected Model
- 2 hidden layers, 100 neurons wide, Tanh activations
- Optimizer: Adam

### 3. Apply Iterative Magnitude Pruning (IMP)
- In each of 10 rounds:
    - Train the model
    - Prune the **smallest 10% of remaining weights**
    - Reset the surviving weights to their **original initialization**
    - Retrain **only the surviving connections**
- The pruning is applied globally across all weight matrices.

### 4. Record and Plot
- For each round:
    - % weights kept
    - Test loss after retraining
- The output plot: `imp_recovery.png`

## 📈 Example Result

The IMP curve shows test loss vs % weights kept. A well-shaped curve demonstrates:
- Initial pruning reduces overfitting
- The model retains good performance even when over half the weights are removed
- Over-pruning eventually degrades performance

## 📂 Output Files

- `imp_recovery.png` – Test loss vs sparsity plot
# 📚 Reference

This project builds on the ideas from:

- [The Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019)](https://arxiv.org/pdf/1803.03635v5)

The paper proposes that dense, randomly initialized networks contain smaller subnetworks that can be trained in isolation to match the performance of the full model — so long as they're reset to the same initial weights and trained properly.

## 🤔 Does DeepSeek Use the Lottery Ticket Hypothesis?

While DeepSeek does not explicitly use the Lottery Ticket Hypothesis (LTH) or iterative pruning, its goals — reducing training cost while maintaining competitive performance — strongly resonate with the *philosophy* behind LTH:

> “You don’t need all the parameters — you just need the right ones, trained the right way.”

However, DeepSeek focuses on system-level efficiency (e.g., mixed precision, GPU optimization) rather than structural sparsity or subnetwork discovery. There is no current public evidence that DeepSeek applies LTH-like weight pruning or rewinding techniques during training.


## 🛠 Requirements

- Python 3.9+
- PyTorch
- matplotlib
- scikit-learn
- numpy

Install with:
```bash
pip install torch matplotlib scikit-learn numpy


#