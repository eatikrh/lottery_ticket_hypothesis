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

## 🛠 Requirements

- Python 3.9+
- PyTorch
- matplotlib
- scikit-learn
- numpy

Install with:
```bash
pip install torch matplotlib scikit-learn numpy
