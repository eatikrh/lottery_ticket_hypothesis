# 🎯 Exploring the Lottery Ticket Hypothesis and Iterative Magnitude Pruning

This project is a hands-on exploration of the **Lottery Ticket Hypothesis (LTH)** and **Iterative Magnitude Pruning (IMP)** using a PyTorch-based regression task. It was conducted during our internal Learning Day as a deep dive into **model sparsity, optimization dynamics, and generalization**.

---

## 📌 Executive Summary

✅ We discovered that **pruning up to ~50% of weights can improve generalization**, and up to 70% with no meaningful performance drop.

🧠 We implemented an **Iterative Magnitude Pruning** (IMP) pipeline with weight rewinding, and validated that **lottery tickets exist** and can be retrained successfully from their original initialization.

📈 Our final experiment showed that **test error decreased** through multiple pruning and retraining cycles — culminating in a sweet spot around **40–50% sparsity**.

---

## 🧭 What We Explored Along the Way

Before discovering the sweet spot with iterative pruning, we tested many variations and captured key insights:

### 🔁 Double Descent
- We explored how **test loss behaves non-monotonically** as network width increases.
- Confirmed the **classic "double descent" curve**: initial improvement, followed by overfitting, then a second improvement with extreme overparameterization.

### 🧹 One-shot Magnitude Pruning
- We pruned weights by magnitude after training and observed **minimal loss degradation**, confirming the presence of **redundant parameters**.
- However, simply pruning and leaving the weights didn’t confirm the LTH.

### 🎟️ One-shot Lottery Ticket Test
- We pruned weights post-training and **reset the surviving weights to their initial values**, retraining only those.
- This proved that **sparse subnetworks can learn independently** — a partial validation of LTH.

### 🔄 Dynamic Training Length
- We scaled training epochs with model width to prevent underfitting of wide models and **exposed instability** at extreme widths.

### 📉 Activation & Regularization Experiments
- Switching from `ReLU` to `Tanh` improved smoothness.
- Adding `weight_decay` (L2 regularization) stabilized wide-model behavior and **accentuated the pruning benefit**.

Each step brought us closer to a deeper understanding of network capacity, redundancy, and generalization behavior.

---

## 🧠 Core Concepts

### Lottery Ticket Hypothesis (LTH)
A dense neural network contains a smaller subnetwork that — when reset to its initial weights — can match or exceed the full model’s performance after training.

### Iterative Magnitude Pruning (IMP)
Gradually prune and rewind:
1. Train the model
2. Prune a small % of lowest-magnitude weights
3. Rewind the surviving weights to their original initialization
4. Retrain — repeat for `N` rounds

---

## 💡 Final Experiment: Iterative Pruning

- 10 pruning rounds × 10% pruning per round
- Model width = 100
- Trained from scratch after each round using masked gradient updates
- Output plot: `imp_recovery.png`

### ✅ Key Outcome

- **Best performance occurred at ~45–50% weights kept**
- The model generalized better than the dense baseline
- Clear tipping point visible as sparsity increases

---

## 📂 Outputs

- `imp_recovery.png`: test loss vs percent of weights retained

---

## 🛠 Technologies

- Python 3.9+
- PyTorch
- NumPy, Matplotlib, scikit-learn
- Podman (optional containerization)

---

## 📈 To Run

Local:

```bash
pip install torch numpy matplotlib scikit-learn
python main.py
