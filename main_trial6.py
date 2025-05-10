import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# --- Seed for reproducibility ---
np.random.seed(0)
torch.manual_seed(0)

# --- Dataset: noisy sin(x) ---
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.4 * np.random.randn(*X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# --- Define model ---
class Net(nn.Module):
  def __init__(self, width):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(1, width),
      nn.Tanh(),
      nn.Linear(width, width),
      nn.Tanh(),
      nn.Linear(width, 1)
    )

  def forward(self, x):
    return self.net(x)

# --- Pruning function ---
def prune_weights(model, threshold=1e-2):
  total_params, pruned_params = 0, 0
  with torch.no_grad():  # This disables gradient tracking during pruning
    for name, param in model.named_parameters():
      if param.requires_grad:
        mask = param.abs() >= threshold
        pruned_params += (~mask).sum().item()
        total_params += mask.numel()
        param.copy_(param * mask)  # Safe in-place update
  percent = 100 * (total_params - pruned_params) / total_params
  print(f"Kept {percent:.1f}% of weights (threshold={threshold})")
  return percent

# --- Train and evaluate ---
def train_and_evaluate(width, threshold=1e-2, base_epochs=75):
  model = Net(width)
  optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
  loss_fn = nn.MSELoss()
  epochs = base_epochs + (width // 5)

  for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = loss_fn(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

  model.eval()
  with torch.no_grad():
    train_loss = loss_fn(model(X_train_t), y_train_t).item()
    test_loss_before = loss_fn(model(X_test_t), y_test_t).item()

  percent_kept = prune_weights(model, threshold)

  with torch.no_grad():
    test_loss_after = loss_fn(model(X_test_t), y_test_t).item()

  return train_loss, test_loss_before, test_loss_after, percent_kept

# --- Experiment setup ---
widths = list(range(1, 60)) + list(range(60, 300, 10))
train_losses, test_losses_before, test_losses_after, sparsities = [], [], [], []

for w in widths:
  print(f"\nTraining width {w}")
  train_loss, test_before, test_after, percent_kept = train_and_evaluate(w)
  train_losses.append(train_loss)
  test_losses_before.append(test_before)
  test_losses_after.append(test_after)
  sparsities.append(percent_kept)

# --- Plot 1: Double descent before and after pruning ---
plt.figure()
plt.plot(widths, test_losses_before, label='Test Loss (Before Prune)')
plt.plot(widths, test_losses_after, label='Test Loss (After Prune)')
plt.axvline(80, color='gray', linestyle='--', label='Interpolation Threshold')
plt.xlabel("Width of Hidden Layers")
plt.ylabel("MSE Loss")
plt.title("Test Loss Before vs After Pruning")
plt.legend()
plt.grid()
plt.savefig("double_descent_pruning_loss.png")
print("Saved: double_descent_pruning_loss.png")

# --- Plot 2: Percent of weights kept vs test loss ---
plt.figure()
plt.plot(sparsities, test_losses_after, marker='o')
plt.xlabel("Percent of Weights Kept After Pruning (%)")
plt.ylabel("Test Loss After Pruning")
plt.title("Compression vs Performance")
plt.grid()
plt.savefig("sparsity_vs_loss.png")
print("Saved: sparsity_vs_loss.png")
