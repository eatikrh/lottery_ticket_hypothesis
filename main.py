import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Reproducibility ---
np.random.seed(0)
torch.manual_seed(0)

# --- Dataset ---
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.4 * np.random.randn(*X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# --- Model ---
class Net(nn.Module):
  def __init__(self, width):
    super().__init__()
    self.fc1 = nn.Linear(1, width)
    self.act1 = nn.Tanh()
    self.fc2 = nn.Linear(width, width)
    self.act2 = nn.Tanh()
    self.fc3 = nn.Linear(width, 1)

  def forward(self, x):
    x = self.act1(self.fc1(x))
    x = self.act2(self.fc2(x))
    return self.fc3(x)

# --- Utilities ---
def train(model, epochs=100, lr=0.01, weight_decay=1e-4):
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  loss_fn = nn.MSELoss()
  for _ in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
  model.eval()
  with torch.no_grad():
    return loss_fn(model(X_test_t), y_test_t).item()

def get_flattened_weights(model, mask):
  weights = []
  for name, param in model.named_parameters():
    if 'weight' in name:
      unmasked = param[mask[name].bool()]
      weights.append(unmasked.view(-1))
  return torch.cat(weights)

def get_mask(model, threshold, current_mask):
  mask = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if 'weight' in name:
        if name in current_mask:
          surviving = current_mask[name]
        else:
          surviving = torch.ones_like(param)
        new_mask = (param.abs() >= threshold).float() * surviving
        mask[name] = new_mask
  return mask

def apply_mask(model, mask):
  with torch.no_grad():
    for name, param in model.named_parameters():
      if name in mask:
        param.mul_(mask[name])

def apply_gradient_masking(model, mask):
  for name, param in model.named_parameters():
    if name in mask:
      def hook(grad, m=mask[name]):
        return grad * m
      param.register_hook(hook)

# --- Iterative Pruning Setup ---
width = 100
rounds = 10
prune_percent = 0.10
epochs_per_round = 125

model = Net(width)
initial_state = {k: v.clone() for k, v in model.state_dict().items()}
current_mask = {name: torch.ones_like(p) for name, p in model.named_parameters() if 'weight' in name}

losses = []
sparsities = []

for r in range(rounds + 1):  # include round 0 = unpruned
  print(f"\n== Round {r} ==")

  # Reset to initial weights
  model = Net(width)
  model.load_state_dict(initial_state)

  # Apply current mask and gradient masking
  apply_mask(model, current_mask)
  apply_gradient_masking(model, current_mask)

  # Train and evaluate
  loss = train(model, epochs=epochs_per_round)
  kept = sum(current_mask[k].sum().item() for k in current_mask)
  total = sum(current_mask[k].numel() for k in current_mask)
  percent_kept = 100 * kept / total

  print(f"  Weights kept: {percent_kept:.1f}% — Test Loss: {loss:.4f}")
  losses.append(loss)
  sparsities.append(percent_kept)

  # Skip pruning after final round
  if r == rounds:
    break

  # Get all surviving weights and compute global threshold
  all_weights = get_flattened_weights(model, current_mask)
  k = int(prune_percent * all_weights.numel())
  threshold = torch.topk(all_weights.abs(), k, largest=False).values.max().item()

  # Update mask
  current_mask = get_mask(model, threshold, current_mask)

# --- Plot ---
plt.figure()
plt.plot(sparsities, losses, marker='o')
plt.xlabel("Percent of Weights Kept (%)")
plt.ylabel("Test Loss After Retraining")
plt.title("Iterative Magnitude Pruning (IMP) Recovery")
plt.grid()
plt.savefig("imp_recovery.png")
print("Saved plot to imp_recovery.png")
