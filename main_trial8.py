import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Reproducibility ---
np.random.seed(0)
torch.manual_seed(0)

# --- Dataset: sin(x) with noise ---
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.4 * np.random.randn(*X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# --- Model definition ---
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

# --- Pruning utilities ---
def get_mask(model, threshold):
  mask = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if 'weight' in name:
        mask[name] = (param.abs() >= threshold).float()
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

# --- Training routine ---
def train(model, epochs=200, lr=0.01, weight_decay=1e-4):
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

# --- Experiment: sweep pruning thresholds ---
width = 100
thresholds = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]

# Train full model once
full_model = Net(width)
initial_state = {k: v.clone() for k, v in full_model.state_dict().items()}
loss_full = train(full_model)

results = []
for threshold in thresholds:
  print(f"\nThreshold {threshold:.1e}")

  # Prune and get mask
  mask = get_mask(full_model, threshold)

  # Reset to initial weights
  ticket_model = Net(width)
  ticket_model.load_state_dict(initial_state)
  apply_mask(ticket_model, mask)
  apply_gradient_masking(ticket_model, mask)

  # Retrain only surviving weights
  loss_ticket = train(ticket_model)

  # Count weights
  total = sum(p.numel() for n, p in full_model.named_parameters() if 'weight' in n)
  kept = sum(mask[n].sum().item() for n in mask)
  percent = 100 * kept / total
  print(f"  Kept {percent:.1f}% weights — Loss: {loss_ticket:.4f}")
  results.append((percent, loss_ticket))

# --- Plot results ---
sparsity, losses = zip(*results)
plt.figure()
plt.plot(sparsity, losses, marker='o')
plt.axvline(100, linestyle='--', color='gray', label="Unpruned")
plt.xlabel("Percent of Weights Kept (%)")
plt.ylabel("Test Loss After Retraining")
plt.title("Lottery Ticket Recovery vs Sparsity")
plt.grid()
plt.legend()
plt.savefig("lottery_ticket_recovery.png")
print("Plot saved to lottery_ticket_recovery.png")
