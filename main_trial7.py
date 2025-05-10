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

# --- Network ---
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

def train(model, epochs, lr=0.01, weight_decay=1e-4):
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

# --- Step 1: Init full model and save weights ---
width = 100
full_model = Net(width)
initial_state = {k: v.clone() for k, v in full_model.state_dict().items()}

# --- Step 2: Train full model ---
loss_full = train(full_model, epochs=200)

# --- Step 3: Prune trained model ---
threshold = 1e-2
mask = get_mask(full_model, threshold)

# --- Step 4: Reset surviving weights to initial values ---
ticket_model = Net(width)
ticket_model.load_state_dict(initial_state)
apply_mask(ticket_model, mask)

# --- Step 5: Retrain only surviving weights ---
# Freeze zeroed weights by registering a hook to mask gradients
def apply_gradient_masking(model, mask):
  for name, param in model.named_parameters():
    if name in mask:
      def hook(grad, m=mask[name]):
        return grad * m
      param.register_hook(hook)

apply_gradient_masking(ticket_model, mask)
loss_ticket = train(ticket_model, epochs=200)

# --- Results ---
print(f"Test Loss (Full Model):     {loss_full:.4f}")
print(f"Test Loss (Lottery Ticket): {loss_ticket:.4f}")
