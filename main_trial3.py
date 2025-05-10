import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# --- Set random seed for reproducibility ---
np.random.seed(0)
torch.manual_seed(0)

# --- Generate synthetic noisy dataset ---
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.4 * np.random.randn(*X.shape)  # high noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# --- Define neural network class ---
class Net(nn.Module):
  def __init__(self, width):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(1, width),
      nn.ReLU(),  # Try nn.Tanh() for smoother behavior
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, 1)
    )

  def forward(self, x):
    return self.net(x)

# --- Train and evaluate one model ---
def train_and_evaluate(width, epochs=75):
  model = Net(width)
  optimizer = optim.Adam(model.parameters(), lr=0.01)  # Consider lowering for wide models
  loss_fn = nn.MSELoss()

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
    test_loss = loss_fn(model(X_test_t), y_test_t).item()

  return train_loss, test_loss

# --- Average multiple runs for stability ---
def averaged_loss(width, trials=5):
  train_losses = []
  test_losses = []
  for i in range(trials):
    torch.manual_seed(i)  # change seed per trial
    train_loss, test_loss = train_and_evaluate(width, epochs=100 if width > 200 else 75)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
  return np.mean(train_losses), np.mean(test_losses)

# --- Sweep across widths ---
widths = list(range(1, 60)) + list(range(60, 300, 10))
train_results = []
test_results = []

for w in widths:
  print(f"Training width {w}")
  train_loss, test_loss = averaged_loss(w)
  train_results.append(train_loss)
  test_results.append(test_loss)

# --- Plot ---
plt.plot(widths, train_results, label='Train Loss')
plt.plot(widths, test_results, label='Test Loss')
plt.axvline(80, color='gray', linestyle='--', label='Interpolation Threshold')
plt.xlabel("Width of Hidden Layers")
plt.ylabel("MSE Loss")
plt.title("Double Descent in Neural Networks (Averaged)")
plt.legend()
plt.grid()
plt.savefig("double_descent_plot.png")
print("Plot saved to double_descent_plot.png")
