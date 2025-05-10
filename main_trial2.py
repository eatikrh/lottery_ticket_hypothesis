import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# --- Generate synthetic noisy dataset ---
np.random.seed(0)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.4 * np.random.randn(*X.shape)  # more noise

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
      nn.ReLU(),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, 1)
    )

  def forward(self, x):
    return self.net(x)

# --- Training function ---
def train_and_evaluate(width):
  model = Net(width)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  loss_fn = nn.MSELoss()

  for epoch in range(75):  # fewer epochs to reveal overfitting
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

# --- Sweep widths with more resolution near interpolation threshold (~80 params) ---
widths = list(range(1, 60)) + list(range(60, 300, 10))
train_losses = []
test_losses = []

for w in widths:
  print(f"Training model with width {w}")
  train_loss, test_loss = train_and_evaluate(w)
  train_losses.append(train_loss)
  test_losses.append(test_loss)

# --- Plot results ---
plt.plot(widths, train_losses, label='Train Loss')
plt.plot(widths, test_losses, label='Test Loss')
plt.axvline(80, color='gray', linestyle='--', label='Interpolation Threshold')
plt.xlabel("Width of Hidden Layers")
plt.ylabel("MSE Loss")
plt.title("Double Descent in Neural Networks")
plt.legend()
plt.grid()
plt.savefig("double_descent_plot.png")
print("Plot saved to double_descent_plot.png")
