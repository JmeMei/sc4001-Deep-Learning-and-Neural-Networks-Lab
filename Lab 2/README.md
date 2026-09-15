# SC4001 Notes

## `Lab 2/` — single-neuron gradient descent examples

Each numbered example (2.2, 2.3) has an `a`/`b` pair that keeps the same
underlying problem but swaps out one implementation detail, so you can
compare the two approaches side by side.

### 2.2 — logistic regression neuron

[`eg2.2a.ipynb`](eg2.2a.ipynb) vs [`eg2.2b.ipynb`](eg2.2b.ipynb) — same
neuron and training data, but 2.2a derives gradients by hand while 2.2b lets
PyTorch autograd compute them.

**Forward pass (identical in both):**

```
u = X·w + b            (pre-activation / logit)
f(u) = 1 / (1 + e^-u)  (sigmoid → predicted probability)
```

**[eg2.2a.ipynb](eg2.2a.ipynb) — manual gradients**

Cross-entropy loss, computed by hand:

```
J = -Σ_p [ d_p·log(f(u_p)) + (1-d_p)·log(1-f(u_p)) ]
```

```python
def loss(targets, logits):
  entropy = -torch.sum(torch.tensor(targets)*torch.log(logits) + (1-torch.tensor(targets))*torch.log(1-logits))
  class_err = torch.sum(torch.not_equal(logits > 0.5, torch.tensor(targets)))
  return entropy, class_err
```

Weight update, using the closed-form gradient of that loss:

```
grad_u = f(u) - d
w ← w - α·Xᵀ·grad_u   (== w + α·Xᵀ(d - f(u)))
b ← b - α·1ᵀ·grad_u
```

```python
def train(model, inputs, targets, learning_rate):
  _, f_u = model(inputs)
  grad_u = -(torch.tensor(targets) - f_u)
  grad_w = torch.matmul(torch.transpose(torch.tensor(inputs),0,1), grad_u)
  grad_b = torch.sum(grad_u)
  model.w -= learning_rate * grad_w
  model.b -= learning_rate * grad_b
```

**[eg2.2b.ipynb](eg2.2b.ipynb) — autograd**

Same loss, but via PyTorch's built-in `nn.BCELoss()` (binary cross-entropy),
and the weight update comes from `.backward()` instead of a hand-coded
gradient:

```python
loss = nn.BCELoss()

def train(model, inputs, targets, learning_rate):
  _, logits = model(inputs)
  loss_ = loss(logits, torch.tensor(targets, dtype=torch.double))
  err = torch.sum(torch.not_equal(logits > 0.5, torch.tensor(targets)))

  loss_.backward()                     # autograd computes grad_w, grad_b
  with torch.no_grad():
      model.w -= learning_rate * model.w.grad
      model.b -= learning_rate * model.b.grad
  model.w.grad = None
  model.b.grad = None
  return loss_, err
```

`model.w`/`model.b` are created with `requires_grad=True` in 2.2b (not in
2.2a), which is what makes `.backward()` work.

### 2.3 — regression neuron, learning-rate sweep

[`eg2.3a.py`](eg2.3a.py) vs [`eg2.3b.py`](eg2.3b.py) — same neuron,
same 4 learning rates swept in parallel via `multiprocessing.Pool`, but they
differ in _when_ the weights get updated within an epoch.

**[eg2.3a.py](eg2.3a.py) — stochastic gradient descent (update per sample)**

```python
def my_train(alpha):
  model = Perceptron()
  cost = []
  idx = np.arange(no_data)
  XX, YY = X, Y
  for i in range(no_iters):
    np.random.shuffle(idx)
    XX, YY = XX[idx], YY[idx]
    cost_ = []
    for p in range(len(XX)):
      train(model, XX[p], YY[p], alpha)   # one weight update PER SAMPLE
      cost_.append(loss(model(XX[p]), torch.tensor(YY[p])).detach().numpy())
    cost.append(np.sum(cost_)/no_data)
  return cost
```

**[eg2.3b.py](eg2.3b.py) — batch gradient descent (update per epoch)**

```python
def my_train(rate):
  model = Perceptron()
  err = []
  for i in range(no_iters):
    train(model, X, Y, rate)              # one weight update per EPOCH, using all data at once
    loss_ = loss(model(X), torch.tensor(Y))
    err.append(loss_.detach().numpy())
  return err
```

Same `train()` weight-update rule in both (`w -= lr * w.grad`) — the
difference is only how much data each call to `train()` sees: one row at a
time in 2.3a (`XX[p], YY[p]`) vs. the whole `X, Y` matrix at once in 2.3b.

Both save a comparison plot (cost/loss vs. iteration, one line per learning
rate) to `figures/`.

### Why `.ipynb` for 2.2 but `.py` for 2.3

2.2 is meant to be stepped through cell-by-cell, inspecting `w`, `b`, and the
loss after each step — a notebook fits that workflow. 2.3 uses
`multiprocessing.Pool` under `if __name__ == '__main__':`, which doesn't play
well with Jupyter's process model, so it's a plain script that saves its
output figures to disk instead of displaying them inline.
