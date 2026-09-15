# SC4001 Notes

## `Lab 3/` — single-layer neural net examples (forward pass vs. training)

Both examples build a single layer (2 inputs → 3 outputs) on toy 2D data,
but 3.1 only runs a forward pass while 3.2 actually trains the layer as a
3-class classifier.

### 3.1 — perceptron layer, forward pass only

[`eg3.1.ipynb`](eg3.1.ipynb)

Defines a `PerceptronLayer` with **sigmoid** activation and runs a single
forward pass on 4 sample points. There is no loss, no training loop, and no
notion of "classes" — each of the 3 output neurons independently squashes
its own logit into $(0,1)$:

```
u = X·w + b            (pre-activation / logits, one per output neuron)
y = 1 / (1 + e^-u)      (sigmoid, applied independently per neuron)
```

The 3 outputs do **not** sum to 1 — they're 3 unrelated probabilities, not
a distribution over classes.

### 3.2 — softmax layer, trained as a 3-class classifier

[`eg3.2.ipynb`](eg3.2.ipynb)

Defines a `Softmax_Layer` and trains it with gradient descent (1500 epochs)
to classify 18 2D points into 3 classes (A/B/C), then derives and plots the
decision boundaries between classes.

**Forward pass:**

```
u = X·w + b                        (logits)
p_i = e^{u_i} / Σ_j e^{u_j}        (softmax → probability distribution over classes)
ŷ = argmax_i p_i                   (predicted class)
```

Unlike 3.1's sigmoid, softmax outputs **do** sum to 1, making `p` a proper
probability distribution — the right choice when each sample belongs to
exactly one class.

**Cross-entropy loss** (with one-hot targets `k`):

```
J = -Σ_p Σ_i k_{p,i}·log(p_{p,i})
```

**Gradient descent update** — softmax + cross-entropy gives a clean
gradient w.r.t. the logits, `grad_u = p - k`:

```
grad_u = p - k
grad_w = Xᵀ·grad_u
grad_b = Σ_p grad_u
w ← w - α·grad_w
b ← b - α·grad_b
```

This is the same update-rule shape as the manual logistic regression in
[`Lab 2/eg2.2a.ipynb`](../Lab%202/eg2.2a.ipynb) — softmax + cross-entropy is
just the multi-class generalization of sigmoid + binary cross-entropy.

**Decision boundaries** — after training, the boundary between class `i`
and class `i+1` is where their logits are equal (`u_i = u_{i+1}`), which
reduces to a line `ww·x + bb = 0` in the `(x1, x2)` plane, where
`ww = w_i - w_{i+1}` and `bb = b_i - b_{i+1}` (cyclic, so class 2 pairs
back with class 0). The notebook converts each line to slope-intercept
form (`m`, `c`) and plots all 3 boundaries against the training points.

### Why 3.1 has no loss/training but 3.2 does

3.1 exists to show what a layer's forward computation looks like in
isolation — useful for checking shapes and the sigmoid activation before
anything more complex is layered on. 3.2 builds on that same forward-pass
pattern but adds the pieces needed for actual learning: a loss function
that measures how wrong the predictions are, and a training loop that uses
the loss's gradient to update `w` and `b` over many epochs.
