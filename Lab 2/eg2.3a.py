import torch
import numpy as np
import pylab as plt
import multiprocessing as mp
from functools import partial

import os
if not os.path.isdir('figures'):
	os.makedirs('figures')


print(torch.__version__)

#Check for GPU and set device
device = (
    "cuda" if torch.cuda.is_available() else "mps"  # macbook uses metal performance shaders to GPU accelearation
    if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

no_data = 16
no_iters = 500
SEED = 10
np.random.seed(SEED)

## training data
X = np.random.rand(no_data, 2)
Y = 1.0 +3.3*X[:,0]**2-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(no_data)

# a class for the preceptron
class Perceptron():
  def __init__(self):
    self.w = torch.tensor(0.01*np.random.rand(2), requires_grad=True)
    self.b = torch.tensor(0., requires_grad=True)

  def __call__(self, x):
    u = torch.inner(torch.tensor(x), self.w) + self.b
    y = 6.0*torch.sigmoid(u)-1.5
    return y


# squared error as the loss function
loss = torch.nn.MSELoss(reduction="sum")

def train(model, inputs, outputs, lr): 
  y_ = model(inputs)
  
  loss_ = loss(y_, torch.tensor(outputs))
    
  loss_.backward() # claculate gradients with respect to tensor variables
    
  with torch.no_grad():
    model.w -= lr * model.w.grad
    model.b -= lr * model.b.grad
    
  model.w.grad = None
  model.b.grad = None


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
      train(model, XX[p], YY[p], alpha)
      cost_.append(loss(model(XX[p]), torch.tensor(YY[p])).detach().numpy())
    cost.append(np.sum(cost_)/no_data)

    if not i%10:
      print(i, cost[i])

  return cost


if __name__ == '__main__':

  rates = [0.005, 0.01, 0.05, 0.1]

  no_threads = mp.cpu_count()
  with mp.Pool(processes=no_threads) as p:
    costs = p.map(my_train, rates)

  plt.figure()
  for r in range(len(rates)):
    plt.plot(range(no_iters), costs[r], label='lr = {}'.format(rates[r]))
  plt.xlabel('iterations')
  plt.ylabel('cost')
  plt.title('stochastic gradient descent')
  plt.legend()
  plt.savefig("2-regression_classification/figures/2.3a.png")
  plt.close()


