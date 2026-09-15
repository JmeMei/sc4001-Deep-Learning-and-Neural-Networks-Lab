import numpy as np
import pylab as plt
import torch
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')


no_data = 16
no_iters = 500
SEED = 10
np.random.seed(SEED)

# training data
X = np.random.rand(no_data,2)
Y = 1.0 +3.3*X[:,0]**2-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(no_data,1)

# a class for the preceptron
class Perceptron():
  def __init__(self):
    self.w = torch.tensor(np.random.rand(2,1), requires_grad=True)
    self.b = torch.tensor(0., requires_grad=True)

  def __call__(self, x):
    u = torch.matmul(torch.tensor(x), self.w) + self.b
    y = 6.0*torch.sigmoid(u)-1.5
    return y
  

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

def my_train(rate):
 
  model = Perceptron()
  print(rate)

  err = []
  for i in range(no_iters):
    train(model, X, Y, rate)
    loss_= loss(model(X), torch.tensor(Y))
    err.append(loss_.detach().numpy())
    
    if not i%10:
      print(i, err[i])

  return err
  

if __name__ == '__main__':
        
    
        no_threads = mp.cpu_count()
        rates = [0.005, 0.01, 0.05, 0.1]

        with mp.Pool(processes=no_threads) as p:
          results = p.map(my_train, rates)

        plt.figure()
        for i in range(len(rates)):
          plt.plot(range(no_iters), results[i], label='lr = {}'.format(rates[i]))
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.title('gradient descent')
        plt.legend()
        plt.savefig("./2-regression_classification/figures/2.3b.png")
        plt.close()

        