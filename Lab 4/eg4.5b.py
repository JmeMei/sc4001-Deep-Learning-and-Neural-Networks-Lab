import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
figures_dir = os.path.expanduser('~/Dropbox/4001/4-dnn/figures')
if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

import multiprocessing as mp
import time

import numpy as np
import matplotlib.pylab as plt

SEED = 10
torch.manual_seed(SEED)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# set learning rate and number of epochs
learning_rate = 1e-2
epochs = 100

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss(reduction='sum')

# initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
    
    train_loss /= size
    correct /= size
    
    return train_loss, correct

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    
    return test_loss, correct

def my_train(batch_size):
    
    print(batch_size)
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_batches = len(train_dataloader)

    tt = 0
    for t in range(epochs):
        tt_ = time.time()
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        tt += time.time() - tt_
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

    
    te = (tt*1000)/epochs
    tb = (tt*1000)/(epochs*num_batches)

    paras = np.array([train_loss, test_acc, tb, te])
    
    return paras


def main():
  batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  paras = p.map(my_train, batch_sizes)

  paras = np.array(paras)
  entropy, accuracy, time_batch, time_epoch = paras[:,0], paras[:,1], paras[:, 2], paras[:, 3]

  plt.figure(1)
  plt.plot(np.log2(batch_sizes), entropy)
  plt.xticks(np.log2(batch_sizes), batch_sizes)
  plt.xlabel('log2(batch size)')
  plt.ylabel('entropy')
  plt.title('training entropy vs. log batch size')
  plt.savefig(os.path.join(figures_dir, '4.5b_1.png'))

  plt.figure(2)
  plt.plot(np.log2(batch_sizes), accuracy)
  plt.xticks(np.log2(batch_sizes), batch_sizes)
  plt.xlabel('log2(batch size)')
  plt.ylabel('test accuracy')
  plt.title('test accuracy vs. log batch size')
  plt.savefig(os.path.join(figures_dir, '4.5b_2.png'))

  plt.figure(3)
  plt.plot(np.log2(batch_sizes), time_batch)
  plt.xticks(np.log2(batch_sizes), batch_sizes)
  plt.xlabel('log2(batch size)')
  plt.ylabel('time (ms)')
  plt.title('time for a weight update vs. log batch size')
  plt.savefig(os.path.join(figures_dir, '4.5b_3.png'))
  
  plt.figure(4)
  plt.plot(np.log2(batch_sizes), time_epoch)
  plt.xticks(np.log2(batch_sizes), batch_sizes)
  plt.xlabel('log2(batch size)')
  plt.ylabel('time (ms)')
  plt.title('time for an epoch vs. log batch size')
  plt.savefig(os.path.join(figures_dir, '4.5b_4.png'))
 


if __name__ == '__main__':
  main()






