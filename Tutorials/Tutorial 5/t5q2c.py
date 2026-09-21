import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from common_utils import NeuralNetwork_dropout
from common_utils import train_loop, test_loop
from common_utils import device, set_seed

import pandas as pd
from pathlib import Path
import os

import ray
from ray import tune
from ray.tune import Checkpoint, Tuner

batch_size = 256
max_epochs = 100

lr = 0.001
patience = 10
hidden_size = 400

set_seed(0)
print(f"Using {device} device")

from ray.train.torch import enable_reproducibility
enable_reproducibility(seed=42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


param_space = {
    "dropouts": tune.grid_search([0.1, 0.2, 0.3, 0.4]),
    "weight_decay": tune.grid_search([0.0001, 0.001, 0.005, 0.01])
}

def train(config):
    
    print(f"Starting training with config: {config}")
    os.makedirs('./data', exist_ok=True)
    
    data_root = '/Users/admin/Dropbox/4001/5-model_select_overfitting/data'
    try:
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Failed to load CIFAR10: {e}")
        raise e

    model = NeuralNetwork_dropout(hidden_size=hidden_size, drop_out = config["dropouts"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=config["weight_decay"])

    start_epoch = 0
   
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss, train_correct = train_loop(train_dataloader, model, loss_fn, optimizer)

        model.eval()
        test_loss, test_correct = test_loop(test_dataloader, model, loss_fn)

        tune.report({
            "test_accuracy": test_correct,
            "test_loss": test_loss
        })
        print(f"Epoch {epoch}: reported test_loss={test_loss:.4f}, test_accuracy={test_correct:.4f}")

def main():
    
    data_root = '/Users/admin/Dropbox/4001/5-model_select_overfitting/data'
    os.makedirs(data_root, exist_ok=True)
    print("Downloading CIFAR10 data if not present...")
    try:
        torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        print("Data ready.")
    except Exception as e:
        print(f"Failed to download CIFAR10: {e}")
        print("Please download CIFAR10 manually or check network connection.")
        return
    
    tuner = tune.Tuner(train, param_space=param_space)
    results = tuner.fit()

    for i, result in enumerate(results):
        print(f"Trial {i} metrics: {result.metrics}")

    best_result = results.get_best_result("test_loss", mode="min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['test_loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['test_accuracy']}")

    # After running results = tuner.fit()
    df = results.get_dataframe()

    # View specific hyperparameter columns and result metrics
    df = df[["config/dropouts", "config/weight_decay", "test_loss", "test_accuracy"]]
    df.to_csv("ray_results.csv", index=False)


if __name__ == "__main__":
    main()