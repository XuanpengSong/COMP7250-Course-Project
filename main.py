# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:38:31 2024

@author: songx
"""

from models import VGG, MLP, GRU
from dataset import CIFAR10_dataloader, MNIST_dataloader, airquality_dataloader
from functions import train_vgg, train_mlp, train_gru
import torch
import torch.optim as optim
import random
import numpy as np
import os
import pandas as pd


seed = 3407
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def mlp_main(train_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    training_loss_list, train_acc_list, test_acc_list = [], [], []
    
    print("Training MLP on MNIST with SGD")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training MLP on MNIST with SGD momentum")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training MLP on MNIST with SGD with nesterov momentum")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training MLP on MNIST with AdaGrad")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training MLP on MNIST with RMSprop")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training MLP on MNIST with Adam")
    net = MLP(10)
    net.to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_mlp(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    return training_loss_list, train_acc_list, test_acc_list



def vgg_main(train_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    training_loss_list, train_acc_list, test_acc_list = [], [], []
    
    print("Training VGG on CIFAR-10 with SGD")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training VGG on CIFAR-10 with SGD momentum")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training VGG on CIFAR-10 with SGD with nesterov momentum")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training VGG on CIFAR-10 with AdaGrad")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training VGG on CIFAR-10 with RMSprop")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print("Training VGG on CIFAR-10 with Adam")
    net = VGG(10)
    net.to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    training_loss, train_acc, test_acc = train_vgg(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    return training_loss_list, train_acc_list, test_acc_list



def gru_main(train_loader, test_loader, sequence_length):
    criterion = torch.nn.MSELoss()
    training_loss_list, test_loss_list = [], []
    
    print("Training VGG on CIFAR-10 with SGD")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    print("Training VGG on CIFAR-10 with SGD momentum")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    print("Training VGG on CIFAR-10 with SGD with nesterov momentum")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    print("Training VGG on CIFAR-10 with AdaGrad")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    print("Training VGG on CIFAR-10 with RMSprop")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    print("Training VGG on CIFAR-10 with Adam")
    net = GRU(12, 32, 3, sequence_length)
    net.to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    training_loss, test_loss = train_gru(net, optimizer, criterion, train_loader, test_loader)
    training_loss_list.append(training_loss)
    test_loss_list.append(test_loss)
    
    return training_loss_list, test_loss_list



def main():
    if not os.path.exists("./results"):
        os.mkdir("./results")
    
    train_loader, test_loader = MNIST_dataloader(32)
    mlp_training_loss, mlp_training_acc, mlp_test_acc = mlp_main(train_loader, test_loader)
    mlp_results = {'training_loss': mlp_training_loss, 'training_acc': mlp_training_acc, 'test_acc': mlp_test_acc}
    mlp_results = pd.DataFrame(mlp_results)
    mlp_results.to_csv("./results/mlp_results.csv")
    
    train_loader, test_loader = CIFAR10_dataloader(128)
    vgg_training_loss, vgg_training_acc, vgg_test_acc = vgg_main(train_loader, test_loader)
    vgg_results = {'training_loss': vgg_training_loss, 'training_acc': vgg_training_acc, 'test_acc': vgg_test_acc}
    vgg_results = pd.DataFrame(vgg_results)
    vgg_results.to_csv("./results/vgg_results.csv")
    
    sequence_length = 24
    train_loader, test_loader = airquality_dataloader(32, sequence_length)
    gru_training_loss, gru_test_loss = gru_main(train_loader, test_loader, sequence_length)
    gru_results = {'training_loss': gru_training_loss, 'test_loss': gru_test_loss}
    gru_results = pd.DataFrame(gru_results)
    gru_results.to_csv("./results/gru_results.csv")


if __name__ == '__main__':
    main()



"""
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure()
plt.plot(range(len(train_acc)), training_loss)
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(train_acc)), test_acc)
plt.show()
"""



