# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:22:50 2024

@author: songx
"""


import torch
from torch import nn
import torch.nn.functional as F


#MLP for MNIST dataset
class MLP(nn.Module):
    def __init__(self, output_dim=10):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


#VGG-liked CNN for CIFAR-10
#References from: 
#Simonyan, Karen and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” CoRR abs/1409.1556 (2014): n. pag.
class VGG(nn.Module):
    def __init__(self, output_dim):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 512, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



#GRU, references from pytorch website:
#https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(2 * hidden_size * sequence_length, 128), 
                                nn.ReLU(), nn.Linear(128, input_size))
    
    def forward(self, x):
        output, _ = self.gru(x)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output


