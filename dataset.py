# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:39:37 2024

@author: songx
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo 


def MNIST_dataloader(batchsize, download=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    traindata = torchvision.datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    testdata = torchvision.datasets.MNIST(root='./data', train=False, download=download, transform=transform)
    
    train_dataloader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=False)
    
    return train_dataloader, test_dataloader


def CIFAR10_dataloader(batchsize, download=True):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    traindata = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform_train)
    testdata = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=False)
    return train_dataloader, test_dataloader


#download the Air Quality dataset from UCI
def airquality_download():
    air_quality = fetch_ucirepo(id=360) 
    X = air_quality.data.features.drop(["NMHC(GT)"], axis=1)
    X = X.iloc[:,2:]
    X = X.dropna()
    
    train_len = int(len(X) * 0.7)
    train_data, test_data = X.iloc[:train_len, :], X.iloc[train_len:, :]
    
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    
    try:
        os.mkdir("./data/air_quality")
    except:
        None
    pd.DataFrame(train_data).to_csv("./data/air_quality/train_data.csv", index=False)
    pd.DataFrame(test_data).to_csv("./data/air_quality/test_data.csv", index=False)


class csv_dataset(Dataset):
    def __init__(self, file_name, sequence_length):
        self.data = pd.read_csv(file_name)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        feature = self.data.iloc[idx:idx+self.sequence_length,:].values.astype('float32')
        label = self.data.iloc[idx+self.sequence_length+1, :].values.astype('float32')

        feature = torch.tensor(feature)
        label = torch.tensor(label)

        return feature, label


def airquality_dataloader(batchsize, sequence_length, download=True):
    if download:
        airquality_download()
    traindata = csv_dataset("./data/air_quality/train_data.csv", sequence_length)
    testdata = csv_dataset("./data/air_quality/test_data.csv", sequence_length)

    train_dataloader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=False)

    return train_dataloader, test_dataloader
