# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:38:05 2024

@author: songx
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR



def test_mlp(model, test_dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in test_dataloader:
            inputs, labels = inputs.to(device).reshape(inputs.size(0), -1), labels.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
        accuracy = correct / len(test_dataloader.dataset)
    return accuracy



def train_mlp(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs=30, device="cuda"):
    scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    training_loss, train_acc, test_acc = np.zeros((num_epochs)), np.zeros((num_epochs)), np.zeros((num_epochs))
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (inputs, labels) in train_dataloader:
            inputs, labels = inputs.to(device).reshape(inputs.size(0), -1), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_training_loss = total_loss/len(train_dataloader)
        epoch_train_acc = test_mlp(model, train_dataloader, device)
        epoch_test_acc = test_mlp(model, test_dataloader, device)
        last_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}: Training accuracy: {epoch_train_acc:.4f} Test accuracy: {epoch_test_acc:.4f} LR: {round(last_lr[0], 5)}")
        training_loss[epoch] = epoch_training_loss
        train_acc[epoch] = epoch_train_acc
        test_acc[epoch] = epoch_test_acc
        scheduler.step()
    return training_loss, train_acc, test_acc



def test_vgg(model, test_dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
        accuracy = correct / len(test_dataloader.dataset)
    return accuracy



def train_vgg(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs=100, device="cuda"):
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    training_loss, train_acc, test_acc = np.zeros((num_epochs)), np.zeros((num_epochs)), np.zeros((num_epochs))
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (inputs, labels) in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_training_loss = total_loss/len(train_dataloader)
        epoch_train_acc = test_vgg(model, train_dataloader, device)
        epoch_test_acc = test_vgg(model, test_dataloader, device)
        last_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}: Training accuracy: {epoch_train_acc:.4f} Test accuracy: {epoch_test_acc:.4f} LR: {round(last_lr[0], 5)}")
        training_loss[epoch] = epoch_training_loss
        train_acc[epoch] = epoch_train_acc
        test_acc[epoch] = epoch_test_acc
        scheduler.step()
        
    return training_loss, train_acc, test_acc



def test_gru(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()
    average_test_loss = test_loss / len(test_dataloader)
    return average_test_loss



def train_gru(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs=80, device="cuda"):
    model.to(device)
    training_loss, test_loss = np.zeros((num_epochs)), np.zeros((num_epochs))
    scheduler = MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_training_loss = total_loss/len(train_dataloader)
        epoch_test_loss = test_gru(model, test_dataloader, criterion, device)
        last_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}: Training Loss: {epoch_training_loss:.4f} Test Loss: {epoch_test_loss:.4f} LR: {round(last_lr[0], 5)}")
        training_loss[epoch] = epoch_training_loss
        test_loss[epoch] = epoch_test_loss
        scheduler.step()
    return training_loss, test_loss


