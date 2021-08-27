#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Neural network to detect ammonia pressure in a gas sample - largest 18 peaks after performing FFT on input data

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
import pandas as pd
from timeit import default_timer as timer

# Move device to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[2]:


# Prepare Dataset
dataframe = pd.read_csv("data reduced.csv",sep=',',header=None,dtype = np.float32)
features, targets = dataframe.iloc[:, :-1].values, dataframe.iloc[:, [-1]].values

#Normalize data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

# train test split - 80/20
features_train, features_test, targets_train, targets_test = train_test_split(features,targets, train_size = 0.8, 
                                                                              test_size = 0.2,random_state = 3) 

# create feature and targets tensor for train set
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).squeeze(1).type(torch.LongTensor)

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).squeeze(1).type(torch.LongTensor)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
trainloader = DataLoader(train,shuffle=True)
testloader = DataLoader(test)


# In[3]:


# Define Neural Network class
# 18 -> (8) -> 6

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = nn.Linear(18,11)   
        self.out = nn.Linear(11, 6)
        
        # Define relu and softmax activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.relu(self.hid1(x))
        z = self.softmax(self.out(z))
        return z


# In[4]:


# Define model, move to GPU
model = Net()
# model.to(device)

# Define hyperparameters (Using Cross Entropy loss and SGD optimiser)
epochs = 50
loss_fn = nn.CrossEntropyLoss()
# optimiser = optim.SGD(model.parameters(), lr=0.008)
optimiser = optim.Adam(model.parameters(), lr=0.0005)

#List to store losses
losses = []

# Training loop
# Set model to training mode (Update weights)
model.train()
for i in range(epochs):
    for i, (X, y) in enumerate(trainloader):
        # Move data to GPU 
        X, y = Variable(X).to(device), Variable(y).to(device)      
        #Clear the previous gradients
        optimiser.zero_grad()
        #Precit the output for Given input
        y_pred = model(X)
        #Compute Cross entropy loss
        loss = loss_fn(y_pred,y)
        #Add loss to the list
        losses.append(loss.item())
        #Compute gradients
        loss.backward()
        #Adjust weights
        optimiser.step()


# In[5]:


class_correct = [0 for _ in range(6)]
total = [0 for _ in range(6)]
classes = [0,1,2,3,4,5]
predictions_list = []
target_list = []

model.eval()
with torch.no_grad():
    for X, y in testloader:
        X, y = X.to(device), y.to(device)
        target_list.append(y)
        outputs = model(X)
        _,predicted = torch.max(outputs, 1)
        predictions_list.append(predicted)
        class_correct[y.item()] += (predicted == y).item()
        total[y.item()] += 1
        
for i in range(6):
    print("Accuracy of Class {}: {:.2f}%".format(classes[i], class_correct[i] * 100 / total[i]))
    
print("Correct per class:     ",class_correct)
print("Total Classes in data: ",total)
print("Overall Accuracy: {:.4f}%".format(sum(class_correct)/sum(total)*100))

torch.save(model.state_dict(),"ANN Spectroscopy Data Model - Pruned.pth")


# In[6]:


from itertools import chain 

predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [target_list[i].tolist() for i in range(len(target_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

import sklearn.metrics as metrics

metrics.confusion_matrix(labels_l, predictions_l)
print("Classification report for Network :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))

