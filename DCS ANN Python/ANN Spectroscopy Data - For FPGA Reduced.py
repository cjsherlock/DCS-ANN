#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Neural network to detect ammonia pressure in a gas sample - largest 18 peaks after performing FFT on input data
# Loading model to convert weight, biases, and inputs to VHDL

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


# In[3]:


# Redifne model to load in later

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
        z = self.out(z)
        z = self.softmax(z)
        return z


# In[4]:


# Define and load model
model = Net()
model.load_state_dict(torch.load("ANN Spectroscopy Data Model - Pruned.pth"))


# In[6]:


#Adjust weights, biases, and inputs with a constant multiplier of 2^7

multiplier = 2**7

model.out.bias.data = model.out.bias.data*multiplier*multiplier
model.out.weight.data = model.out.weight.data*multiplier
model.hid1.weight.data = model.hid1.weight.data*multiplier
model.hid1.bias.data = model.hid1.bias.data*multiplier*multiplier

with torch.no_grad():

    for i,tens in enumerate(model.hid1.weight):
        for j,val in enumerate(tens):
            model.hid1.weight[i][j] = int(val)
            
    for i,tens in enumerate(model.out.weight):
        for j,val in enumerate(tens):
            model.out.weight[i][j] = int(val)
            
    for i,val in enumerate(model.hid1.bias):
        model.hid1.bias[i] = int(val)
        
    for i,val in enumerate(model.out.bias):
        model.out.bias[i] = int(val)


# In[7]:


# Prepare Dataset
dataframe = pd.read_csv("data reduced.csv",sep=',',header=None,dtype = np.float32)
features, targets = dataframe.iloc[:, :-1].values, dataframe.iloc[:, [-1]].values

#Normalize data between 0 and the multiplier
scaler = MinMaxScaler(feature_range=(0, multiplier))
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

#Convert inputs to integer values
for i,val in enumerate(featuresTest):
    for j,entry in enumerate(val):
        featuresTest[i][j] = int(entry)
        
for i,val in enumerate(featuresTrain):
    for j,entry in enumerate(val):
        featuresTrain[i][j] = int(entry)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
trainloader = DataLoader(train,shuffle=True)
testloader = DataLoader(test)


# In[8]:


# Test the network with integer values
model.eval()

class_correct = [0 for _ in range(6)]
total = [0 for _ in range(6)]
classes = [0,1,2,3,4,5]
predictions_list = []
target_list = []
model.eval()
with torch.no_grad():
    for X, y in testloader:
        target_list.append(y)
        outputs = model(X)
        _,predicted = torch.max(outputs, 1)
        predictions_list.append(predicted)
        class_correct[y.item()] += (predicted == y).item()
        total[y.item()] += 1
        
for i in range(6):
    print("Accuracy of {}: {:.2f}%".format(classes[i], class_correct[i] * 100 / total[i]))
    
print(class_correct)
print(total)
print("Overall Accuracy: {:.4f}%".format(sum(class_correct)/sum(total)*100))


# In[ ]:


#Print weights, biases, and inputs to txt files to a format recognized by VHDL

file1 = open("hid_weights.txt","w")

file1.write("(\n")
for tensor in model.hid1.weight:
    file1.write("(")
    for element in tensor:
        file1.write(str(int((multiplier)*element.item()))+",")
    file1.write("),\n")
file1.write(");")        
file1.close()

file2 = open("hid_bias.txt","w")
file2.write("(")
for val in model.hid1.bias:
    file2.write(str(int((multiplier*multiplier)*val.item()))+",")
file2.write(");")
file2.close()

file3 = open("output_weights.txt","w")

file3.write("(\n")
for tensor in model.out.weight:
    file3.write("(")
    for element in tensor:
        file3.write(str(int((multiplier)*element.item()))+",")
    file3.write("),\n")
file3.write(");")    
        
file3.close()

file4 = open("output_bias.txt","w")

file4.write("(")
for val in model.out.bias:
    file4.write(str(int((multiplier*multiplier)*val.item()))+",")

file4.write(");")
file4.close()

file5 = open("test inputs.txt","w")

for tens in features_test:
    for val in tens:
        file5.write(str(int(val.item()))+"\n")

file5.close()


# In[ ]:




