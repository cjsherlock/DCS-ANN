#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
import pandas as pd
import numpy as np


# In[2]:


# Prepare Dataset
dataframe = pd.read_csv("data reduced.csv",sep=',',header=None,dtype = np.float32)
features, targets = dataframe.iloc[:, :-1].values, dataframe.iloc[:, [-1]].values

multiplier = 2**7

#Normalize data between 0 and (Multiplier)
scaler = MinMaxScaler(feature_range=(0, multiplier))
features = scaler.fit_transform(features)

for i,val in enumerate(features):
    for j,entry in enumerate(val):
        features[i][j] = int(entry)
        

# for val in features:
#     print(val)

# # train test split - 80/20
features_train, features_test, targets_train, targets_test = train_test_split(features,targets, train_size = 0.8, 
                                                                              test_size = 0.2,random_state = 3) 



# # create feature and targets tensor for train set
# featuresTrain = torch.from_numpy(features_train)
# targetsTrain = torch.from_numpy(targets_train).squeeze(1).type(torch.LongTensor)

# # create feature and targets tensor for test set.
# featuresTest = torch.from_numpy(features_test)
# targetsTest = torch.from_numpy(targets_test).squeeze(1).type(torch.LongTensor)

# train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
# test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# # data loader
# trainloader = DataLoader(train,shuffle=True)
# testloader = DataLoader(test)


# In[3]:


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


model = Net()
model.load_state_dict(torch.load("ANN Spectroscopy Data Model - Pruned.pth"))
model.eval()


# In[5]:


file1 = open("hid_weights.txt","w")

file1.write("(\n")
for tensor in model.hid1.weight:
    file1.write("(")
    for element in tensor:
        file1.write(str(int((multiplier)*element.item()))+",")
    file1.write("),\n")
file1.write(");")        
file1.close()


# In[6]:


file2 = open("hid_bias.txt","w")
file2.write("(")
for val in model.hid1.bias:
    file2.write(str(int((multiplier*multiplier)*val.item()))+",")
file2.write(");")
file2.close()


# In[7]:


file3 = open("output_weights.txt","w")

file3.write("(\n")
for tensor in model.out.weight:
    file3.write("(")
    for element in tensor:
        file3.write(str(int((multiplier)*element.item()))+",")
    file3.write("),\n")
file3.write(");")    
        
file3.close()


# In[8]:


file4 = open("output_bias.txt","w")

file4.write("(")
for val in model.out.bias:
    file4.write(str(int((multiplier*multiplier)*val.item()))+",")

file4.write(");")
file4.close()


# In[9]:


file5 = open("test inputs.txt","w")

for tens in features_test:
    for val in tens:
        file5.write(str(int(val.item()))+"\n")

file5.close()


# In[10]:


file6 = open("test inputs find.txt","w")

for tens in features_test:
    for val in tens:
        file6.write(str(int(val.item()))+",")
    file6.write('\n')

file6.close()


# In[15]:


file7 = open("test targets.txt","w")

for val in targets_test:
    file7.write(str(int(val.item()))+"\n")
file7.close()


# In[ ]:




