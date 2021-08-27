#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Program used to read txt files containing raw data, perform FFT, find top 18 peaks between index of 80 and 1000

from scipy.fft import fft,fftfreq
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter
import heapq


# In[3]:


# 0mb data - 151 files starting at 0m
num = 0
test = []
lines=[]
yf=[]
ft_l = []
ft = []
# Number of sample points
N = 8192
for i in range(0,151):

    with open('../ProjectData/Conor Sherlock/TrainingData0mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
        test.append(lines[i])
    file.close()

    # take from 106 to 1130 after getting fft
#     ft = abs(fft(lines[i]))[106:1130]
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(0)
    yf.append(ft_l)
    
    # Filename goes up in increments of 2
    num = num + 2
    
# 1mb data - 175 files starting at 352m
num = 352
lines=[]
ft_l = []
ft = []
for i in range(0,175):

    with open('../ProjectData/Conor Sherlock/TrainingData1mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
    file.close()
    
    # take from 106 to 1130 after getting fft
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(1)
    yf.append(ft_l)
    
    # Filename goes up in increments of 4
    num = num + 4

# 2mb data - 176 files starting at 1050m
num = 1050
lines=[]
ft_l = []
ft = []
for i in range(0,176):

    with open('../ProjectData/Conor Sherlock/TrainingData2mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
    file.close()
    
    # take from 106 to 1130 after getting fft
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(2)
    yf.append(ft_l)
    
    # Filename goes up in increments of 4
    num = num + 4
    
# 3mb data - 175 files starting at 1754m
num = 1754
lines=[]
ft_l = []
ft = []
for i in range(0,175):

    with open('../ProjectData/Conor Sherlock/TrainingData3mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
    file.close()
    
    # take from 106 to 1130 after getting fft
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(3)
    yf.append(ft_l)
    
    # Filename goes up in increments of 4
    num = num + 4

# 4mb data - 175 files starting at 2454m
num = 2454
lines=[]
ft_l = []
ft = []
for i in range(0,175):

    with open('../ProjectData/Conor Sherlock/TrainingData4mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
    file.close()
    
    # take from 106 to 1130 after getting fft
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(4)
    yf.append(ft_l)
    
    # Filename goes up in increments of 4
    num = num + 4
# 5mb data - 175 files starting at 3154m
num = 3154
lines=[]
ft_l = []
ft = []
for i in range(0,175):

    with open('../ProjectData/Conor Sherlock/TrainingData5mbar/'+str(num)+'m.txt','r') as file:
        lines.append([line.rstrip('\n') for line in file])
    file.close()
    
    # take from 106 to 1130 after getting fft
    ft = abs(fft(lines[i]))[80:1000]
    peaks,_ = find_peaks(ft,distance=20)
#     ft_l = ft.tolist()
    ft_l = heapq.nlargest(18, enumerate(ft[peaks]), key=itemgetter(1))
    ft_l = [val for (i, val) in sorted(ft_l)]
    ft_l.append(5)
    yf.append(ft_l)
    
    # Filename goes up in increments of 4
    num = num + 4
    

import csv
with open('data reduced.csv','w',newline='') as csvfile:
    write = csv.writer(csvfile)
    for l in yf:
        write.writerow(l)


# In[6]:


# Print example of peak detection

ft_this = abs(fft(lines[0]))[80:1000]
peaks,_ = find_peaks(ft_this,distance=20)
top32_peaks = heapq.nlargest(18, enumerate(ft_this[peaks]), key=itemgetter(1))
top32_peaks = [val for (i, val) in sorted(top32_peaks)]
peak_index = []
ft_list=ft_this.tolist()
for i, val in enumerate(top32_peaks):
    peak_index.append(ft_list.index(val))
    

    
plt.plot(peak_index, ft_this[peak_index], "ob"); plt.plot(ft_this);
plt.xlabel('index')
plt.ylabel('Power (mW)')
plt.title("Peak Detection - Top 18 Peaks (5mbar of pressure)")
plt.grid()
plt.axis([0,940,0,13.5])
plt.show()


# In[ ]:




