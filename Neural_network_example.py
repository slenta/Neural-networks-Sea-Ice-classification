#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:52:45 2021

@author: S and J
"""

#Example of a neural network to recognize handwritten numbers from 1 to 3, to simplfy the architecture and reduce calculation power, we will only
#look at numbers from 1 to 3, but the principle of course remains the same in any higher version

import numpy as np
import matplotlib.pyplot as plt

#defining the sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x))

#differential of the sigmoid function for later training purposes


def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#defining the activation function as sigmoid funciton of the sum of all weights multiplied by the outputs of the previous layer

def layer_output(L,W):
    return sigmoid(np.dot(L,W))

#defining model inputs of handwritten numbers from 1 to 3, in this case organised in a 5 x 4 matrix to simplify calculation 

L0 = np.array([[0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0],
[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0] ])

#defining model outputs for training purposes

L2_aim = np.array([[1,0,0],[0,1,0],[0,0,1]])

#defining randomized weights for input and hidden layer

W0 = np.random.random((20,30))
W1 = np.random.random((30,3))

#optimizing weights until the cost function falls below a threshold of 0.001

for ix in range(60000):
    L1 = layer_output(L0, W0)
    L2 = layer_output(L1, W1)
    
    L2_error = L2_aim - L2
    
    if np.max(np.abs(L2_error)) < 0.001:
        break
    
    #adjusting the weights at each step to minimize the cost function utilizing its differential
    
    W1 = W1 + np.dot(L1.T, L2_error*d_sigmoid(L2))
    W0 += np.dot(L0.T,np.dot(L2_error * d_sigmoid(L2),W1.T)* d_sigmoid(L1))
    
    
    
#testing the networks abilities with the patter corresponding to the number 2  
    
L0 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
L1 = layer_output(L0,W0)
L2 = layer_output(L1,W1)
print(L2)