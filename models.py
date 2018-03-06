#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:13:54 2018

@author: dennis
"""

import torch as t


class gsCNN(t.nn.Module):
    # needs extension to be GS
    def __init__(self, kernelg=3, kernels=7, num_filters = 8, rate = 2):
        super(gsCNN, self).__init__()
        paddings = kernels//2
        paddingg = kernelg//2
        # Define layers
        pad_rate = rate//2
        
        # Growth layers first
        self.shrink1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernels, padding=paddings),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
                
                
        self.shrink2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernels-rate, padding=paddings-pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
                
        self.shrink3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernels-2*rate, padding=paddings-2*pad_rate),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())
                
        self.fc1 = t.nn.Sequential(
                t.nn.Linear(4*4*32,128),
                t.nn.ReLU())
        
        self.fc2 = t.nn.Sequential(
                t.nn.Linear(128,128),
                t.nn.ReLU())
        
        self.soft = t.nn.Sequential(
                t.nn.Linear(128,10),
                t.nn.Softmax())
        
    def forward(self, x):
        self.sout1 = self.shrink1(x)
        self.sout2 = self.shrink2(self.sout1)
        self.sout3 = self.shrink3(self.sout2)
        self.fcview = self.sout3.view(self.sout3.size(0), -1)
        self.fcout1 = self.fc1(self.fcview)
        self.fcout2 = self.fc2(self.fcout1)
        self.final = self.soft(self.fcout2)
        
        return self.final
        
class CNN(t.nn.Module):
    def __init__(self, kernel=3, num_filters = 8):
        super(CNN, self).__init__()
        padding = kernel//2
        
        self.conv1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernel, padding=padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.conv2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernel, padding=padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.conv3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernel, padding=padding),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())
                
        self.fc1 = t.nn.Sequential(
                t.nn.Linear(num_filters*4*(4*4),128),
                t.nn.ReLU())
        
        self.fc2 = t.nn.Sequential(
                t.nn.Linear(128,128),
                t.nn.ReLU())
        
        self.soft = t.nn.Linear(128,10)
                # don't need softmax since it's included in loss
#                t.nn.Softmax())
        
    def forward(self, x):
        self.convout1 = self.conv1(x)
        self.convout2 = self.conv2(self.convout1)
        self.convout3 = self.conv3(self.convout2)
        self.flatview = self.convout3.view(self.convout3.size(0),-1)
        self.fcout1 = self.fc1(self.flatview)
        self.fcout2 = self.fc2(self.fcout1)
        self.final = self.soft(self.fcout2)
        
        return self.final