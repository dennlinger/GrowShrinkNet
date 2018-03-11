#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:13:54 2018

@author: dennis
"""

import torch as t

""" I am aware that this could be done much more elegant than the current
    implementation by simply inheriting from more general classes.
    I refrain from doing so to keep things perfectly clear and make it easier to
    dive right into a single model.
"""

class gmsCNN(t.nn.Module):
    # needs extension to be GS
    def __init__(self, kernelg=3, kernels=7, kernel=5, num_filters = 8, rate = 2):
        super(gmsCNN, self).__init__()
        self.padding = kernel//2
        self.paddings = kernels//2
        self.paddingg = kernelg//2
        # Define layers
        self.pad_rate = rate//2
        self.rate = rate


        # shrink
        self.shrink1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernels, padding=self.paddings),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
                
                
        self.shrink2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernels-self.rate, padding=self.paddings-self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
                
        self.shrink3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernels-2*self.rate, padding=self.paddings-2*self.pad_rate),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())

        # same
        self.conv1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.conv2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.conv3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())
        
        # grow
        self.grow1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernelg, padding=self.paddingg),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.grow2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernelg+self.rate, padding=self.paddingg+ self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.grow3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernelg+2*self.rate, padding=self.paddingg+2*self.pad_rate),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())

        # rest
        self.fuse = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*12, out_channels=num_filters*4, kernel_size=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())
                
        self.fc1 = t.nn.Sequential(
                t.nn.Linear(num_filters*4*(4*4),128),
                t.nn.ReLU())
        
        self.fc2 = t.nn.Sequential(
                t.nn.Linear(128,128),
                t.nn.ReLU())
        
        self.soft = t.nn.Linear(128,10)
        
    def forward(self, x):
        # shrink part
        self.sout1 = self.shrink1(x)
        self.sout2 = self.shrink2(self.sout1)
        self.sout3 = self.shrink3(self.sout2)
        # middle part
        self.convout1 = self.conv1(x)
        self.convout2 = self.conv2(self.convout1)
        self.convout3 = self.conv3(self.convout2)
        # grow part
        self.gout1 = self.grow1(x)
        self.gout2 = self.grow2(self.gout1)
        self.gout3 = self.grow3(self.gout2)
        # fusing it together
        self.stacked = self.fuse(t.cat((self.gout3, self.convout3, self.sout3), dim=1))
        # fc part
        self.fcview = self.sout3.view(self.stacked.size(0), -1)
        self.fcout1 = self.fc1(self.fcview)
        self.fcout2 = self.fc2(self.fcout1)
        self.final = self.soft(self.fcout2)
        
        return self.final


class gsCNN(t.nn.Module):
    # needs extension to be GS
    def __init__(self, kernelg=3, kernels=7, num_filters = 8, rate = 2):
        super(gsCNN, self).__init__()
        self.paddings = kernels//2
        self.paddingg = kernelg//2
        # Define layers
        self.pad_rate = rate//2
        self.rate = rate

        
        
        # shrink
        self.shrink1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernels, padding=self.paddings),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
                
                
        self.shrink2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernels-self.rate, padding=self.paddings-self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
                
        self.shrink3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernels-2*self.rate, padding=self.paddings-2*self.pad_rate),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())

        # grow
        self.grow1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernelg, padding=self.paddingg),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.grow2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernelg+self.rate, padding=self.paddingg+ self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.grow3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernelg+2*self.rate, padding=self.paddingg+2*self.pad_rate),
                t.nn.MaxPool2d(2, padding=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())

        # rest
        self.fuse = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*8, out_channels=num_filters*4, kernel_size=1),
                t.nn.BatchNorm2d(num_filters*4),
                t.nn.ReLU())
                
        self.fc1 = t.nn.Sequential(
                t.nn.Linear(num_filters*4*(4*4),128),
                t.nn.ReLU())
        
        self.fc2 = t.nn.Sequential(
                t.nn.Linear(128,128),
                t.nn.ReLU())
        
        self.soft = t.nn.Linear(128,10)
        
    def forward(self, x):
        # shrink part
        self.sout1 = self.shrink1(x)
        self.sout2 = self.shrink2(self.sout1)
        self.sout3 = self.shrink3(self.sout2)
        # grow part
        self.gout1 = self.grow1(x)
        self.gout2 = self.grow2(self.gout1)
        self.gout3 = self.grow3(self.gout2)
        # fusing it together
        self.stacked = self.fuse(t.cat((self.gout3, self.sout3), dim=1))
        # fc part
        self.fcview = self.sout3.view(self.stacked.size(0), -1)
        self.fcout1 = self.fc1(self.fcview)
        self.fcout2 = self.fc2(self.fcout1)
        self.final = self.soft(self.fcout2)
        
        return self.final
        
class gCNN(t.nn.Module):
    # increases kernel size for each convolutional layer
    def __init__(self, kernel=3, num_filters = 8, rate = 2):
        super(gCNN, self).__init__()
        self.padding = kernel//2
        self.pad_rate = rate//2
        self.rate = rate
        
        self.conv1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.conv2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernel+self.rate, padding=self.padding+ self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.conv3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernel+2*self.rate, padding=self.padding+2*self.pad_rate),
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
        
class sCNN(t.nn.Module):
    # decreases kernel size for each convolutional layer
    def __init__(self, kernel=7, num_filters = 8, rate = 2):
        super(sCNN, self).__init__()
        self.padding = kernel//2
        self.pad_rate = rate//2
        self.rate = rate
        
        self.conv1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.conv2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernel-self.rate, padding=self.padding-self.pad_rate),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.conv3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernel-2*self.rate, padding=self.padding-2*self.pad_rate),
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
        
    
class CNN(t.nn.Module):
    def __init__(self, kernel=3, num_filters = 8):
        super(CNN, self).__init__()
        self.padding = kernel//2
        
        self.conv1 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters),
                t.nn.ReLU())
        
        self.conv2 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2,
                            kernel_size=kernel, padding=self.padding),
                t.nn.MaxPool2d(2),
                t.nn.BatchNorm2d(num_filters*2),
                t.nn.ReLU())
        
        self.conv3 = t.nn.Sequential(
                t.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4,
                            kernel_size=kernel, padding=self.padding),
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
