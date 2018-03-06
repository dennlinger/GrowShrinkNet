#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:25:12 2018

@author: Dennis Aumilelr
"""

import argparse
import time

import torch as t
from torch.autograd import Variable

import torchvision as tv
import numpy as np
from models import CNN, gsCNN, gCNN, sCNN#, gmsCNN
import os
import math


def dataFetch(dset="MNIST"):
    """
        Params:
            dset (string): Either "MNIST" or "Fashion", returns test and training
            for respective dataset.
            Give "vMNIST" or "vFashion" to get the regular data set.
    """
    
    if (dset=="MNIST"):
        print("Starting to load MNIST dataset...")
        train_data = tv.datasets.MNIST("./data/",train=True, transform=tv.transforms.ToTensor(), download=True)
        test_data = tv.datasets.MNIST("./data/", train=False, transform=tv.transforms.ToTensor())
    elif (dset=="Fashion"):
        print("Starting to load MNIST Fashion dataset...")
        train_data = tv.datasets.FashionMNIST("./fashion/", train=True, transform=tv.transforms.ToTensor(), download=True)
        test_data = tv.datasets.FashionMNIST("./fashion/", train=False, transform=tv.transforms.ToTensor())
    elif (dset=="vMNIST"):
        print("Starting to load MNIST dataset...")
        train_data = tv.datasets.MNIST("./data/",train=True, transform=tv.transforms.ToTensor(), download=True)
        test_data = tv.datasets.MNIST("./data/", train=False, transform=tv.transforms.ToTensor())
        print("Loaded MNIST dataset successfully...")
        return train_data, test_data
    elif (dset=="vFashion"):
        print("Starting to load MNIST dataset...")
        train_data = tv.datasets.MNIST("./data/",train=True, transform=tv.transforms.ToTensor(), download=True)
        test_data = tv.datasets.MNIST("./data/", train=False, transform=tv.transforms.ToTensor())
        print("Loaded MNIST dataset successfully...")
        return train_data, test_data
        
        
        
    print("Loaded MNIST dataset successfully...")
    class_train = [0] * 10
    class_labels = [0] * 10
    finaltr = [0] * 10
    finalte = [0] * 10
    # divide by single numbers
    for i in range(10):
        sel = np.array(train_data.train_labels==i).nonzero()[0]
        class_train[i] = train_data.train_data[sel,:,:]
        class_labels[i] = train_data.train_labels[train_data.train_labels==i]
#        print(class_train[i].shape, max(class_labels[i]))
#            print("Length of class {}: {}".format(i, len(class_test[i])))
    
    # now double the size of the respective classes and fill up with equal sizes from other classes
    print("Starting to divide by class...")
    for i in range(10):
        print("Processing number {}".format(i))
        num = math.ceil(len(class_train[i])/9)
        temptr = class_train[i]
        tempte = class_labels[i]
#            print(max(class_labels[i]))
        for j in range(10):
            if (i!=j):
                sel = np.random.choice(len(class_train[j]), num, replace=False)
                tr = class_train[j][sel,:,:]
                te = class_labels[j][0:len(sel)]
                temptr = t.cat((temptr, tr))
                tempte = t.cat((tempte, te))

#                    print("Length of class {}, subpart {}: {}".format(i,j,te.shape))
        finaltr[i] = temptr
        finalte[i] = tempte
       
    return train_data, test_data, finaltr, finalte
    
        
class customDataset(t.utils.data.TensorDataset):
    def __init__(self, data_tensor, target_tensor, transform=tv.transforms.ToTensor):
        super(t.utils.data.TensorDataset, self)
        self.transform = transform
        
    def __getitem__(self, index):
        pass
        
        
def val_accuracy(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = model(images)
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy (10000 test images): {0:.2f}'.format(100 * correct / total))



def val_loss(test_loader, model, L):
    model.train()
    tloss = []
    for i, (img, lbl) in enumerate(test_loader):
        if args.gpu:
            images = Variable(img).cuda()
            labels = Variable(lbl).cuda()
        else:
            images = Variable(img)
            labels = Variable(lbl)

        #clear gradient
        optimizer.zero_grad() # Clear stored gradients
        outputs = model(images) # CNN forward pass
        loss = L(outputs, labels) # Calculate error
        tloss.append(loss.data[0])
    
    print ("Validation Loss: {0:.6f}".format(np.mean(tloss)))
    return np.mean(tloss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Grow/Shrink-Nets")
    parser.add_argument('-d', '--data', default="vMNIST",
                        help="Specify which dataset to use")
    parser.add_argument('-c', '--checkpoint', default="",
                        help="Load model from checkpoint")
    parser.add_argument('-v', '--validate', default=False,
                        help="Skips training and evaluates on test set only")
    parser.add_argument('-m', '--model', default="CNN", 
                        help="Model to be used. Either CNN, gCNN, sCNN, gsCNN")
    parser.add_argument('-g', '--gpu', default=False, type=bool,
                        help="Enable GPU support")
    parser.add_argument('-k', '--kernel', default=3, type=int,
                        help="Convolution kernel size for shrink layer")
    parser.add_argument('-ks', '--kernels', default=7, type=int,
                        help="Convolution kernel size for shrink layer")
    parser.add_argument('-kg', '--kernelg', default=3, type=int,
                        help="Convolution kernel size for growth layer")
    parser.add_argument('-r', '--rate', default=2, type=int,
                        help="Kernel growing/shrinking rate")
    parser.add_argument('-f', '--num_filters', default=16, type=int,
                        help="Base number of convolution filters")
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', default=20, type=int,
                        help="Number of epochs to train")
    parser.add_argument('-s', '--seed', default=1234, type=int,
                        help="Numpy random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    if args.gpu:
        t.cuda.manual_seed(args.seed)

    # DEFINE MODEL HERE
    if args.model == "CNN":
        model = CNN(kernel=args.kernel, num_filters=args.num_filters)
        fname = "models/CNN_"+str(args.kernel)+"_"+str(args.num_filters)+"_"+str(args.batch_size)+".model"
    elif args.model == "gCNN":
        model = gCNN(kernel=args.kernelg, num_filters=args.num_filters, rate= args.rate)
        fname = "models/gCNN_"+str(args.kernelg)+"_"+str(args.num_filters)+"_"+str(args.batch_size)+"_"+str(args.rate)+".model"
    elif args.model == "sCNN":
        model = sCNN(kernel=args.kernels, num_filters=args.num_filters, rate=args.rate)
        fname = "models/sCNN_"+str(args.kernels)+"_"+str(args.num_filters)+"_"+str(args.batch_size)+"_"+str(args.rate)+".model"
    elif args.model == "gsCNN":
        model = gsCNN(kernelg=args.kernelg, kernels=args.kernels, num_filters=args.num_filters, rate=args.rate)
        fname = "models/gsCNN_"+str(args.kernelg)+"_"+str(args.kernels)+"_"+str(args.num_filters)+"_"+str(args.batch_size)+"_"+str(args.rate)+".model"
        # not yet implemented
        
        
    if args.gpu:
        model.cuda()
    
    # Training setup
    L = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(),lr=args.learn_rate)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    
    # load to continue with pre-existing model
    if os.path.exists(fname):
        model.load_state_dict(t.load(fname))
        print("Successfully loaded previous model")
    
    
    # start with a model defined on 0
#    train_mix, test, train_data, train_labels = dataFetch()
#    # select only 0 category
#    train_dataset = customDataset(train_data[0], train_labels[0])
#    
#    # define train and test as DataLoaders
#    train_loader = t.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=args.batch_size, 
#                                           shuffle=True)
#    test_loader = test
    
    train_data, test_data = dataFetch(dset=args.data)
    
    train_loader = t.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size, 
                                           shuffle=True)
    test_loader = t.utils.data.DataLoader(dataset=test_data,
                                          batch_size=args.batch_size, 
                                          shuffle=False)

    print("Starting with training process...")
    if not args.validate:
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            tloss = []
            best = 1000
            for i, (img, lbl) in enumerate(train_loader):
                if args.gpu:
                    images = Variable(img).cuda()
                    labels = Variable(lbl).cuda()
                else:
                    images = Variable(img)
                    labels = Variable(lbl)
    
                
                # Pass and Backpropagate
                
                #clear gradient
                optimizer.zero_grad() # Clear stored gradients
                outputs = model(images) # CNN forward pass
                loss = L(outputs, labels) # Calculate error
                loss.backward() # Compute gradients
                tloss.append(loss.data[0])
                optimizer.step() # Update weights
        
            # output training loss
            print ("Epoch [{}/{}], Training Loss: {}"
                   .format(epoch+1, args.epochs, np.mean(tloss)))
            
            # compute validation loss
    
            val_accuracy(test_loader, model)
            vl = val_loss(test_loader, model, L)
            
            # elapsed time
            time_elapsed = time.time() - start
            print("Time for this epoch: {0:.2f} seconds".format(time_elapsed))
            
            # save the model if it has a better loss on the validation set
            if (vl<best):
                t.save(model.state_dict(),fname)
            
    print("Finished training.")
    # Change to evaluation
    model.eval()  
    # Compute accuracy
    print("Final accuracy:")
    val_accuracy(test_loader)
    val_loss(test_loader, model, L)
    
    
    
    
    
    