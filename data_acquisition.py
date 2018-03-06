#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:25:52 2018

@author: Dennis Aumiller
"""

import numpy as np
import torchvision as tv
import torch as t
# moved to main
if __name__ == "__main__":
    print("Starting to load MNIST dataset...")
    train_data = tv.datasets.MNIST("./data/",train=True, transform=tv.transforms.ToTensor(), download=True)
    test_data = tv.datasets.MNIST("./data/", train=False, transform=tv.transforms.ToTensor())
    print("Loaded MNIST dataset successfully...")
    print("Starting to load MNIST Fashion dataset...")
    train_fasion = tv.datasets.FashionMNIST("./fashion/", train=True, transform=tv.transforms.ToTensor(), download=True)
    test_fashion = tv.datasets.FashionMNIST("./fashion/", train=False, transform=tv.transforms.ToTensor())