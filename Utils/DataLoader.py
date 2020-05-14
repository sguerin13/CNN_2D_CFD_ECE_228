from __future__ import print_function, division
import os
import torch


# import pandas as pd

#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

class DataSet(Dataset,data):

    def __init__(self,list_IDs, labels):
        'initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'returns the length'
        return len(self.list_IDs)

    def __getitem__(self,index):
        pass
