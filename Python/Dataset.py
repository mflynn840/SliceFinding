import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.preprocessing import StandardScaler


class Dataset:

    def __init__(self, X, Y, logger=None):


        self.X = X
        self.Y = Y
        self.lazy = False
        
    def __len__(self):
        return len(self.X)  

            
    def __getitem__(self, idxs):
        return self.X[idxs, :], self.Y[idxs]
         

    @staticmethod
    def loadAdult():
        
        X_train, Y_train, X_test, Y_test = None, None, None, None
        with open("Data/Adult/train.pkl", 'rb') as file:
            X_train, Y_train = pkl.load(file)
        with open("Data/Adult/test.pkl", 'rb') as file:
            X_test, Y_test = pkl.load(file)
        
        

        #form train and test sets
        trainSet = Dataset(X = X_train, Y=Y_train)
        testSet = Dataset(X = X_test, Y = Y_test)
        
        return trainSet, testSet


class DataSubset(Dataset):
    def __init__(self, X, Y, idxs, logger=None):
        super().__init__(X, Y)
        self.mapping = idxs
        
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idxs):
        return (self.X[self.self.mapping[idxs]], self.Y[self.mapping[idxs]])
    
'''

Iterator object for datasets

'''
class DataLoader:
    def __init__(self, dataset: Dataset, batchSize=10, shuffle=False, seed=1 ):
    
        self.data = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.batchSize = batchSize
        self.indexes = np.arange(len(self.data))
        self.currentIndex = 0
        self.batchNumber = -1
        self.size = len(self.data)
        np.random.seed(self.seed)
      
    #select a new permutation   
    def reset(self):
        
        if self.shuffle:
            # select a new permutation
            self.indexes = np.random.permutation(self.indexes)
        self.currentIndex = 0
        self.batchNumber = -1
        
    def __iter__(self):
        self.reset()
        return self
    
    #get the next batch of data
    def __next__(self):
        
        if self.currentIndex >= len(self.data):
            raise StopIteration
        
        #update bounds and batch number
        endIdx = self.currentIndex + self.batchSize
        oldStart = self.currentIndex
        self.currentIndex = endIdx
        self.batchNumber += 1
        
        #return subset

        return self.batchNumber, self.data[self.indexes[oldStart:endIdx]]
        
        
        
        



'''usage '''

'''
trainSet, testSet = Dataset.loadAdult()
trainloader = DataLoader(trainSet)
for index, (x_batch, y_batch) in trainloader:

    ...
'''


        