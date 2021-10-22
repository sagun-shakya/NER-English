# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:21:34 2021

@author: Sagun Shakya
"""

from torch.utils.data import Dataset

class NERDataset(Dataset):
    """
    This is a custom dataset class. It gets the X and Y data to be fed into the Dataloader.
    """
    def __init__(self, X, Y, pad_id):
        self.X = X
        self.Y = Y
        self.pad_id = pad_id
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _y = self.Y[index]
        
        _x = self.X[index]
        _l = len(tuple(filter(lambda x: x != self.pad_id, self.X[index])))  # Calculates the length of the original unpadded sentence.

        return (_x,_l), _y

