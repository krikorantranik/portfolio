import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import torch
from torch.utils.data import DataLoader

#split in sklearn
outp = train_test_split(ds, train_size=0.7)
finaleval=outp[1]
subset=outp[0]

#rebalance samples
def rebalance(sset, min, max):
 classes = list(set(sset["target"]))
 a = []
 for clas in classes:
  positives = sset[sset['target']==clas]
  if len(positives) < min:
   positives = resample(positives, n_samples=min, replace=True)
  if len(positives) > max:
   positives = resample(positives, n_samples=max, replace=False)
  a.append(positives)
 rebalanced = pd.concat(a, axis=0, ignore_index=True)
 return rebalanced

#Pytorch way
train, test = torch.utils.data.random_split(fullset, [int(np.floor(len(fullset)*0.8)), int(np.ceil(len(fullset)*0.2))])
train, val = torch.utils.data.random_split(train, [int(np.floor(len(train)*0.75)), int(np.ceil(len(train)*0.25))])

#dataloaders divided per batch
batch_size = 200
dataloaders = {'train': DataLoader(train, batch_size=batch_size),
               'val': DataLoader(val, batch_size=batch_size),
               'test': DataLoader(test, batch_size=batch_size)}

dataset_sizes = {'train': len(train),
                 'val': len(val),
                 'test': len(test)}
print(f'dataset_sizes = {dataset_sizes}')

