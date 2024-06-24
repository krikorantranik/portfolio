import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
