import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

tr = tree.DecisionTreeClassifier(max_depth=4)
tr = tr.fit(X, Y_train)
#I am limiting the depth to three
fig, ax1 = plt.subplots(figsize=(25,15))
tree.plot_tree(tr, ax=ax1, feature_names=featnames, proportion=True, filled=True, fontsize=7)
plt.show()