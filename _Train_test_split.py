import pandas as pd
from sklearn.model_selection import train_test_split

#split in sklearn
outp = train_test_split(ds, train_size=0.7)
finaleval=outp[1]
subset=outp[0]