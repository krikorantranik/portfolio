import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("R2: ", r2_score(output['actual'], output['predicted']))
print("MeanSqError: ",np.sqrt(mean_squared_error(output['actual'], output['predicted'])))
print("MeanAbsError: ", mean_absolute_error(output['actual'],output['predicted']))

#nice confusion matrix
output["RangePredicted"] = np.where(output['predicted']<=2.5,"1.Bad","2.Other")
output["RangeActual"] = np.where(output['actual']<=2.5,"1.Bad","2.Other")

ConfusionMatrixDisplay.from_predictions(y_true=output['RangeActual'] ,y_pred=output['RangePredicted'] , cmap='PuBu')

