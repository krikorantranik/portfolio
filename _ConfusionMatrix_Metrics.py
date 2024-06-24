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

#sample of heatmaps
artistheatAggCl = pd.DataFrame(filtered.groupby(['artist_name','clAggCl'])['artist_name'].count()).reset_index(names=["artist","cluster"])
artistheatAggCl = artistheatAggCl.pivot(index="artist", columns="cluster", values="artist_name")
artistheatKMeans = pd.DataFrame(filtered.groupby(['artist_name','clKMeans'])['artist_name'].count()).reset_index(names=["artist","cluster"])
artistheatKMeans = artistheatKMeans.pivot(index="artist", columns="cluster", values="artist_name")
artistheatBirch = pd.DataFrame(filtered.groupby(['artist_name','clBirch'])['artist_name'].count()).reset_index(names=["artist","cluster"])
artistheatBirch = artistheatBirch.pivot(index="artist", columns="cluster", values="artist_name")
artistheatOptics = pd.DataFrame(filtered.groupby(['artist_name','clOptics'])['artist_name'].count()).reset_index(names=["artist","cluster"])
artistheatOptics = artistheatOptics.pivot(index="artist", columns="cluster", values="artist_name")
artistheatDbscan = pd.DataFrame(filtered.groupby(['artist_name','clDbscan'])['artist_name'].count()).reset_index(names=["artist","cluster"])
artistheatDbscan = artistheatDbscan.pivot(index="artist", columns="cluster", values="artist_name")
fig, axes = plt.subplots(3,2, figsize=(20,20))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
sns.heatmap(artistheatAggCl, cmap="YlOrBr", ax=ax1)
sns.heatmap(artistheatKMeans, cmap="YlOrBr", ax=ax2)
sns.heatmap(artistheatBirch, cmap="YlOrBr", ax=ax3)
sns.heatmap(artistheatOptics, cmap="YlOrBr", ax=ax4)
sns.heatmap(artistheatDbscan, cmap="YlOrBr", ax=ax5)
ax1.set_title("Agg. Clustering")
ax2.set_title("KMeans")
ax3.set_title("Birch")
ax4.set_title("Optics")
ax5.set_title("DBScan")

