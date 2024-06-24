import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
import ast

#metrics for K
a = []
X = datasetR.to_numpy(dtype='float')
for ncl in np.arange(2, int(20), 1):
 clusterer = AgglomerativeClustering(n_clusters=int(ncl))
 cluster_labels1 = clusterer.fit_predict(X)
 silhouette_avg1 = silhouette_score(X, cluster_labels1)
 calinski1 = calinski_harabasz_score(X, cluster_labels1)
 clusterer = KMeans(n_clusters=int(ncl))
 with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  cluster_labels2 = clusterer.fit_predict(X)
 silhouette_avg2 = silhouette_score(X, cluster_labels2)
 calinski2 = calinski_harabasz_score(X, cluster_labels2)
 clusterer = Birch(n_clusters=int(ncl))
 cluster_labels3 = clusterer.fit_predict(X)
 silhouette_avg3 = silhouette_score(X, cluster_labels3)
 calinski3 = calinski_harabasz_score(X, cluster_labels3)
 row = pd.DataFrame({"ncl": [ncl],
                     "silAggCl": [silhouette_avg1], "c_hAggCl": [calinski1],
                     "silKMeans": [silhouette_avg2], "c_hKMeans": [calinski2],
                     "silBirch": [silhouette_avg3], "c_hBirch": [calinski3],
                     })
 a.append(row)
scores = pd.concat(a, ignore_index=True)
plt.style.use('bmh')
fig, [ax_sil, ax_ch] = plt.subplots(1,2,figsize=(15,7))
ax_sil.plot(scores["ncl"], scores["silAggCl"], 'g-')
ax_sil.plot(scores["ncl"], scores["silKMeans"], 'b-')
ax_sil.plot(scores["ncl"], scores["silBirch"], 'r-')
ax_ch.plot(scores["ncl"], scores["c_hAggCl"], 'g-', label='Agg Clust')
ax_ch.plot(scores["ncl"], scores["c_hKMeans"], 'b-', label='KMeans')
ax_ch.plot(scores["ncl"], scores["c_hBirch"], 'r-', label='Birch')
ax_sil.set_title("Silhouette curves")
ax_ch.set_title("Calinski Harabasz curves")
ax_sil.set_xlabel('clusters')
ax_sil.set_ylabel('silhouette_avg')
ax_ch.set_xlabel('clusters')
ax_ch.set_ylabel('calinski_harabasz')
ax_ch.legend(loc="upper right")
plt.show()

#set parameters, run each algorithm
ncl_AggCl = 11
ncl_KMeans = 17
ncl_Birch = 9

X = datasetR.to_numpy(dtype='float')
clusterer1 = AgglomerativeClustering(n_clusters=int(ncl_AggCl))
cluster_labels1 = clusterer1.fit_predict(X)
n_clusters1 = max(cluster_labels1)
silhouette_avg1 = silhouette_score(X, cluster_labels1)
sample_silhouette_values1 = silhouette_samples(X, cluster_labels1)

with warnings.catch_warnings():
 warnings.simplefilter("ignore")
 clusterer2 = KMeans(n_clusters=int(ncl_KMeans))
 cluster_labels2 = clusterer2.fit_predict(X)
n_clusters2 = max(cluster_labels2)
silhouette_avg2 = silhouette_score(X, cluster_labels2)
sample_silhouette_values2 = silhouette_samples(X, cluster_labels2)

clusterer3 = Birch(n_clusters=int(ncl_Birch))
cluster_labels3 = clusterer3.fit_predict(X)
n_clusters3 = max(cluster_labels3)
silhouette_avg3 = silhouette_score(X, cluster_labels3)
sample_silhouette_values3 = silhouette_samples(X, cluster_labels3)

clusterer4 = OPTICS(min_samples=2)
cluster_labels4 = clusterer4.fit_predict(X)
n_clusters4 = max(cluster_labels4)
silhouette_avg4 = silhouette_score(X, cluster_labels4)
sample_silhouette_values4 = silhouette_samples(X, cluster_labels4)

clusterer5 = DBSCAN(eps=1, min_samples=2)
cluster_labels5 = clusterer5.fit_predict(X)
n_clusters5 = max(cluster_labels5)
silhouette_avg5 = silhouette_score(X, cluster_labels5)
sample_silhouette_values5 = silhouette_samples(X, cluster_labels5)

finalDF = datasetR.copy()
finalDF["clAggCl"] = cluster_labels1
finalDF["clKMeans"] = cluster_labels2
finalDF["clBirch"] = cluster_labels3
finalDF["clOptics"] = cluster_labels4
finalDF["clDbscan"] = cluster_labels5
finalDF["silAggCl"] = sample_silhouette_values1
finalDF["silKMeans"] = sample_silhouette_values2
finalDF["silBirch"] = sample_silhouette_values3
finalDF["silOptics"] = sample_silhouette_values4
finalDF["silDbscan"] = sample_silhouette_values5

#silhouetes
fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(1,5,figsize=(20,20))
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X) + (n_clusters1 + 1) * 10])
y_lower = 10
for i in range(min(cluster_labels1),max(cluster_labels1)+1):
 ith_cluster_silhouette_values = sample_silhouette_values1[cluster_labels1 == i]
 ith_cluster_silhouette_values.sort()
 size_cluster_i = ith_cluster_silhouette_values.shape[0]
 y_upper = y_lower + size_cluster_i
 color = cm.nipy_spectral(float(i) / n_clusters1)
 ax1.fill_betweenx(
  np.arange(y_lower, y_upper),
  0,
  ith_cluster_silhouette_values,
  facecolor=color,
  edgecolor=color,
  alpha=0.7,
 )
 ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 y_lower = y_upper + 10  # 10 for the 0 samples
ax1.set_title("Silhouette plot for Agg. Clustering")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster labels")
ax1.axvline(x=silhouette_avg1, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

ax2.set_xlim([-0.1, 1])
ax2.set_ylim([0, len(X) + (n_clusters2 + 1) * 10])
y_lower = 10
for i in range(min(cluster_labels2),max(cluster_labels2)+1):
 ith_cluster_silhouette_values = sample_silhouette_values2[cluster_labels2 == i]
 ith_cluster_silhouette_values.sort()
 size_cluster_i = ith_cluster_silhouette_values.shape[0]
 y_upper = y_lower + size_cluster_i
 color = cm.nipy_spectral(float(i) / n_clusters2)
 ax2.fill_betweenx(
  np.arange(y_lower, y_upper),
  0,
  ith_cluster_silhouette_values,
  facecolor=color,
  edgecolor=color,
  alpha=0.7,
 )
 ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 y_lower = y_upper + 10  # 10 for the 0 samples
ax2.set_title("Silhouette plot for KMeans")
ax2.set_xlabel("Silhouette coefficient values")
ax2.set_ylabel("Cluster labels")
ax2.axvline(x=silhouette_avg2, color="red", linestyle="--")
ax2.set_yticks([])  # Clear the yaxis labels / ticks
ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

ax3.set_xlim([-0.1, 1])
ax3.set_ylim([0, len(X) + (n_clusters3 + 1) * 10])
y_lower = 10
for i in range(min(cluster_labels3),max(cluster_labels3)+1):
 ith_cluster_silhouette_values = sample_silhouette_values3[cluster_labels3 == i]
 ith_cluster_silhouette_values.sort()
 size_cluster_i = ith_cluster_silhouette_values.shape[0]
 y_upper = y_lower + size_cluster_i
 color = cm.nipy_spectral(float(i) / n_clusters3)
 ax3.fill_betweenx(
  np.arange(y_lower, y_upper),
  0,
  ith_cluster_silhouette_values,
  facecolor=color,
  edgecolor=color,
  alpha=0.7,
 )
 ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 y_lower = y_upper + 10  # 10 for the 0 samples
ax3.set_title("Silhouette plot for Birch")
ax3.set_xlabel("Silhouette coefficient values")
ax3.set_ylabel("Cluster labels")
ax3.axvline(x=silhouette_avg3, color="red", linestyle="--")
ax3.set_yticks([])  # Clear the yaxis labels / ticks
ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

ax4.set_xlim([-0.1, 1])
ax4.set_ylim([0, len(X) + (n_clusters4 + 1) * 10])
y_lower = 10
for i in range(min(cluster_labels4),max(cluster_labels4)+1):
 ith_cluster_silhouette_values = sample_silhouette_values4[cluster_labels4 == i]
 ith_cluster_silhouette_values.sort()
 size_cluster_i = ith_cluster_silhouette_values.shape[0]
 y_upper = y_lower + size_cluster_i
 color = cm.nipy_spectral(float(i) / n_clusters4)
 ax4.fill_betweenx(
  np.arange(y_lower, y_upper),
  0,
  ith_cluster_silhouette_values,
  facecolor=color,
  edgecolor=color,
  alpha=0.7,
 )
 ax4.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 y_lower = y_upper + 10  # 10 for the 0 samples
ax4.set_title("Silhouette plot for Optics")
ax4.set_xlabel("Silhouette coefficient values")
ax4.set_ylabel("Cluster labels")
ax4.axvline(x=silhouette_avg4, color="red", linestyle="--")
ax4.set_yticks([])  # Clear the yaxis labels / ticks
ax4.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

ax5.set_xlim([-0.1, 1])
ax5.set_ylim([0, len(X) + (n_clusters5 + 1) * 10])
y_lower = 10
for i in range(min(cluster_labels5),max(cluster_labels5)+1):
 ith_cluster_silhouette_values = sample_silhouette_values5[cluster_labels5 == i]
 ith_cluster_silhouette_values.sort()
 size_cluster_i = ith_cluster_silhouette_values.shape[0]
 y_upper = y_lower + size_cluster_i
 color = cm.nipy_spectral(float(i) / n_clusters5)
 ax5.fill_betweenx(
  np.arange(y_lower, y_upper),
  0,
  ith_cluster_silhouette_values,
  facecolor=color,
  edgecolor=color,
  alpha=0.7,
 )
 ax5.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 y_lower = y_upper + 10  # 10 for the 0 samples
ax5.set_title("Silhouette plot for DBScan")
ax5.set_xlabel("Silhouette coefficient values")
ax5.set_ylabel("Cluster labels")
ax5.axvline(x=silhouette_avg5, color="red", linestyle="--")
ax5.set_yticks([])  # Clear the yaxis labels / ticks
ax5.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


