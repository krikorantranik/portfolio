
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#scale
def properscaler(simio):
 scaler = StandardScaler()
 resultsWordstrans = scaler.fit_transform(simio)
 resultsWordstrans = pd.DataFrame(resultsWordstrans)
 resultsWordstrans.index = simio.index
 resultsWordstrans.columns = simio.columns
 return resultsWordstrans

datasetR = properscaler(textvectors)

#PCA
def varred(simio):
 scaler = PCA(n_components=0.8, svd_solver='full')
 resultsWordstrans = simio.copy()
 resultsWordstrans = scaler.fit_transform(resultsWordstrans)
 resultsWordstrans = pd.DataFrame(resultsWordstrans)
 resultsWordstrans.index = simio.index
 resultsWordstrans.columns = resultsWordstrans.columns.astype(str)
 return resultsWordstrans

datasetR = varred(datasetR)