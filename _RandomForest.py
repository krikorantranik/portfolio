import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


num_columns = ['age', 'cpi_country', 'cpi_change_country', 'gdp_country', 'gross_tertiary_education_enrollment', 'gross_primary_education_enrollment_country',
 'life_expectancy_country', 'tax_revenue_country_country', 'total_tax_rate_country', 'population_country', 'latitude_country', 'longitude_country']
cat_columns = ['category', 'country', 'selfMade', 'gender']

#define object
cat_preprocessor = OneHotEncoder(handle_unknown="ignore")
num_preprocessor = StandardScaler()
preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", cat_preprocessor, cat_columns),
        ("standard_scaler", num_preprocessor, num_columns),
    ])

#dataset
train = maindatasetC.copy()
train = train.dropna()
X_train = train[['category', 'age', 'country',
       'selfMade', 'gender', 'cpi_country', 'cpi_change_country',
       'gdp_country', 'gross_tertiary_education_enrollment',
       'gross_primary_education_enrollment_country', 'life_expectancy_country',
       'tax_revenue_country_country', 'total_tax_rate_country',
       'population_country', 'latitude_country', 'longitude_country']]
Y_train = train[['finalWorth']]

#prepare data and labels
X = pd.DataFrame.sparse.from_spmatrix(preprocessor.fit_transform(X_train))
catnames = preprocessor.transformers_[0][1].get_feature_names_out(cat_columns).tolist()
numnames = preprocessor.transformers_[1][1].get_feature_names_out(num_columns).tolist()
featnames = catnames + numnames

#regressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, Y_train)


#show variable importance
imp = rf.feature_importances_
imp = pd.Series(imp, index=featnames)
std = pd.Series(np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0), index=featnames)
fig, ax = plt.subplots()
imp.plot(kind='barh', yerr=std, ax=ax, figsize=(15,15))
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()