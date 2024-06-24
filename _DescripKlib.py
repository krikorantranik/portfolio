import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import klib
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from scipy.stats import chi2_contingency


#description of numerical variables
klib.dist_plot(maindataset['finalWorth'])
klib.dist_plot(maindataset['age'])
klib.dist_plot(maindataset['cpi_country'])
klib.dist_plot(maindataset['cpi_change_country'])
klib.dist_plot(maindataset['gdp_country'])
klib.dist_plot(maindataset['gross_tertiary_education_enrollment'])
klib.dist_plot(maindataset['gross_primary_education_enrollment_country'])
klib.dist_plot(maindataset['life_expectancy_country'])
klib.dist_plot(maindataset['tax_revenue_country_country'])
klib.dist_plot(maindataset['population_country'])

#description of categorical variables
klib.cat_plot(maindataset[['category','country','selfMade','gender']], top=5, bottom=5)

#feature correlation
maindatasetC = maindataset[['finalWorth', 'category', 'age', 'country',
       'selfMade', 'gender', 'cpi_country', 'cpi_change_country',
       'gdp_country', 'gross_tertiary_education_enrollment',
       'gross_primary_education_enrollment_country', 'life_expectancy_country',
       'tax_revenue_country_country', 'total_tax_rate_country',
       'population_country', 'latitude_country', 'longitude_country']]

klib.corr_plot(maindatasetC, target='finalWorth')

#blox plot
plt.figure(figsize=(15,15))
ax = sns.boxplot(data=maindatasetC, x="finalWorth", y="category")
plt.figure(figsize=(15,15))
ax = sns.boxplot(data=maindatasetC, x="finalWorth", y="country")

#anova test
#formula of the anova (+: linear, *: non linear)
model = ols('num_salary ~ region + workclass + occupation + relationship + race + sex', data=maindatasetF).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

#chi2 test
catvar = ['workclass', 'occupation', 'relationship', 'race', 'sex', 'region', 'salary']
xtabs = []
pairs = []
for i in catvar:
 for j in catvar:
  if i!=j:
   pair = pd.DataFrame([[i,j]], columns=['i','j'])
   data_xtab = pd.crosstab(maindatasetF[i],maindatasetF[j],margins=False)
   xtabs.append(data_xtab)
   pairs.append(pair)
pairs = pd.concat(pairs, ignore_index=True, axis=0)
ps = []
for i in xtabs:
 stat, p, dof, expected = chi2_contingency(i)
 ps.append(p)
pairs['p values'] = ps
pairs

#correlation (numerical)
maindatasetF['num_salary'] = np.where(maindatasetF[' salary']==' <=50K',0,1)
klib.corr_plot(maindatasetF)
klib.corr_plot(maindatasetF, target='num_salary')