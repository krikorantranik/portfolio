import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import klib

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

