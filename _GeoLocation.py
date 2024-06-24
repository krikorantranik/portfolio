import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geo

#format
locationdf = maindataset.groupby(["longitude_country","latitude_country"])["personName"].count()
locationdf = locationdf.reset_index(drop=False)

#display in map
fig, ax = plt.subplots(figsize=(20,10))
countries = geo.read_file(geo.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey", ax=ax)
locationdf.plot(x="longitude_country", y="latitude_country", kind="scatter",c="personName", colormap="YlOrRd",title=f"Location of billionaires",ax=ax)
plt.show()