# Many U.S. cities, the U.S. federal government, and even other cities and governments abroad have started
# subscribing to an Open Data policy, because some data should be transparent and available to everyone to use
# and republish freely, without restrictions from copyright, patents, or other mechanisms of control.
# After reading their terms of use, in this lab you'll be exploring the City of Chicago's Crime data set,
# which is part of their Open Data initiative.

# To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'. The city's website itself has hundreds of other datasets you can browse and do machine learning on.
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'

# Fill out the doKMeans method to find and plot seven clusters and print out their centroids.
# These could be places a police officer investigates to check for on-going illegal activities.



# .. your code here ..
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Look Pretty
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')



def doKMeans(df, title):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_xlim(-87.95, -87.50)
  ax.set_ylim(41.60, 42.10)
  ax.set_title(title)

  #
  # TODO: Filter df so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #
  # .. your code here ..
  df=df.loc[:, ['Longitude', 'Latitude']]
  df=df.dropna(axis=0, how='any')


  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  # Be sure to name your kmeans model `model` so that the printing works.
  #
  # .. your code here ..
  model=KMeans(n_clusters=7).fit(df)


  #
  # INFO: Print and plot the centroids...
  centroids = model.cluster_centers_
  ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='green', alpha=0.5, linewidths=3, s=169)
  ax.legend(['Location', 'Centroid'])
  print (centroids)
  plt.savefig(title + '.png', bbox_inches='tight', dpi=300)



#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..
df=pd.read_csv('Datasets/Crimes2001_to_present.csv')
#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..

df.dropna(axis=0, how='any')
#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..
print (df.dtypes)
#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..

df.Date=pd.to_datetime(df.Date, errors='coerce')
# INFO: Print & Plot your data




doKMeans(df, 'Gambling_locations_in_Chicago')


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..


df=df[df.Date> '2011-01-01']
# INFO: Print & Plot your data
doKMeans(df, 'Gambling_locations_in_Chicago (Year 2011 to present)')
plt.show()


# You'll notice that the cluster assignments are pretty accurate. After limited the date range to >2011,
# you will find that all clusters have moved but only slightly, and the centroid arrangement still has the same
# shape for the most part.
# The output figures are attached ('Gambling locations in Chicago.png', 'Gambling locations in Chicago (Year 2011 to present).png')