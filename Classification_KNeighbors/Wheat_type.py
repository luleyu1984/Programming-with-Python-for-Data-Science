# While training your machine with supervised learning, it's important to track how well its performing. In order to avoid
# overfitting, Your goal with machine learning is to create a generalizable algorithm that can be applied to data it hasn't seen yet
# and still do the task you've trained it to do. all of your training transformation and modeling needs to done using just your
# training data, without ever seeing your testing data. This will be your way of validating the true accuracy of your model.
# This is doable by splitting your training data into two portions. One part will actually be used for the training as
# usual, but the other part of the data is retained and used during testing only.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


def plotDecisionBoundary(model, X_train, y_train, X_test, y_test):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']

  # Calculate the boundaris from traing data
  x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
  y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plot the test original points as well...

  for label in range(len(np.unique(y_test))):
    indices = np.where(y_test == label)
    plt.scatter(X_test[indices, 0], X_test[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('Wheat type, Transformed Boundary, KNeighbors = ' + str(p['n_neighbors']))
  plt.text(0.7, 0.8, 'canadian', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
          color='black', fontsize=15)
  plt.text(0.1, 0.8, 'kama', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
          color='black', fontsize=15)
  plt.text(0.1, 0.1, 'rosa', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
           color='black', fontsize=15)
  plt.savefig('Wheat_type.png', bbox_inches='tight', dpi=300)
  plt.show()

# 
# TODO: Load up the dataset into a variable called X. Check the .head and
# compare it to the file you loaded in a text editor. Make sure you're
# loading your data properly--don't fail on the 1st step!
#
# .. your code here ..
X=pd.read_csv('Datasets/wheat.data')



#
# TODO: Copy the 'wheat_type' series slice out of X, and into a series
# called 'y'. Then drop the original 'wheat_type' column from the X
#
# .. your code here ..
y=X['wheat_type'].copy()
X=X.drop(labels=['id', 'wheat_type'], axis=1)



# TODO: Do a quick, "ordinal" conversion of 'y'. In actuality our
# classification isn't ordinal, but just as an experiment...
#
# .. your code here ..
print (y.unique())
y=y.astype('category').cat.codes
print (y.unique())
# kama:1, canadian 0, rosa 2

#
# TODO: Basic nan munging. Fill each row's nans with the mean of the feature
#
# .. your code here ..
print (X.isnull().any())
print (y.isnull().any())
X=X.fillna(X.mean())

print (X.isnull().any())

#
# TODO: Split X into training and testing data sets using train_test_split().
# INFO: Use 0.33 test size, and use random_state=1. This is important
# so that your answers are verifiable. In the real world, you wouldn't
# specify a random_state.
#
# .. your code here ..

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=1)


# TODO: Create an instance of SKLearn's Normalizer class and then train it
# using its .fit() method against your *training* data.
#
# NOTE: The reason you only fit against your training data is because in a
# real-world situation, you'll only have your training data to train with!
# In this lab setting, you have both train+test data; but in the wild,
# you'll only have your training data, and then unlabeled data you want to
# apply your models to.
#
# .. your code here ..

from sklearn import preprocessing
T=preprocessing.Normalizer().fit(X_train)


#
# TODO: With your trained pre-processor, transform both your training AND
# testing data.
#
# NOTE: Any testing data has to be transformed with your preprocessor
# that has ben fit against your training data, so that it exist in the same
# feature-space as the original data used to train your models.
#
# .. your code here ..
#from sklearn import preprocessing
#T=preprocessing.normalize(X)
#T=preprocessing.Normalizer().fit_transform(X)
T_train=T.transform(X_train)
T_test=T.transform(X_test)


#
# TODO: Just like your preprocessing transformation, create a PCA
# transformation as well. Fit it against your training data, and then
# project your training and testing features into PCA space using the
# PCA model's .transform() method.
#
# NOTE: This has to be done because the only way to visualize the decision
# boundary in 2D would be if your KNN algo ran in 2D as well:
#
# .. your code here ..

from sklearn.decomposition import PCA
pca = PCA(n_components = 2).fit(T_train)
pca_train=pca.transform(T_train)
pca_test=pca.transform(T_test)




#
# TODO: Create and train a KNeighborsClassifier. Start with K=9 neighbors.
# NOTE: Be sure train your classifier against the pre-processed, PCA-
# transformed training data above! You do not, of course, need to transform
# your labels.
#
# .. your code here ..

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test=train_test_split(pca_X, y, test_size=0.33, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 9).fit(pca_train, y_train)





# HINT: Ensure your KNeighbors classifier object from earlier is called 'knn'
plotDecisionBoundary(knn, pca_train, y_train, pca_test, y_test)


#------------------------------------
#
# TODO: Display the accuracy score of your test data/labels, computed by
# your KNeighbors model.
#
# NOTE: You do NOT have to run .predict before calling .score, since
# .score will take care of running your predictions for you automatically.
#
# .. your code here ..

print (knn.score(pca_test, y_test))

#
# BONUS: Instead of the ordinal conversion, try and get this assignment
# working with a proper Pandas get_dummies for feature encoding. HINT:
# You might have to update some of the plotDecisionBoundary code.



