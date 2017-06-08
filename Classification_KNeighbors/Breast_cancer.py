# Breast cancer doesn't develop over night and, like any other cancer, can be treated extremely effectively if detected
# in its earlier stages. Part of the understanding cancer is knowing that not all irregular cell growths are malignant;
# some are benign, or non-dangerous, non-cancerous growths. A benign tumor does not mean the mass doesn't increase in
# size, but only means it does not pose a threat to nearby tissue, nor is it likely to spread to other parts of the body.
# The mass simply stays wherever it's growing. Benign tumors are actually pretty popular, such as moles and some warts.
# Being able to properly assess if a tumor is actually benign and ignorable, or malignant and alarming is therefore of
# importance, and also is a problem that might be solvable through data and machine learning.
# In this lab, you'll be using the Breast Cancer Wisconsin Original  data set, provided courtesy of
# UCI's Machine Learning Repository, to classify tumor growths as benign or malignant, based off of a handful of features.

# If you'd like to try this lab with PCA instead of Isomap, as the dimensionality reduction technique:

import pandas as pd
Test_PCA =  False

def plotDecisionBoundary(model, X_train, y_train, X_test, y_test):
  print ("Plotting...")
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
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
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y_test):
    indices = np.where(y_test== label)
    plt.scatter(X_test[indices, 0], X_test[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('Breast Cancer, Transformed Boundary, KNeighbors = ' + str(p['n_neighbors']))
  plt.text(0.75, 0.1, 'benign', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
           color='black', fontsize=15)
  plt.text(0.1, 0.1, 'malignant', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
           color='black', fontsize=15)
  plt.savefig('Breast_cancer.png', bbox_inches='tight', dpi=300)
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
# .. your code here ..

df=pd.read_csv('Datasets/breast-cancer-wisconsin.data', header=None,
               names=list(['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei',
                           'chromatin', 'nucleoli', 'mitoses', 'status']))




# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. Always verify you properly executed the drop by double checking
# (printing out the resulting operating)! Many people forget to set the right
# axis here.
#
# If you goofed up on loading the dataset and notice you have a `sample` column,
# this would be a good place to drop that too if you haven't already.
#
# .. your code here ..
status=df.status.copy()
df=df.drop(labels=['status', 'sample'], axis=1)



#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
# .. your code here ..
print (df.isnull().any())
print (df.dtypes)
print (df.nuclei.unique())
df.nuclei=pd.to_numeric(df.nuclei, errors='coerce')
print (df.dtypes)
df.nuclei=df.nuclei.fillna(df.nuclei.mean())

print (df.nuclei.unique())
#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
# .. your code here ..
from sklearn.model_selection import train_test_split
df_train, df_test, status_train, status_test=train_test_split(df, status, test_size=0.5, random_state=7)



#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
# .. your code here ..

print (df.describe())
from sklearn import preprocessing
#T=df_train
#T=preprocessing.Normalizer().fit(df_train)
T=preprocessing.MinMaxScaler().fit(df_train) # best scaler in this case
#T=preprocessing.RobustScaler().fit(df_train)
#T=preprocessing.StandardScaler().fit(df_train)
T_train=T.transform(df_train)
T_test=T.transform(df_test)


#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print ("Computing 2D Principle Components")
  #
  # TODO: Implement PCA here. Save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  from sklearn.decomposition import PCA
  model=PCA(n_components=2).fit(T_train)
  

else:
  print ("Computing 2D Isomap Manifold")
  #
  # TODO: Implement Isomap here. Save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  from sklearn import manifold
  model=manifold.Isomap(n_neighbors=5, n_components=2).fit(T_train)



#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
# .. your code here ..

data_train=model.transform(T_train)
data_test=model.transform(T_test)


#There are two types of errors this classification can make, and they are NOT equal. The first is a false positive.
# This would be the algorithm errantly classifying a benigh tumor as malignant, which would then prompt doctors to
# investigate it further, perhaps even schedule a surgery to have it removed. It would be wasteful monetairly and
# in terms of resources, but not much more than that.

# The other type of error would be a false negative. This would be the algorithm incorrectly classifying a dangerious,
#  malignant tumor as benign. If that were to occur, the tumor would be given time to progress into later,
# more serious stages, and could potentially spread to other parts of the body. A much more dangerious situation
# to be in.

#
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# .. your code here ..



from sklearn.neighbors import KNeighborsClassifier
knmodel=KNeighborsClassifier(n_neighbors=10).fit(data_train, status_train)







#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.



#
# TODO: Calculate + Print the accuracy of the testing set
#
# .. your code here ..
print (knmodel.score(data_test, status_test))

plotDecisionBoundary(knmodel, data_train, status_train, data_test, status_test)
