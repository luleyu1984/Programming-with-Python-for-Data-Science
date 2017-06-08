#In this lab, you're going to use decision trees to peruse The Mushroom Data Set, drawn from the Audobon Society Field
# Guide to North American Mushrooms (1981). The data set details mushrooms described in terms of many physical
# characteristics, such as cap size and stalk length, along with a classification of poisonous or edible.
# As a standard disclaimer, if you eat a random mushroom you find, you are doing so at your own risk. While every
# effort has been made to ensure that the information contained with the data set is correct, please understand that
# no one associated with this course accepts any responsibility or liability for errors, omissions or representations,
#  expressed or implied, contained therein, or that might arise from you mistakenly identifying a mushroom.
# Exercise due caution and just take this lab as informational purposes only.

import pandas as pd
import numpy as np


#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
# .. your code here ..

# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]


colnames=['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
          'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
          'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
          'spore-print-color', 'population', 'habitat']
X=pd.read_csv('Datasets/agaricus-lepiota.data', header=None, names=colnames)
X.replace(to_replace='?', value=np.nan, inplace=True)
print (X['stalk-root'].unique())
print (X[X.isnull().any(axis=1)])

# 
# TODO: Go ahead and drop any row with a nan
#
# .. your code here ..
print (X.shape)
X=X.dropna(axis=0, how='any').reset_index(drop=True)
print (X.tail())
#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
# .. your code here ..
y=X.label.map({'p':1, 'e':0})
X.drop(labels='label', axis=1, inplace=True)
print (X.shape)


#
# TODO: Encode the entire dataset using dummies
#
# .. your code here ..
X=pd.get_dummies(X)
print (X.shape)

# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)

#
# TODO: Create an DT classifier. No need to set any parameters
#
# .. your code here ..

from sklearn import tree
dtc=tree.DecisionTreeClassifier()
#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
# .. your code here ..

dtc.fit(X_train, y_train)
score=dtc.score(X_test, y_test)

print ("High-Dimensionality Score: ", round((score*100), 3))


#
# TODO: Use the code on the course's SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz`. If you can't, use: http://webgraphviz.com/.
# On Windows 10, graphviz installs via a msi installer that you can download from
# the graphviz website. Also, a graph editor, gvedit.exe can be used to view the
# tree directly from the exported tree.dot file without having to issue a call.
#
# .. your code here ..
tree.export_graphviz(dtc.tree_, out_file='Mushroom_Poisonous_1_eadible_0.dot', feature_names=X.columns, class_names=['edible','poisonous'])

from subprocess import call
call(['dot', '-T', 'png', 'Mushroom_Poisonous_1_eadible_0.dot', '-o', 'Mushroom_Poisonous_1_eadible_0.png'])


