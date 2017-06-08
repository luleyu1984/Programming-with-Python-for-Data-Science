#Human activity monitoring is a growing field within data science. It has practical use within the healthcare industry,
#particular with tracking the elderly to make sure they don't end up doing things which might cause them to hurt
# themselves. Governments are also very interested in it do that they can detect unusual crowd activities, perimeter
# breaches, or the identification of specific activities, such as loitering, littering, or fighting. Fitness apps also
# make use of activity monitoring to better estimate the amount of calories used by the body during a period of time.

#In this lab, you will be training a random forest against a public domain Human Activity Dataset titled Wearable
# Computing: Accelerometers' Data Classification of Body Postures and Movements, containing 165,633, one of which is
# invalid. Within the dataset, there are five target activities:

# Sitting
# Sitting Down
# Standing
# Standing Up
# Walking

# These activities were captured from four people wearing accelerometers mounted on their waist, left thigh,
# right arm, and right ankle.


import pandas as pd
import time

# Grab the DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


#
# TODO: Load up the dataset into dataframe 'X'
#
# .. your code here ..
X=pd.read_csv('Datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';', decimal=',')



#
# TODO: Encode the gender column, 0 as male, 1 as female
#
# .. your code here ..
X.gender=X.gender.map({'Man':0, 'Woman':1})

#
# TODO: Clean up any column with commas in it
# so that they're properly represented as decimals instead
#
# .. your code here ..


#
# INFO: Check data types
print (X.dtypes)



#
# TODO: Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic
#
# .. your code here ..
print (X.z4[122076])
X=X.drop(X.index[122076])
X.z4=pd.to_numeric(X.z4, errors='raise')
print (X.dtypes)


#
# INFO: If you find any problematic records, drop them before calling the
# to_numeric methods above...


#
# TODO: Encode your 'y' value as a dummies version of your dataset's "class" column
#
# .. your code here ..

y=pd.get_dummies(X['class'])

#
# TODO: Get rid of the user and class columns
#
# .. your code here ..
X=X.drop(labels=['user', 'class'], axis=1)

print (X.describe())


#
# INFO: An easy way to show which rows have nans in them
print (X[pd.isnull(X).any(axis=1)])



#
# TODO: Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0
#
# .. your code here ..

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=30, max_depth=10, oob_score=True, random_state=0)

# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)

print ("Fitting...")
s = time.time()
#
# TODO: train your model on your training set
#
# .. your code here ..
model.fit(X_train, y_train)


print ("Fitting completed in: ", time.time() - s)


#
# INFO: Display the OOB Score of your data
score = model.oob_score_
print ("OOB Score: ", round(score*100, 3))




print ("Scoring...")
s = time.time()
#
# TODO: score your model on your test set
#
# .. your code here ..

score=model.score(X_test, y_test)
print ("Score: ", round(score*100, 3))
print ("Scoring completed in: ", time.time() - s)


#
# TODO: Answer the lab questions, then come back to experiment more
# Fitting completed in:  11.814117908477783
# OOB Score:  98.744
# model's Score:  95.687

#
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0
# Try encoding it to pandas dummies
# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.



#
# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.

