#Parkinson's disease itself is a long-term disorder of the nervous system that affects many aspects of a person's
# mobility over time. It's characterized by shaking, slowed movement, rigidity, dementia, and depression. In 2013,
# some 53 million people were diagnosed with it, mostly men.
# In this lab, you will be applying SVC to the Parkinson's Data Set, provided courtesy of UCI's Machine Learning
# Repository. The dataset was created at the University of Oxford, in collaboration with 10 medical centers around
# the US, along with Intel who developed the device used to record the primary features of the dataset: speech signals.
#  Your goals for this assignment are first to see if it's possible to differentiate between people who have Parkinson's
#  and who don't using SciKit-Learn's support vector classifier, and then to take a first-stab at a naive way of
# fine-tuning your parameters in an attempt to maximize the accuracy of your testing set.

#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.preprocessing as pre
from sklearn.decomposition import PCA
from sklearn import manifold

Test_PCA=False

X=pd.read_csv('Datasets/parkinsons.data')
X.drop(labels='name', axis=1, inplace=True)
print (X.head())
y=X.status
X.drop(labels='status', axis=1, inplace=True)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)


#T=pre.Normalizer().fit(X_train)
#T=pre.MaxAbsScaler().fit(X_train)
#T=pre.MinMaxScaler().fit(X_train)
#T=pre.KernelCenterer().fit(X_train)
T=pre.StandardScaler().fit(X_train)

T_train=T.transform(X_train)
T_test=T.transform(X_test)

if Test_PCA:
  pca=PCA(n_components=5).fit(T_train)
  X_train=pca.transform(T_train)
  X_test=pca.transform(T_test)
else:
  iso=manifold.Isomap(n_components=5, n_neighbors=2).fit(T_train)
  X_train=iso.transform(T_train)
  X_test=iso.transform(T_test)


model=SVC()
model.fit(X_train, y_train)

score0=model.score(X_test, y_test)
print ('Score of default SVC is:', score0)


best_score=[]

for C in np.linspace(0.05, 2, num=40, endpoint=True):
    for gamma in np.linspace(0.001, 0.1, num=100):
        model=SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        score=model.score(X_test, y_test)
        best_score.append(score)
        #if score>best_score:
         #   best_score=score


print ('best_score is:', max(best_score))




# The highest accuracy score is: 0.949152542373