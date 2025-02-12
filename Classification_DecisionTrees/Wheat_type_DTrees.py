#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time

#
# INFO: Your Parameters.
# You can adjust them after completing the lab
C = 1
kernel = 'linear'
iterations = 5000  # TODO: Change to 200000 once you get to Question#2

#
# INFO: You can set this to false if you want to
# draw the full square matrix
FAST_DRAW = True


def drawPlots(model, X_train, X_test, y_train, y_test, wintitle='Figure 1'):
    # INFO: A convenience function for you
    # You can use this to break any higher-dimensional space down
    # And view cross sections of it.

    # If this line throws an error, use plt.style.use('ggplot') instead
    mpl.style.use('ggplot')  # Look Pretty

    padding = 3
    resolution = 0.5
    max_2d_score = 0

    y_colors = ['#ff0000', '#00ff00', '#0000ff']
    my_cmap = mpl.colors.ListedColormap(['#ffaaaa', '#aaffaa', '#aaaaff'])
    colors = [y_colors[i] for i in y_train]
    num_columns = len(X_train.columns)

    fig = plt.figure()
    fig.canvas.set_window_title(wintitle)

    cnt = 0
    for col in range(num_columns):
        for row in range(num_columns):
            # Easy out
            if FAST_DRAW and col > row:
                cnt += 1
                continue

            ax = plt.subplot(num_columns, num_columns, cnt + 1)
            plt.xticks(())
            plt.yticks(())

            # Intersection:
            if col == row:
                plt.text(0.5, 0.5, X_train.columns[row], verticalalignment='center', horizontalalignment='center',
                         fontsize=12)
                cnt += 1
                continue

            # Only select two features to display, then train the model
            X_train_bag = X_train.ix[:, [row, col]]
            X_test_bag = X_test.ix[:, [row, col]]
            model.fit(X_train_bag, y_train)

            # Create a mesh to plot in
            x_min, x_max = X_train_bag.ix[:, 0].min() - padding, X_train_bag.ix[:, 0].max() + padding
            y_min, y_max = X_train_bag.ix[:, 1].min() - padding, X_train_bag.ix[:, 1].max() + padding
            xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                                 np.arange(y_min, y_max, resolution))

            # Plot Boundaries
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            # Prepare the contour
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=my_cmap, alpha=0.8)
            plt.scatter(X_train_bag.ix[:, 0], X_train_bag.ix[:, 1], c=colors, alpha=0.5)

            score = round(model.score(X_test_bag, y_test) * 100, 3)
            plt.text(0.5, 0, "Score: {0}".format(score), transform=ax.transAxes, horizontalalignment='center',
                     fontsize=8)
            max_2d_score = score if score > max_2d_score else max_2d_score

            cnt += 1

    print("Max 2D Score: ", max_2d_score)
    fig.set_tight_layout(True)


def benchmark(model, X_train, X_test, y_train, y_test, wintitle='Figure 1'):
    print('\n\n' + wintitle + ' Results')
    s = time.time()
    for i in range(iterations):
        #
        # TODO: train the classifier on the training data / labels:
        #
        # .. your code here ..
        model.fit(X_train, y_train)
    print("{0} Iterations Training Time: ".format(iterations), time.time() - s)

    s = time.time()
    for i in range(iterations):
        #
        # TODO: score the classifier on the testing data / labels:
        #
        # .. your code here ..
        score = model.score(X_test, y_test)
    print("{0} Iterations Scoring Time: ".format(iterations), time.time() - s)
    print("High-Dimensionality Score: ", round((score * 100), 3))


#
# TODO: Load up the wheat dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled, nor weird headers...
#
# .. your code here ..

X = pd.read_csv('Datasets/wheat.data')
X = X.drop(labels='id', axis=1)
print(X.head())

# INFO: An easy way to show which rows have nans in them
# print X[pd.isnull(X).any(axis=1)]

print(X.isnull().any(axis=1))
#
# TODO: Go ahead and drop any row with a nan
#
# .. your code here ..

X = X.dropna(axis=0).reset_index(drop=True)

#
# INFO: # In the future, you might try setting the nan values to the
# mean value of that column, the mean should only be calculated for
# the specific class rather than across all classes, now that you
# have the labels



#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
# .. your code here ..
y = X['wheat_type'].map({'canadian': 0, 'kama': 1, 'rosa': 2})
X = X.drop(labels='wheat_type', axis=1)
print(y)

#
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7.
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

#
# TODO: Create an SVC classifier named svc
# Use a linear kernel, and set the C value to C
#
# .. your code here ..
from sklearn.svm import SVC

svc = SVC(kernel=kernel, C=C)

#
# TODO: Create an KNeighbors classifier named knn
# Set the neighbor count to 5
#
# .. your code here ..

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)


from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(max_depth=9, random_state=2)

benchmark(knn, X_train, X_test, y_train, y_test, 'KNeighbors')
drawPlots(knn, X_train, X_test, y_train, y_test, 'KNeighbors')

benchmark(svc, X_train, X_test, y_train, y_test, 'SVC')
drawPlots(svc, X_train, X_test, y_train, y_test, 'SVC')

benchmark(dtc, X_train, X_test, y_train, y_test, 'DecisionTree')
drawPlots(dtc, X_train, X_test, y_train, y_test, 'DecisionTree')

plt.savefig('Wheat_type_DTrees.png', dpi=300)

plt.show()


# KNeighbors Results
# 5000 Iterations Training Time:  2.2077319622039795
# 5000 Iterations Scoring Time:  4.935338973999023
# High-Dimensionality Score:  83.607
# Max 2D Score:  90.164



# SVC Results
# 5000 Iterations Training Time:  4.282010078430176
# 5000 Iterations Scoring Time:  1.9834461212158203
# High-Dimensionality Score:  86.885
# Max 2D Score:  93.443


# DecisionTree Results
# 5000 Iterations Training Time:  3.3365869522094727
# 5000 Iterations Scoring Time:  1.6474010944366455
# High-Dimensionality Score:  91.803
# Max 2D Score:  90.164



#
# BONUS: After submitting your answers, toy around with
# gamma, kernel, and C.


