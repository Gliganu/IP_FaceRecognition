from time import time
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import pandas as pd



def makePrediction():
    print('Loading Data')
    people = fetch_lfw_people(resize=0.4)
    print('Done!')

    # Find out how many faces we have, and
    # the size of each picture from.
    n_samples, h, w = people.images.shape

    X = people.data
    n_features = X.shape[1]

    y = people.target
    target_names = people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_images: %d " % n_samples)
    print("n_features: %d " % n_features)
    print("n_classes: %d " % n_classes)

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    # Compute the PCA (eigenfaces) on the face dataset
    n_components = 150

    pca = RandomizedPCA(
        n_components=n_components, whiten=True).fit(X_train)

    eigenfaces = pca.components_.reshape((n_components, h, w))
    X_train_pca = pca.transform(X_train)