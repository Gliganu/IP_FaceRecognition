from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def trainLogisticRegression(xTrain, yTrain):

    C = 10
    classifier = LogisticRegression(C=C, n_jobs=-1)

    # paramGrid = {
    #         "C":[0.1,1,10,100],
    #     }

    # classifier = trainUsingGridSearch(classifier,paramGrid,xTrain,yTrain)

    classifier.fit(xTrain, yTrain)

    return classifier


def trainRandomForestClassifier(xTrain, yTrain):

    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    # classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1, verbose=1)

    # paramGrid = {
    #     "n_estimators":[80,90,100],
    #     "max_features":[0.7,0.8,0.9]
    # }


    #Best  choice is: {'max_features': 0.8, 'n_estimators': 100}
    classifier = RandomForestClassifier(n_jobs=-1, verbose=1, max_features=0.8, n_estimators=100)

    classifier.fit(xTrain, yTrain)


    return classifier



def trainGradientBoostingClassifier(xTrain, yTrain):


    # n_estimators = 120, learning_rate = 0.07
    # max_features= 0.5, max_depth= 6
    # subsample = 0.9
    classifier = GradientBoostingClassifier(n_estimators=120,max_depth=6,min_samples_leaf=1,learning_rate=0.07,max_features=0.5, verbose=1)
    # classifier = GradientBoostingClassifier(verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier



def trainNB(xTrain, yTrain):

    classifier = MultinomialNB()

    classifier.fit(xTrain, yTrain)

    return classifier


def trainUsingGridSearch(classifier, paramGrid, xTrain, yTrain):

    cv = StratifiedKFold(yTrain,n_folds=3)

    newClassifier = GridSearchCV(classifier, scoring="accuracy", param_grid=paramGrid, cv=cv, n_jobs=-1, verbose=1)

    newClassifier.fit(xTrain, yTrain)

    print("Best choice is: {}".format(newClassifier.best_params_))

    return newClassifier

def trainSVM(xTrain, yTrain):
    #
    # classifier = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
    #           decision_function_shape=None, degree=3, gamma=0.005, kernel='rbf',
    #           max_iter=-1, probability=False, random_state=None, shrinking=True,
    #           tol=0.001, verbose=False)

    C = 1000
    gamma = 0.005
    kernel = 'rbf'

    classifier = SVC(C=C,gamma=gamma, kernel=kernel, cache_size=200, class_weight='balanced', coef0=0.0,
                     decision_function_shape=None, degree=3,
                     max_iter=-1, probability=False, random_state=None, shrinking=True,
                     tol=0.001, verbose=True)

    # paramGrid = {
    #         "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
    #     }

    # classifier = trainUsingGridSearch(classifier,paramGrid,xTrain,yTrain)

    classifier = classifier.fit(xTrain, yTrain)

    return classifier



def trainClassifier(xTrain,yTrain):

    print("Training classifier...")

    # classifier = trainGradientBoostingClassifier(xTrain, yTrain)
    # classifier = trainRandomForestClassifier(xTrain, yTrain)
    # classifier = trainNB(xTrain, yTrain)

    # classifier = trainLogisticRegression(xTrain, yTrain)
    classifier = trainSVM(xTrain,yTrain)

    return classifier
