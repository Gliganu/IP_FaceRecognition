from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


import matplotlib.pyplot as plt

def performValidation(yPred, yTest,target_names):
    print(metrics.classification_report(yPred, yTest))

    print("\nNumber of test entries: {}".format(len(yTest)))
