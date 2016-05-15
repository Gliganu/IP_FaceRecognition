import numpy as np
from sklearn.decomposition import RandomizedPCA
import cv2
from sklearn.neural_network import BernoulliRBM
import scipy as sp
import zaCode.Visualizer as Visualizer

from scipy.ndimage.filters import convolve

def printStatistics(people):
    n_samples, h, w = people.images.shape

    X = people.data
    n_features = X.shape[1]

    target_names = people.target_names
    n_classes = target_names.shape[0]

    print("H = {}, W = {}".format(h,w))
    print("Total dataset size:")
    print("n_images: %d " % n_samples)
    print("n_features: %d " % n_features)
    print("n_classes: %d " % n_classes)


def performRBM(xTrain, xTest,n_components, withVisualization = False):
    print("Performing RBM...")

    rbm = BernoulliRBM(n_components=n_components, learning_rate=0.01, batch_size=10, n_iter=50, verbose=True, random_state=None).fit(xTrain)

    xTrain = rbm.transform(xTrain)
    xTest = rbm.transform(xTest)

    #for Olivetti
    if withVisualization:
        comp = rbm.components_
        image_shape = (64, 64)
        Visualizer.plot_gallery('RBM componenets', comp[:16],image_shape, 4,4)

    return xTrain, xTest


def performPCA(xTrain, xTest, n_components, h, w):
    print("Performing PCA...")

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(xTrain)

    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)

    # eigenfaces = pca.components_.reshape((n_components, h, w))
    # eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    # Visualizer.plot_gallery(eigenfaces, eigenface_titles, h, w)

    return xTrain, xTest


def nudge_dataset(X, Y,width,height):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
    [[0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 1],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 0],
    [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((width, height)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# def nudge(X, y,width,height):
# 	# initialize the translations to shift the image one pixel
# 	# up, down, left, and right, then initialize the new data
# 	# matrix and targets
# 	translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
# 	data = []
# 	target = []
#
# 	# loop over each of the digits
# 	for (image, label) in zip(X, y):
# 		# pixel intensities to a width x height 'image'
# 		image = image.reshape(width, height)
#
# 		# loop over the translations
# 		for (tX, tY) in translations:
# 			# translate the image
# 			M = np.float32([[1, 0, tX], [0, 1, tY]])
# 			trans = cv2.warpAffine(image, M, (width, height))
#
# 			# update the list of data and target
# 			data.append(trans.flatten())
# 			target.append(label)
#
# 	# return a tuple of the data matrix and targets
# 	return np.array(data), np.array(target)


def scale(X):
    """scaling between 0 and 1 of grayscale images """
    X = np.asarray(X, 'float32')
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    return X


def convertBinary(X):
    X = X > 0.5
    return X