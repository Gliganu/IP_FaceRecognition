import time

from zaCode import Visualizer,ClassifierTrainer,FileManager,Validator,Toolbox

from sklearn.cross_validation import train_test_split

def makePrediction():
    # get dataset
    people = FileManager.getLfwPeople(min_faces_per_person=35, resize=0.4)
    # people = FileManager.getOlivettiFaces()

    # Toolbox.printStatistics(people)

    n_samples, h, w = people.images.shape
    # target_names = people.target_names

    X = people.data
    y = people.target

    #in order to generate more data
    X, y = Toolbox.nudge_dataset(X, y,w,h)

    #needed for RMB
    X = Toolbox.scale(X)
    X = Toolbox.convertBinary(X)

    # split into a training and testing set
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25)

    # Compute PCA
    # xTrain, xTest = Toolbox.performPCA(xTrain, xTest, n_components=150,h=h,w=w)


    #Compute RBM
    # xTrain, xTest = Toolbox.performRBM(xTrain, xTest, n_components=180)


    #train the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain,yTrain)

    yPred = classifier.predict(xTest)

    Validator.performValidation(yPred,yTest,"")




if __name__ == '__main__':
    startTime = time.time()

    makePrediction()

    # Visualizer.calculateLearningCurve(keptColumns)
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("Total run time:{}".format(endTime - startTime))

