import time

import zaCode.Toolbox as Toolbox

if __name__ == '__main__':
    startTime = time.time()

    Toolbox.makePrediction()

    # Visualizer.calculateLearningCurve(keptColumns)
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("Total run time:{}".format(endTime - startTime))
