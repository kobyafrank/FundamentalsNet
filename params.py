import sys
import os

    
layerSizeTuple = (20, 12)
neuronizingFunction = "selu"
eta = .08
dropoutRate = 0.35
dataPointsPerBatch = 16
fractionOfDataUsedToTrain = .7
numTrainingEpochs = 5000
steepnessOfCostFunction = 1.2
eta0 = 1
decayRate = .01
