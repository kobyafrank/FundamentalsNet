import sys
import os

    
layerSizeTuple = (20, 12)
neuronizingFunction = "selu"
eta = .08
dropoutRate = 0.35
dataPointsPerBatch = 64
fractionOfDataUsedToTrain = .7
numTrainingEpochs = 1200
steepnessOfCostFunction = .8
eta0 = 1
decayRate = .01
