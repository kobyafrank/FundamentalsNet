
import sys
import os
import numpy as np
import random
import time
import params
from dataManager import data
import staticFuncs as sF

frequencyOfPrint = 400

class fundamentalsNet:

    def __init__(self):
        self.printCounter = 0
        self.dataObj = data(trainingRatio = params.fractionOfDataUsedToTrain)
        self.inputLayerSize = self.dataObj.inputLayerSize
        self.numTestingPoints = self.dataObj.numTestingCompanies
        if params.neuronizingFunction in ["SELU", "Selu", "selu"]:
            self.neuronizingFunction = sF.SELU
            self.dNeuronizingFunctiondV = sF.dSELUdV
        elif params.neuronizingFunction in ["SIGMOID", "Sigmoid", "sigmoid"]:
            self.neuronizingFunction = sF.sigmoid
            self.dNeuronizingFunctiondV = sF.dSigmoiddV
        elif params.neuronizingFunction in ["SOFTPLUS", "Softplus", "softplus"]:
            self.neuronizingFunction = sF.softplus
            self.dNeuronizingFunctiondV = sF.dSoftplusdV
        else:
            raise ValueError("Invalid neuronizing function. Please choose between SELU, Sigmoid, and Softplus")
        
        if isinstance(params.layerSizeTuple, int):
            self.numLayers = 3
            a = params.layerSizeTuple
            self.LAYER1SIZE = 0
            self.LAYER2SIZE = 0
            self.LAYER3SIZE = self.inputLayerSize
            self.LAYER4SIZE = a
        elif len(params.layerSizeTuple) == 2:
            self.numLayers = 4
            a, b = params.layerSizeTuple
            self.LAYER1SIZE = 0
            self.LAYER2SIZE = self.inputLayerSize
            self.LAYER3SIZE = a
            self.LAYER4SIZE = b
        elif len(params.layerSizeTuple) == 3:
            self.numLayers = 5
            a, b, c = params.layerSizeTuple
            self.LAYER1SIZE = self.inputLayerSize
            self.LAYER2SIZE = a
            self.LAYER3SIZE = b
            self.LAYER4SIZE = c
        else:
            raise ValueError("fundamentalsNet.py only supports 3, 4, or 5 layered networks. Please enter tuple of length 1, 2, or 3, respectively, in the form (layerTwoSize, (layerThreeSize), (layerFourSize)), where the first and final layer is not of variable length so you should not set it.")
        self.LAYER5SIZE = 2

        self.dropoutRate = params.dropoutRate
        self.dataPointsPerBatch = params.dataPointsPerBatch
        self.numTrainingEpochs = params.numTrainingEpochs
        self.steepnessOfCostFunction = params.steepnessOfCostFunction
        
        if self.neuronizingFunction == sF.SELU:
            self.initializeWeightsLeCun()
        else:
            self.initializeWeightsXavier()

    def initializeWeightsXavier(self):
        if self.numLayers >= 5:
            self.layer2Biases = [0 for x in range (self.LAYER2SIZE)]
        if self.numLayers >= 4:
            self.layer3Biases = [0 for x in range (self.LAYER3SIZE)]
        self.layer4Biases = [0 for x in range (self.LAYER4SIZE)]
        self.layer5Biases = [0 for x in range (self.LAYER5SIZE)]
        
        if self.numLayers >= 5:
            constant = np.sqrt(6.) / np.sqrt(self.LAYER1SIZE + self.LAYER2SIZE)
            self.layer21Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER1SIZE)] \
                for y in range (self.LAYER2SIZE)]
        if self.numLayers >= 4:
            constant = np.sqrt(6.) / np.sqrt(self.LAYER2SIZE + self.LAYER3SIZE)
            self.layer32Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER2SIZE)] \
                for y in range (self.LAYER3SIZE)]
        constant = np.sqrt(6.) / np.sqrt(self.LAYER3SIZE + self.LAYER4SIZE)
        self.layer43Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER3SIZE)] \
            for y in range (self.LAYER4SIZE)]
        constant = np.sqrt(6.) / np.sqrt(self.LAYER4SIZE + self.LAYER5SIZE)
        self.layer54Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER4SIZE)] \
            for y in range (self.LAYER5SIZE)]
    
    def initializeWeightsLeCun(self):
        if self.numLayers >= 5:
            self.layer2Biases = [0 for x in range (self.LAYER2SIZE)]
        if self.numLayers >= 4:
            self.layer3Biases = [0 for x in range (self.LAYER3SIZE)]
        self.layer4Biases = [0 for x in range (self.LAYER4SIZE)]
        self.layer5Biases = [0 for x in range (self.LAYER5SIZE)]
    
        if self.numLayers >= 5:
            self.layer21Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER2SIZE)) for x in range (self.LAYER1SIZE)] \
                for y in range (self.LAYER2SIZE)]
        if self.numLayers >= 4:
            self.layer32Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER3SIZE)) for x in range (self.LAYER2SIZE)] \
                for y in range (self.LAYER3SIZE)]
        self.layer43Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER4SIZE)) for x in range (self.LAYER3SIZE)] \
            for y in range (self.LAYER4SIZE)]
        self.layer54Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER5SIZE)) for x in range (self.LAYER4SIZE)] \
            for y in range (self.LAYER5SIZE)]

    def sendThroughNetTrain(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        l4Kept = []
        l3Kept = []
        l2Kept = []
        if self.numLayers == 5:
            if len(inputData) != self.LAYER1SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
            layer1Values = inputData
            layer2Values = [0 for x in range (self.LAYER2SIZE)]
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L2Neuron in range (self.LAYER2SIZE):
                if random.uniform(0, 1) < self.dropoutRate:
                    if self.neuronizingFunction == sF.SELU:
                        layer2Values[L2Neuron] = sF.alphaStar
                    else:
                        layer2Values[L2Neuron] = 0
                else:
                    l2Kept.append(L2Neuron)
                    layer2Values[L2Neuron] = self.neuronizingFunction(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
            for L3Neuron in range (self.LAYER3SIZE):
                if random.uniform(0, 1) < self.dropoutRate:
                    if self.neuronizingFunction == sF.SELU:
                        layer3Values[L3Neuron] = sF.alphaStar
                    else:
                        layer3Values[L3Neuron] = 0
                else:
                    l3Kept.append(L3Neuron)
                    layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
                    
        elif self.numLayers == 4:
            if len(inputData) != self.LAYER2SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER2SIZE)
            layer2Values = inputData
            l2Kept = [x for x in range (self.LAYER2SIZE)]
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L3Neuron in range (self.LAYER3SIZE):
                if random.uniform(0, 1) < self.dropoutRate:
                    if self.neuronizingFunction == sF.SELU:
                        layer3Values[L3Neuron] = sF.alphaStar
                    else:
                        layer3Values[L3Neuron] = 0
                else:
                    l3Kept.append(L3Neuron)
                    layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
                    
        else:
            if len(inputData) != self.LAYER3SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER3SIZE)
            layer3Values = inputData
            l3Kept = [x for x in range (self.LAYER3SIZE)]
            
        layer4Values = [0 for x in range (self.LAYER4SIZE)]
        layer5Values = [0 for x in range (self.LAYER5SIZE)]
        for L4Neuron in range (self.LAYER4SIZE):
            if random.uniform(0, 1) < self.dropoutRate:
                if self.neuronizingFunction == sF.SELU:
                    layer4Values[L4Neuron] = sF.alphaStar
                else:
                    layer4Values[L4Neuron] = 0
            else:
                l4Kept.append(L4Neuron)
                layer4Values[L4Neuron] = self.neuronizingFunction(self.layer4Biases[L4Neuron] + np.dot(layer3Values, self.layer43Weights[L4Neuron]))
        for L5Neuron in range (self.LAYER5SIZE):
            layer5Values[L5Neuron] = self.layer5Biases[L5Neuron] + np.dot(layer4Values, self.layer54Weights[L5Neuron])
            
        layer5Values = sF.softmax(layer5Values)
        if self.printCounter == 0:
            print((layer5Values, trueResult))
        self.printCounter = (self.printCounter + 1) % frequencyOfPrint
        squaredError = self.calculateSquaredError(layer5Values, trueResult)
        correctDirection = sF.sameSign(sF.directionize(layer5Values), trueResult)
        
        #Calculates gradient for training purposes
        if self.numLayers == 5:
            gradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
            gradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
            gradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
            gradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        elif self.numLayers == 4:
            gradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
            gradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        
        gradientLayer4Biases = [0 for x in range(self.LAYER4SIZE)]
        gradientLayer5Biases = [0 for x in range(self.LAYER5SIZE)]
        gradientLayer43Weights = [[0 for x in range(self.LAYER3SIZE)] for y in range (self.LAYER4SIZE)]
        gradientLayer54Weights = [[0 for x in range(self.LAYER4SIZE)] for y in range (self.LAYER5SIZE)]

        for L5Neuron in range (self.LAYER5SIZE):
            flattened = (1. / (1. + np.exp(-1. * self.steepnessOfCostFunction * trueResult)))
            if (L5Neuron == 0):
                dCostdL5PostNeuronizingFunction = 2 * (layer5Values[L5Neuron] - flattened)
            else:
                dCostdL5PostNeuronizingFunction = 2 * (layer5Values[L5Neuron] - (1 - flattened))
            dL5PostNeuronizingFunctiondL5V = sF.dSoftmaxdV(layer5Values, L5Neuron)
            gradientLayer5Biases[L5Neuron] = dCostdL5PostNeuronizingFunction * dL5PostNeuronizingFunctiondL5V
            
            for L4Neuron in l4Kept:
                dL5VdL54Weight = layer4Values[L4Neuron]
                gradientLayer54Weights[L5Neuron][L4Neuron] = gradientLayer5Biases[L5Neuron] * dL5VdL54Weight
        
                dL5VdL4PostNeuronizingFunction = self.layer54Weights[L5Neuron][L4Neuron]
                dL4PostNeuronizingFunctiondL4V = self.dNeuronizingFunctiondV(layer4Values[L4Neuron])
                gradientLayer4Biases[L4Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V)
        
                for L3Neuron in l3Kept:
                    dL4VdL43Weight = layer3Values[L3Neuron]
                    gradientLayer43Weights[L4Neuron][L3Neuron] += gradientLayer4Biases[L4Neuron] * dL4VdL43Weight
                    
                    if self.numLayers >= 4:
                        dL4VdL3PostNeuronizingFunction = self.layer43Weights[L4Neuron][L3Neuron]
                        dL3PostNeuronizingFunctiondL3V = self.dNeuronizingFunctiondV(layer3Values[L3Neuron])
                        gradientLayer3Biases[L3Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V)
            
                        for L2Neuron in l2Kept:
                            dL3VdL32Weight = layer2Values[L2Neuron]
                            gradientLayer32Weights[L3Neuron][L2Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL32Weight)
                        
                            if self.numLayers >= 5:
                                dL3VdL2PostNeuronizingFunction = self.layer32Weights[L3Neuron][L2Neuron]
                                dL2PostNeuronizingFunctiondL2V = self.dNeuronizingFunctiondV(layer2Values[L2Neuron])
                                gradientLayer2Biases[L2Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL2PostNeuronizingFunction * dL2PostNeuronizingFunctiondL2V)
                
                                for L1Neuron in range (self.LAYER1SIZE):
                                    dL2VdL21Weights = layer1Values[L1Neuron]
                                    gradientLayer21Weights[L2Neuron][L1Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL2PostNeuronizingFunction * dL2PostNeuronizingFunctiondL2V * dL2VdL21Weights)
                                                
        #print("Actual: l4drop = %r, l3drop = %r, l2drop = %r" %(self.LAYER4SIZE - len(l4Kept), self.LAYER3SIZE - len(l3Kept), self.LAYER2SIZE - len(l2Kept)))
        if self.numLayers == 5:
            return (correctDirection, squaredError, gradientLayer21Weights, gradientLayer32Weights, gradientLayer43Weights, gradientLayer54Weights, gradientLayer2Biases, gradientLayer3Biases, gradientLayer4Biases, gradientLayer5Biases)
        elif self.numLayers == 4:
            return (correctDirection, squaredError, gradientLayer32Weights, gradientLayer43Weights, gradientLayer54Weights, gradientLayer3Biases, gradientLayer4Biases, gradientLayer5Biases)
        else:
            return (correctDirection, squaredError, gradientLayer43Weights, gradientLayer54Weights, gradientLayer4Biases, gradientLayer5Biases)

    def runBatch(self, epoch):
        #Runs a batch of data, logs average gradients and error
        totalCorrectDirection = 0
        totalSquaredError = 0
        
        if self.numLayers == 5:
            totalGradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
            totalGradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
        if self.numLayers == 4:
            totalGradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
            totalGradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
            
        totalGradientLayer43Weights = [[0 for x in range(self.LAYER3SIZE)] for y in range (self.LAYER4SIZE)]
        totalGradientLayer54Weights = [[0 for x in range(self.LAYER4SIZE)] for y in range (self.LAYER5SIZE)]
        totalGradientLayer4Biases = [0 for x in range(self.LAYER4SIZE)]
        totalGradientLayer5Biases = [0 for x in range(self.LAYER5SIZE)]
        
        for x in range (self.dataPointsPerBatch):
            inputData, trueResult = self.dataObj.getNewDataPoint()
            resTuple = self.sendThroughNetTrain(inputData, trueResult)
            if self.numLayers == 5:
                correctDirection, newSquaredError, newGradientLayer21Weights, newGradientLayer32Weights, newGradientLayer43Weights, newGradientLayer54Weights, newGradientLayer2Biases, newGradientLayer3Biases, newGradientLayer4Biases, newGradientLayer5Biases = resTuple
            elif self.numLayers == 4:
                correctDirection, newSquaredError, newGradientLayer32Weights, newGradientLayer43Weights, newGradientLayer54Weights, newGradientLayer3Biases, newGradientLayer4Biases, newGradientLayer5Biases = resTuple
            elif self.numLayers == 3:
                correctDirection, newSquaredError, newGradientLayer43Weights, newGradientLayer54Weights, newGradientLayer4Biases, newGradientLayer5Biases = resTuple
            else:
                raise ValueError("Unforseen number of layers")
            
            if (correctDirection):
                totalCorrectDirection += 1
            totalSquaredError += newSquaredError
            if self.numLayers == 5:
                totalGradientLayer21Weights = np.add(totalGradientLayer21Weights, newGradientLayer21Weights)
                totalGradientLayer2Biases = np.add(totalGradientLayer2Biases, newGradientLayer2Biases)
            if self.numLayers == 4:
                totalGradientLayer32Weights = np.add(totalGradientLayer32Weights, newGradientLayer32Weights)
                totalGradientLayer3Biases = np.add(totalGradientLayer3Biases, newGradientLayer3Biases)
            totalGradientLayer43Weights = np.add(totalGradientLayer43Weights, newGradientLayer43Weights)
            totalGradientLayer54Weights = np.add(totalGradientLayer54Weights, newGradientLayer54Weights)
            totalGradientLayer4Biases = np.add(totalGradientLayer4Biases, newGradientLayer4Biases)
            totalGradientLayer5Biases = np.add(totalGradientLayer5Biases, newGradientLayer5Biases)

        correctDirectionRate = float(totalCorrectDirection) / float(self.dataPointsPerBatch)
        averageSquaredError = totalSquaredError / float(self.dataPointsPerBatch)
        if self.numLayers == 5:
            averageGradientLayer21Weights = np.divide(totalGradientLayer21Weights, float(self.dataPointsPerBatch))
            averageGradientLayer2Biases = np.divide(totalGradientLayer2Biases, float(self.dataPointsPerBatch))
        if self.numLayers == 4:
            averageGradientLayer32Weights = np.divide(totalGradientLayer32Weights, float(self.dataPointsPerBatch))
            averageGradientLayer3Biases = np.divide(totalGradientLayer3Biases, float(self.dataPointsPerBatch))
        averageGradientLayer43Weights = np.divide(totalGradientLayer43Weights, float(self.dataPointsPerBatch))
        averageGradientLayer54Weights = np.divide(totalGradientLayer54Weights, float(self.dataPointsPerBatch))
        averageGradientLayer4Biases = np.divide(totalGradientLayer4Biases, float(self.dataPointsPerBatch))
        averageGradientLayer5Biases = np.divide(totalGradientLayer5Biases, float(self.dataPointsPerBatch))
        
        #Updates weights and biases accordingly
        if self.numLayers == 5:
            self.layer21Weights = np.subtract(self.layer21Weights, np.multiply(averageGradientLayer21Weights, sF.eta(epoch)))
            self.layer2Biases = np.subtract(self.layer2Biases, np.multiply(averageGradientLayer2Biases, sF.eta(epoch)))
        if self.numLayers == 4:
            self.layer32Weights = np.subtract(self.layer32Weights, np.multiply(averageGradientLayer32Weights, sF.eta(epoch)))
            self.layer3Biases = np.subtract(self.layer3Biases, np.multiply(averageGradientLayer3Biases, sF.eta(epoch)))
        self.layer43Weights = np.subtract(self.layer43Weights, np.multiply(averageGradientLayer43Weights, sF.eta(epoch)))
        self.layer54Weights = np.subtract(self.layer54Weights, np.multiply(averageGradientLayer54Weights, sF.eta(epoch)))
        self.layer4Biases = np.subtract(self.layer4Biases, np.multiply(averageGradientLayer4Biases, sF.eta(epoch)))
        self.layer5Biases = np.subtract(self.layer5Biases, np.multiply(averageGradientLayer5Biases, sF.eta(epoch)))
        
        return averageSquaredError, correctDirectionRate
        
    def train(self):
        averageSquaredErrorProgress = correctDirectionRateProgress = 0
        for epoch in range (self.numTrainingEpochs):
            if (epoch == 1):
                start = time.perf_counter()
            averageSquaredError, correctDirectionRate  = self.runBatch(epoch)
            averageSquaredErrorProgress += averageSquaredError
            correctDirectionRateProgress += correctDirectionRate
            if (epoch == 1):
                end = time.perf_counter()
                timeElapsed = end - start
                print("\n[At this rate of %r sec/epoch, it will take approximately %r minutes, or %r hours, to train the neural net]\n" %(round(timeElapsed, 4), round((timeElapsed * self.numTrainingEpochs) / 60., 2), round((timeElapsed * self.numTrainingEpochs) / 3600., 3)))
            #print("Epoch %r || Mean Squared Error = %r" %(epoch, round(averageSquaredError, 4)))
            if epoch % 20 == 0 and epoch != 0:
                print("\nPROGRESS TRACKER: Epoch %r || MSE Avg. = %r || Correct Direction Rate = %r\n" %(epoch, round(averageSquaredErrorProgress / 20., 4), round(correctDirectionRateProgress / 20., 4)))
                averageSquaredErrorProgress = correctDirectionRateProgress = 0
        
    def sendThroughNetTest(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        if self.numLayers == 5:
            if len(inputData) != self.LAYER1SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
            layer1Values = inputData
            layer2Values = [0 for x in range (self.LAYER2SIZE)]
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L2Neuron in range (self.LAYER2SIZE):
                layer2Values[L2Neuron] = self.neuronizingFunction(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
            for L3Neuron in range (self.LAYER3SIZE):
                layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        elif self.numLayers == 4:
            if len(inputData) != self.LAYER2SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER2SIZE)
            layer2Values = inputData
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L3Neuron in range (self.LAYER3SIZE):
                layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        else:
            if len(inputData) != self.LAYER3SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER3SIZE)
            layer3Values = inputData
        layer4Values = [0 for x in range (self.LAYER4SIZE)]
        layer5Values = [0 for x in range (self.LAYER5SIZE)]
        for L4Neuron in range (self.LAYER4SIZE):
            layer4Values[L4Neuron] = self.neuronizingFunction(self.layer4Biases[L4Neuron] + np.dot(layer3Values, self.layer43Weights[L4Neuron]))
        for L5Neuron in range (self.LAYER5SIZE):
            layer5Values[L5Neuron] = self.layer5Biases[L5Neuron] + np.dot(layer4Values, self.layer54Weights[L5Neuron])
        layer5Values = sF.softmax(layer5Values)
        squaredError = self.calculateSquaredError(layer5Values, trueResult)
        return layer5Values
        
    def sendThroughNetCompaniesToLookAt(self, inputData):
        if self.numLayers == 5:
            if len(inputData) != self.LAYER1SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
            layer1Values = inputData
            layer2Values = [0 for x in range (self.LAYER2SIZE)]
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L2Neuron in range (self.LAYER2SIZE):
                layer2Values[L2Neuron] = self.neuronizingFunction(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
            for L3Neuron in range (self.LAYER3SIZE):
                layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        elif self.numLayers == 4:
            if len(inputData) != self.LAYER2SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER2SIZE)
            layer2Values = inputData
            layer3Values = [0 for x in range (self.LAYER3SIZE)]
            for L3Neuron in range (self.LAYER3SIZE):
                layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        else:
            if len(inputData) != self.LAYER3SIZE:
                raise ValueError("Input data is not of length %r" % self.LAYER3SIZE)
            layer3Values = inputData
        layer4Values = [0 for x in range (self.LAYER4SIZE)]
        layer5Values = [0 for x in range (self.LAYER5SIZE)]
        for L4Neuron in range (self.LAYER4SIZE):
            layer4Values[L4Neuron] = self.neuronizingFunction(self.layer4Biases[L4Neuron] + np.dot(layer3Values, self.layer43Weights[L4Neuron]))
        for L5Neuron in range (self.LAYER5SIZE):
            layer5Values[L5Neuron] = self.layer5Biases[L5Neuron] + np.dot(layer4Values, self.layer54Weights[L5Neuron])
        layer5Values = sF.softmax(layer5Values)
        return layer5Values
    
    
    def test(self):
        if self.numLayers == 5:
            np.multiply(1. - self.dropoutRate, self.layer21Weights)
        if self.numLayers == 4:
            np.multiply(1. - self.dropoutRate, self.layer32Weights)
        np.multiply(1. - self.dropoutRate, self.layer43Weights)
        
        trueTotalUp = 0
        guessedTotalUp = 0
        totalCorrectDirection = 0
        correctPointSixPlus = 0
        correctPointSixPlusPos = 0
        totalPointSixPlus = 0
        totalPointSixPlusPos = 0
        correctPointSevenPlus = 0
        correctPointSevenPlusPos = 0
        totalPointSevenPlus = 0
        totalPointSevenPlusPos = 0
        correctPointEightPlus = 0
        correctPointEightPlusPos = 0
        totalPointEightPlus = 0
        totalPointEightPlusPos = 0
        correctPointNinePlus = 0
        correctPointNinePlusPos = 0
        totalPointNinePlus = 0
        totalPointNinePlusPos = 0
        self.dataObj.switchToTest()
        
        for test in range (self.numTestingPoints):
            inputData, trueResult = self.dataObj.getNewDataPoint()
            print(inputData)
            print(trueResult)
            guessedResult = self.sendThroughNetTest(inputData, trueResult)
            guessedUp = (1 == sF.directionize(guessedResult))
            if trueResult >= 0:
                trueTotalUp += 1
            if guessedUp:
                guessedTotalUp += 1
            correctDirection = sF.sameSign(sF.directionize(guessedResult), trueResult)
            if guessedResult[0] <= .1 or guessedResult[0] >= .9:
                totalPointNinePlus += 1
                totalPointEightPlus += 1
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if trueResult >= 0:
                    totalPointNinePlusPos += 1
                    totalPointEightPlusPos += 1
                    totalPointSevenPlusPos += 1
                    totalPointSixPlusPos += 1
                if correctDirection:
                    correctPointNinePlus += 1
                    correctPointEightPlus += 1
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
                    if guessedUp:
                        correctPointNinePlusPos += 1
                        correctPointEightPlusPos += 1
                        correctPointSevenPlusPos += 1
                        correctPointSixPlusPos += 1
            elif guessedResult[0] <= .2 or guessedResult[0] >= .8:
                totalPointEightPlus += 1
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if trueResult >= 0:
                    totalPointEightPlusPos += 1
                    totalPointSevenPlusPos += 1
                    totalPointSixPlusPos += 1
                if correctDirection:
                    correctPointEightPlus += 1
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
                    if guessedUp:
                        correctPointEightPlusPos += 1
                        correctPointSevenPlusPos += 1
                        correctPointSixPlusPos += 1
            elif guessedResult[0] <= .3 or guessedResult[0] >= .7:
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if trueResult >= 0:
                    totalPointSevenPlusPos += 1
                    totalPointSixPlusPos += 1
                if correctDirection:
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
                    if guessedUp:
                        correctPointSevenPlusPos += 1
                        correctPointSixPlusPos += 1
            elif guessedResult[0] <= .4 or guessedResult[0] >= .6:
                totalPointSixPlus += 1
                if trueResult >= 0:
                    totalPointSixPlusPos += 1
                if correctDirection:
                    correctPointSixPlus += 1
                    if guessedUp:
                        correctPointSixPlusPos += 1
            if correctDirection:
                totalCorrectDirection += 1
            print("Test %r || True Value = %r || Correct : %r || Guessed %r%% Up, %r%% Down" %(test, trueResult, correctDirection, round(guessedResult[0] * 100., 4), round(guessedResult[1] * 100., 4)))
        print("\nTesting over\n")
        print("%r fraction of companies outperformed the S and P" %(float(trueTotalUp) / float(self.numTestingPoints)))
        print("%r fraction of companies were guessed to outperform the S and P" %(float(guessedTotalUp) / float(self.numTestingPoints)))
        print("%r fraction of companies had their performance correctly guessed, or %r / %r" %(float(totalCorrectDirection) / float(self.numTestingPoints), totalCorrectDirection, self.numTestingPoints))
        if totalPointSixPlus > 0:
            ratio = float(correctPointSixPlus) / float(totalPointSixPlus)
        else:
            ratio = 0.
        print("\n%r fraction of companies with confidence over .6 had their performance correctly guessed, or %r / %r" %(ratio, correctPointSixPlus, totalPointSixPlus))
        if totalPointSixPlusPos > 0:
            ratio = float(correctPointSixPlusPos) / float(totalPointSixPlusPos)
        else:
            ratio = 0.
        print("An outperform guess with .6 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointSixPlusPos, totalPointSixPlusPos))
        if totalPointSixPlus - totalPointSixPlusPos > 0:
            ratio = float(correctPointSixPlus - correctPointSixPlusPos) / float(totalPointSixPlus - totalPointSixPlusPos)
        else:
            ratio = 0.
        print("An underperform guess with .6 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointSixPlus - correctPointSixPlusPos, totalPointSixPlus - totalPointSixPlusPos))
        
        if totalPointSevenPlus > 0:
            ratio = float(correctPointSevenPlus) / float(totalPointSevenPlus)
        else:
            ratio = 0.
        print("\n%r fraction of companies with confidence over .7 had their performance correctly guessed, or %r / %r" %(ratio, correctPointSevenPlus, totalPointSevenPlus))
        if totalPointSevenPlusPos > 0:
            ratio = float(correctPointSevenPlusPos) / float(totalPointSevenPlusPos)
        else:
            ratio = 0.
        print("An outperform guess with .7 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointSevenPlusPos, totalPointSevenPlusPos))
        if totalPointSevenPlus - totalPointSevenPlusPos > 0:
            ratio = float(correctPointSevenPlus - correctPointSevenPlusPos) / float(totalPointSevenPlus - totalPointSevenPlusPos)
        else:
            ratio = 0.
        print("An underperform guess with .7 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointSevenPlus - correctPointSevenPlusPos, totalPointSevenPlus - totalPointSevenPlusPos))
        
        if totalPointEightPlus > 0:
            ratio = float(correctPointEightPlus) / float(totalPointEightPlus)
        else:
            ratio = 0.
        print("\n%r fraction of companies with confidence over .8 had their performance correctly guessed, or %r / %r" %(ratio, correctPointEightPlus, totalPointEightPlus))
        if totalPointEightPlusPos > 0:
            ratio = float(correctPointEightPlusPos) / float(totalPointEightPlusPos)
        else:
            ratio = 0.
        print("An outperform guess with .8 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointEightPlusPos, totalPointEightPlusPos))
        if totalPointEightPlus - totalPointEightPlusPos > 0:
            ratio = float(correctPointEightPlus - correctPointEightPlusPos) / float(totalPointEightPlus - totalPointEightPlusPos)
        else:
            ratio = 0.
        print("An underperform guess with .8 confidence or higher had success rate %r, or %r / %r" %(ratio, correctPointEightPlus - correctPointEightPlusPos, totalPointEightPlus - totalPointEightPlusPos))
        
        if totalPointNinePlus > 0:
            ratio = float(correctPointNinePlus) / float(totalPointNinePlus)
        else:
            ratio = 0.
        print("\n%r fraction of companies with confidence over .9 had their performance correctly guessed, or %r / %r" %(ratio, correctPointNinePlus, totalPointNinePlus))
        if totalPointNinePlusPos > 0:
            ratio = float(correctPointNinePlusPos) / float(totalPointNinePlusPos)
        else:
            ratio = 0.
        print("An outperform guess with .9 confidence or higher  had success rate %r, or %r / %r" %(ratio, correctPointNinePlusPos, totalPointNinePlusPos))
        if totalPointNinePlus - totalPointNinePlusPos > 0:
            ratio = float(correctPointNinePlus - correctPointNinePlusPos) / float(totalPointNinePlus - totalPointNinePlusPos)
        else:
            ratio = 0.
        print("An underperform guess with .9 confidence or higher  had success rate %r, or %r / %r" %(ratio, correctPointNinePlus - correctPointNinePlusPos, totalPointNinePlus - totalPointNinePlusPos))
        
        #Now look at companies which I'm actually interested in predicting
        inputData = self.dataObj.getPLNTDataPoint()
        print("\nPLNT")
        #print(inputData)
        guessedResult = self.sendThroughNetCompaniesToLookAt(inputData)
        print("Prediction for PLNT: %r%% Up, %r%% Down" %(round(guessedResult[0] * 100., 4), round(guessedResult[1] * 100., 4)))
        
        inputData = self.dataObj.getNYTDataPoint()
        print("\nNYT")
        #print(inputData)
        guessedResult = self.sendThroughNetCompaniesToLookAt(inputData)
        print("Prediction for NYT: %r%% Up, %r%% Down" %(round(guessedResult[0] * 100., 4), round(guessedResult[1] * 100., 4)))
        

    def calculateSquaredError(self, guess, trueResult):
        flattened = (1. / (1. + np.exp(-1. * self.steepnessOfCostFunction * trueResult)))
        return (guess[0] - flattened)**2 + (guess[1] - (1 - flattened))**2

    
