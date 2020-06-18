from fundamentalsNet import fundamentalsNet
import params
import numpy as np

network = fundamentalsNet()
print("\nTraining neural net with the following parameters")
print("Steepness of Cost Function = %r" %(params.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Using %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Dropout Rate : %r" %(network.dropoutRate))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))

network.train()
print("\n------------END TRAINING------------\n")
print("\n------------BEGIN TESTING------------\n")
network.test()
print("\n------------END TESTING------------\n")

print("\nTraining neural net with the following parameters")
print("Steepness of Cost Function = %r" %(params.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Using %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Dropout Rate : %r" %(network.dropoutRate))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))
