import numpy as np
import random
import params

alpha = 1.6732632423543772848170429916717
gamma = 1.0507009873554804934193349852946
alphaStar = -1. * alpha * gamma

def sameSign(x, y):
    return ((x >= 0 and y >= 0) or (x < 0 and y < 0))

def directionize(ls):
    if (ls[0] >= ls[1]):
        return 1
    else:
        return -1
        
def softmax(ls):
    sum = 0
    m = max(ls)
    for x in ls:
        sum += np.exp(x - m)
    return [np.exp(i - m) / sum for i in ls]
    
def dSoftmaxdV(ls, i):
    sum = 0
    m = max(ls)
    for x in ls:
        sum += np.exp(x - m)
    s = np.exp(i - m) / sum
    return s * (1 - s)

def softplus(x):
    try:
        out = np.log(1 + np.exp(x))
    except:
        out = x
    return out
    
def dSoftplusdV(x):
    return (1. / (1. + np.exp(-1. * x)))

def SELU(x):
    if x > 0:
        return gamma * x
    else:
        return gamma * alpha * (np.exp(x) - 1.)
    
def dSELUdV(x):
    if x > 0:
        return gamma
    else:
        return gamma * alpha * np.exp(x)
    
def sigmoid(x):
    try:
        out =  (1. / (1. + np.exp(-1. * x)))
    except:
        if x < 0:
            out = 0.
        else:
            out = 1.
    return out
    
def dSigmoiddV(x):
    try:
        out = (1. / (1. + np.exp(-1. * x))) * (1. - (1. / (1. + np.exp(-1. * x))))
    except:
        out = 0
    return out
        
def mapMinMax(ls):
    #print(ls)
    maximum = max(ls)
    minimum = min(ls)
    range = maximum - minimum
    return [2. * ((x - minimum) / range) - 1. for x in ls]
    #print(ls)
        
def mapMinMaxLog(ls):
    #print(ls)
    maximum = max(ls)
    minimum = min(ls)
    range = maximum - minimum
    return [2. * (np.log(x - minimum + 1) / np.log(maximum - minimum + 1)) - 1. for x in ls]
    #print(ls)
    
def eta(epoch):
    return params.eta0 / (1 + params.decayRate * epoch)
