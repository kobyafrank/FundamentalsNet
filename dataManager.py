import sys
import os
import numpy as np
import random
import staticFuncs as sF

class data:
    def __init__(self, trainingRatio = .6):
        self.trainingRatio = trainingRatio
        SPJan2018 = 268.85
        SPMay2019 = 286.
        self.SPPerformance = 100. * (SPMay2019 - SPJan2018) / SPJan2018
        try:
            self.path = os.environ['FUNDDATADIR']
        except KeyError:
            raise KeyError('Environment variable "FUNDDATADIR" not set! Please set "FUNDDATADIR" to point where all market data should live first by appropriately updating variable in .bash_profile, for example using the phrase: export FUNDDATADIR=\'path/goes/here\'')
        
        self.industryDict = {}
        fullPath = os.path.join(self.path, "industries.txt")
        with open(fullPath, 'r') as f:
            industryFileInfo = f.readlines()
        industryCounter = 0
        for line in industryFileInfo:
            line = line.rstrip('\n')
            if line not in self.industryDict:
                self.industryDict[line] = industryCounter
                industryCounter += 1
        self.numIndustries = industryCounter
        print(self.industryDict)
        
        self.ticker = []
        self.sector = []
        self.price = []
        self.PToE = []
        self.divYield = []
        self.earningsPerShare = []
        self.high = []
        self.low = []
        self.normalizedRange = []
        self.marketCap = []
        self.ebitda = []
        self.priceToSales = []
        self.priceToBook = []
        self.nextYearsPrice = []
        self.twoYearsPrice = []
        
        self.inputLayerSize = 8 + self.numIndustries

        fullPath = os.path.join(self.path, "fundamentals.txt")
        with open(fullPath, 'r') as f:
            dataFileInfo = f.readlines()
        self.numCompanies = len(dataFileInfo) - 2
        self.numTrainingCompanies = int(self.numCompanies * self.trainingRatio)
        self.numTestingCompanies = self.numCompanies - self.numTrainingCompanies
        print("num companies")
        print(self.numCompanies)
        print("num training companies")
        print(self.numTrainingCompanies)
        print("num testing companies")
        print(self.numTestingCompanies)
        
        counter = 0
        for rawLine in dataFileInfo:
            rawLine = rawLine.rstrip('\n')
            line = rawLine.split('\t')
            #print(rawLine)
            #print(line)
            self.ticker.append(line[0])
            self.sector.append(line[2])
            self.price.append(float(line[3]))
            self.PToE.append(float(line[4]))
            self.divYield.append(float(line[5]))
            self.earningsPerShare.append(float(line[6]))
            self.high.append(float(line[7]))
            self.low.append(float(line[8]))
            normalizedRange = (float(line[7]) - float(line[8])) / float(line[7])
            self.normalizedRange.append(normalizedRange)
            self.marketCap.append(float(line[9]))
            self.ebitda.append(float(line[10]))
            self.priceToSales.append(float(line[11]))
            self.priceToBook.append(float(line[12]))
            self.nextYearsPrice.append(float(line[13]))
            self.twoYearsPrice.append(float(line[14]))
            #print(self.ticker[counter])
            '''
            print("ticker")
            print(self.ticker[counter])
            print("sector")
            print(self.sector[counter])
            print("price")
            print(self.price[counter])
            print("PtoE")
            print(self.PToE[counter])
            print("divYield")
            print(self.divYield[counter])
            print("EPS")
            print(self.earningsPerShare[counter])
            print("high")
            print(self.high[counter])
            print("low")
            print(self.low[counter])
            print("normRange")
            print(self.normalizedRange[counter])
            print("marketCap")
            print(self.marketCap[counter])
            print("ebitda")
            print(self.ebitda[counter])
            print("PToS")
            print(self.priceToSales[counter])
            print("PToB")
            print(self.priceToBook[counter])
            print("nextYearsPrice")
            print(self.nextYearsPrice[counter])
            print("twoYearsPrice")
            print(self.twoYearsPrice[counter])
            '''
            counter += 1
            
            
        #print("p to e")
        self.PToE = sF.mapMinMax(self.PToE)
        #print("divyield")
        self.divYield = sF.mapMinMax(self.divYield)
        #print("earnings per share")
        self.earningsPerShare = sF.mapMinMax(self.earningsPerShare)
        #print("normalized range")
        self.normalizedRange = sF.mapMinMax(self.normalizedRange)
        #print("market cap")
        self.marketCap = sF.mapMinMaxLog(self.marketCap)
        #print("ebitda")
        self.ebitda = sF.mapMinMaxLog(self.ebitda)
        #print("price to sales")
        self.priceToSales = sF.mapMinMaxLog(self.priceToSales)
        #print("price to book")
        self.priceToBook = sF.mapMinMaxLog(self.priceToBook)
        
        self.permutedCompanies = np.random.permutation(self.numCompanies)
        self.trainingSet = self.permutedCompanies[0 : self.numTrainingCompanies]
        self.testingSet = self.permutedCompanies[self.numTrainingCompanies:]
        '''
        print("TRAINING COMPANIES")
        for x in self.trainingSet:
            print(self.ticker[x])
        print("\n\nTESTING COMPANIES\n\n")
        for x in self.testingSet:
            print(self.ticker[x])
        '''
        self.currentSet = self.trainingSet
        self.indexAt = 0
    
    def getNewDataPoint(self):
        if self.indexAt >= len(self.currentSet):
            self.currentSet = np.random.permutation(self.trainingSet)
            self.indexAt = 0
        index = self.currentSet[self.indexAt]
        toReturn = []
        for i in range (self.numIndustries):
            if i == self.industryDict[self.sector[index]]:
                toReturn.append(1)
            else:
                toReturn.append(0)
        toReturn.append(self.PToE[index])
        toReturn.append(self.divYield[index])
        toReturn.append(self.earningsPerShare[index])
        toReturn.append(self.normalizedRange[index])
        toReturn.append(self.marketCap[index])
        toReturn.append(self.ebitda[index])
        toReturn.append(self.priceToSales[index])
        toReturn.append(self.priceToBook[index])
        target = (100. * (self.nextYearsPrice[index] - self.price[index]) / self.price[index]) - self.SPPerformance
        #print(self.ticker[index])
        #print(self.nextYearsPrice[index])
        #print(self.price[index])
        #print(100. * (self.nextYearsPrice[index] - self.price[index]) / self.price[index])
        #print(toReturn)
        #print(target)
        self.indexAt += 1
        return toReturn, target
        
    def getPLNTDataPoint(self):
        index = -2
        toReturn = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        toReturn.append(self.PToE[index])
        toReturn.append(self.divYield[index])
        toReturn.append(self.earningsPerShare[index])
        toReturn.append(self.normalizedRange[index])
        toReturn.append(self.marketCap[index])
        toReturn.append(self.ebitda[index])
        toReturn.append(self.priceToSales[index])
        toReturn.append(self.priceToBook[index])
        return toReturn
        
    def getNYTDataPoint(self):
        index = -1
        toReturn = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        toReturn.append(self.PToE[index])
        toReturn.append(self.divYield[index])
        toReturn.append(self.earningsPerShare[index])
        toReturn.append(self.normalizedRange[index])
        toReturn.append(self.marketCap[index])
        toReturn.append(self.ebitda[index])
        toReturn.append(self.priceToSales[index])
        toReturn.append(self.priceToBook[index])
        return toReturn
        
    def switchToTest(self):
        self.currentSet = self.testingSet
        self.indexAt = 0


