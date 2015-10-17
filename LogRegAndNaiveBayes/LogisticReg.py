'''
Created on Oct 4, 2015

@author: NiharKhetan
'''
from DataParser.ReadCSV import readFileAsVector
from Bean.Feature import Feature
from Evaluation.Metrics import *
import operator
import math


def findSigmoidWX(w,x):
    wx = 0   
    for i in range(0,len(w)):        
        wx += int(w[i])*int(x[i])
    wx *= -1
    #print wx
    e = math.exp(wx)
    sigmoid = 1 / float(1 + e)
    return sigmoid
          

def trainModel(vector):
    '''
    Generates a list of conditional probabilities for given setof features against the class label
    '''
    #columnarVector = convertVectorToColumnar(vector)    
    #probList = []
    w = []
    g = []
    #initializing w weight vector
    #initializing gradient vector
    for i in range(0, len(vector[0]) - 1):
        w.append(0)
    
    for eachPoint in vector:
        eachPoint.insert(0,0)
    #print vector
    #print w
    wItalic = 0
    error = 1000
    while(error != 0.0001):
        #initializing gradient vector
        for i in range(0, len(vector[0]) - 1):
            g.append(0)
        
    
        for eachPoint in vector[1:]:
            #print eachPoint
            pOfi = findSigmoidWX(w, eachPoint[:-1])
            #print pOfi
            error = int(eachPoint[-1]) - pOfi
           
            for j in range(0, len(eachPoint) - 1):
                g[j] = g[j] + error*int(eachPoint[j])
            #print g
            
            for k in range(0,len(w)):
                w[k] += 0.001*g[k]
        print error
        print w        
        


if __name__ == '__main__':
    #training_data = "BuyCondodataSet.csv"
    #test_data = "BuyCondodataSetTest.csv"
    training_data = "zoo-train.csv"
    test_data = "zoo-test.csv"
    trainingVector = readFileAsVector(training_data)
    trainModel(trainingVector)
    
    '''
    columnarTrainingVector = convertVectorToColumnar(trainingVector)
    testVector = readFileAsVector(test_data)    
    columnarTestVector = convertVectorToColumnar(testVector)
    
    
    condPList = trainModel(trainingVector)
    expectedResults = columnarTestVector[-1].getData()
    predictedResults = testModel(condPList, columnarTrainingVector, columnarTestVector, True)
    
    # Evaluation 
        
    #Accuracy
    findAccuracy(predictedResults, expectedResults)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedResults, expectedResults)
    printConfusionMatrix(confusionMatrix)
    '''
    
    
    