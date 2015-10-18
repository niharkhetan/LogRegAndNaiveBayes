'''
Created on Oct 4, 2015

@author: NiharKhetan
'''
from DataParser.ReadCSV import readFileAsVector
from Bean.Feature import Feature
from Evaluation.Metrics import *
import operator
import math
from copy import deepcopy


def findSigmoidWX(w,x):
    wx = 0   

    for i in range(len(w)):        
        wx += float(w[i])*float(x[i])
    #Using Try to avoid overflow of range; When e to power gets very large it is equivalent to infinity
    try :
        e = math.exp(-wx)
    except :
        e = 0
    sigmoid = 1 / float(1 + e)
    return sigmoid
          
def findNorm(vector):
    norm = 0
    for point in vector:
        norm += point*point
    
    return math.sqrt(norm)
    

def trainModel(vector):
    '''
    Generates a list of conditional probabilities for given setof features against the class label
    '''

    print "\n\tTraining Model .",
    # Initializing parameters
    eta = 0.001
    prec = 5
    convergenceThreshold = 0.001
    formatPrintCount = 0

    # Initializing w weight vector to 0 for every feature
    w = [0]*(len(vector[0])-1)
    
    # Transformations for Training examples and Weight vector
    
    # Transforming the training examples to canonical representation <1,x1,x2,..,xN>
    for eachPoint in vector:
        eachPoint.insert(0,1)
    
    # Adding 0 to weight vector at index 0
    w.insert(0,0)
    
    # Repeat until convergence
    while True:
        
        # Save old Weight vector to check for convergence
        wOld = deepcopy(w)

        #initializing gradient vector    
        g = [0]*(len(vector[0])-1)
            
        for eachPoint in vector[1:]:
            #pi = 1 /( 1 * exp[ w.xi])
            pOfi = findSigmoidWX(w, eachPoint[:-1])
            
            #error=yi - pi
            error = int(eachPoint[-1]) - pOfi
           
            for j in range(0, len(eachPoint) - 1):
                g[j] = g[j] + error*int(eachPoint[j])
            
            for k in range(0,len(w)):
                w[k] += eta*g[k]
                
        wChange = round(findNorm(w) - findNorm(wOld),prec)
        
        if wChange <= convergenceThreshold:
            break
        
        formatPrintCount +=1
        if formatPrintCount % 100 == 0:
            print ".",
            
    return w

def predictLabelBasedOnThreshold(dataPoint, weightVec):
    
    thresholdCheck = 0
    
    for i in range(len(dataPoint)):
        thresholdCheck += float(dataPoint[i]) * float(weightVec[i]) 
    
    if thresholdCheck < 0.5:
        return 0
    else:
        return 1
        
def testModel(vector, weightVec):
    
    print "\n\n\tTesting Model .",
    formatPrintCount = 0
    
    predictedLabels = []
    for dataPoint in vector[1:]:    
        # Transforming the test datapoints to canonical representation <1,x1,x2,..,xN>
        dataPoint.insert(0,1)
        predictedLabels.append(predictLabelBasedOnThreshold(dataPoint[:-1], weightVec))
        
        formatPrintCount +=1
        if formatPrintCount % 10 == 0:
            print ".",
            
    print "\n"
    return predictedLabels

if __name__ == '__main__':
    
    print "=" * 90
    print "\t\t\tWelcome to Logistic Regression Modeler"
    print "=" * 90


    training_data = "zoo-train.csv"
    test_data = "zoo-test.csv"
    
    trainingVector = readFileAsVector(training_data)
    testVector = readFileAsVector(test_data)    

    weightVec = trainModel(trainingVector)
    
    expectedResults = []
    
    for dataPoint in testVector[1:]:
        expectedResults.append(int(dataPoint[-1]))
        
    predictedResults = testModel(testVector,weightVec)
    
    
    # Evaluation 
        
    #Accuracy
    findAccuracy(predictedResults, expectedResults)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedResults, expectedResults)
    printConfusionMatrix(confusionMatrix)

    print "\n\n","-" * 90
    print "\t\tThank you for using the Logistic Regression Modeler"
    print "-" * 90,"\n\n"
    