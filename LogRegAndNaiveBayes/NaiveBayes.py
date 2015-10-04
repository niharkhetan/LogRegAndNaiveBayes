'''
Created on Oct 4, 2015

@author: NiharKhetan
'''
from DataParser.ReadCSV import readFileAsVector
from Bean.Feature import Feature
from Evaluation.Metrics import *
import operator

def convertVectorToColumnar(vector):
    '''
    This returns the whole data as a list of features objects
    currently data is in form [[],[],[]]
    It converts it to [f0,f1,f2] where each fi is and object
    @param vector: type(list) vector which is the whole dataset
    @return columns: type(list) which is the whole data stored as columns
    '''
    columns = []
    columnLabels = vector[0]
    for column in range(0, len(columnLabels)):
        columnData = []
        for row in vector[1:]:
            columnData.append(row[column])
        columns.append(Feature(columnLabels[column], columnData))
    return columns

def computeConditionalProbability(feature, classLabel, laplacianCorrection = False):
    '''
    To compute conditional probabilities for a feature given class label
    '''
    probDict = {}
    
    classLabelData = classLabel.getData()
    featureData = feature.getData()
    
    #compute P(y|x=0), P(y|x=1)... P(y|x=n)
    for eachClassLabelType in classLabel.getDiscreteSet().keys():        
        for eachType in feature.getDiscreteSet().keys():
            # this big enumerate on right gets me all the indexes of eachType in my feature data
            count = 0
            for eachIndex in [i for i, x in enumerate(featureData) if x == eachType]:
                if classLabelData[eachIndex] == eachClassLabelType:
                    count += 1
            key = 'P(%s=%s|%s=%s)' % (feature.getName(), eachType, classLabel.getName(), eachClassLabelType)
            if laplacianCorrection:
                probDict[key] = (count + 1)/ float((classLabel.getDiscreteSet()[eachClassLabelType] + len(feature.getDiscreteSet().keys())))
            else:
                probDict[key] = count / float(classLabel.getDiscreteSet()[eachClassLabelType])
    
    return probDict 
        
def trainModel(vector):
    '''
    Generates a list of conditional probabilities for given setof features against the class label
    '''
    columnarVector = convertVectorToColumnar(vector)
    priorProbTrue = columnarVector[-1].getDiscreteSet()['1'] / float(columnarVector[-1].getDiscreteSet()['1'] + columnarVector[-1].getDiscreteSet()['0']) 
    priorProbFalse = columnarVector[-1].getDiscreteSet()['0'] / float(columnarVector[-1].getDiscreteSet()['1'] + columnarVector[-1].getDiscreteSet()['0'])
    probList = []
    
    for eachFeature in columnarVector:
        probList.append(computeConditionalProbability(eachFeature, columnarVector[-1], True))
   
    conditionalProbabilityDict = {}
    
    for eachDict in probList:
        for k,v in eachDict.iteritems():
            conditionalProbabilityDict[k] = v
    
    return conditionalProbabilityDict

def testModel(conditionalProbabilityDict, columnarTrainingVector, columnarTestVector):
    '''
    gets all the conditional probabilities generated and computes a list of predicted values
    '''
 
    #iterates each row in test data
    predictedResults = []
    for i in range(0, len(columnarTestVector[0].getData())):
        #iterates each classLabel and find likelihood        
        likelihoodList = {} 
        for eachClassLabel in columnarTrainingVector[-1].getDiscreteSet().keys():
            probXGivenY = 1
            
            for eachFeature in columnarTestVector[:-1]:
                key = 'P(%s=%s|%s=%s)' % (eachFeature.getName(), eachFeature.getData()[i], columnarTestVector[-1].getName(), eachClassLabel)
                if key in conditionalProbabilityDict.keys():
                    probXGivenY *= conditionalProbabilityDict[key]
                else:
                    # TODO I DUNNO IF DOING THIS IS CORRECT
                    probXGivenY *= 0
            likelihoodList[eachClassLabel] = probXGivenY
        print likelihoodList
        predictedResults.append(max(likelihoodList.iteritems(), key=operator.itemgetter(1))[0])
    
    return predictedResults
                
                
  
            

if __name__ == '__main__':
    training_data = "BuyCondodataSet.csv"
    test_data = "BuyCondodataSetTest.csv"
    training_data = "zoo-train.csv"
    test_data = "zoo-test.csv"
    trainingVector = readFileAsVector(training_data)
    columnarTrainingVector = convertVectorToColumnar(trainingVector)
    testVector = readFileAsVector(test_data)    
    columnarTestVector = convertVectorToColumnar(testVector)
    
    
    condPList = trainModel(trainingVector)
    expectedResults = columnarTestVector[-1].getData()
    predictedResults = testModel(condPList, columnarTrainingVector, columnarTestVector)
    
    # Evaluation 
        
    #Accuracy
    findAccuracy(predictedResults, expectedResults)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedResults, expectedResults)
    printConfusionMatrix(confusionMatrix)
    
    
    