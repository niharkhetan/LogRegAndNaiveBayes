'''
Created on Oct 4, 2015

@author  : NiharKhetan, Ghanshyam Malu
@desc    : Naive Bayes Modeler
 
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
    
    print "\n\tTraining Model .",
    formatPrintCount = 0
    
    columnarVector = convertVectorToColumnar(vector)    
    probList = []
    
    for eachFeature in columnarVector:
        probList.append(computeConditionalProbability(eachFeature, columnarVector[-1], True))
        if formatPrintCount % 10 == 0:
            print ".",
            
    print "\n"
   
    conditionalProbabilityDict = {}
    
    for eachDict in probList:
        for k,v in eachDict.iteritems():
            conditionalProbabilityDict[k] = v
        formatPrintCount +=1
        
    return conditionalProbabilityDict

def testModel(conditionalProbabilityDict, columnarTrainingVector, columnarTestVector, laplacianCorrection = False):
    '''
    gets all the conditional probabilities generated and computes a list of predicted values
    '''
    print "\n\tTesting Model .",
    formatPrintCount = 0
    
    priorProbTrue = columnarTrainingVector[-1].getDiscreteSet()['1'] / float(columnarTrainingVector[-1].getDiscreteSet()['1'] + columnarTrainingVector[-1].getDiscreteSet()['0']) 
    priorProbFalse = columnarTrainingVector[-1].getDiscreteSet()['0'] / float(columnarTrainingVector[-1].getDiscreteSet()['1'] + columnarTrainingVector[-1].getDiscreteSet()['0'])
    #iterates each row in test data
    predictedResults = []
    for i in range(0, len(columnarTestVector[0].getData())):
        #iterates each classLabel and find likelihood        
        likelihoodList = {} 
        for eachClassLabel in columnarTrainingVector[-1].getDiscreteSet().keys():
            probXGivenY = 1
            if laplacianCorrection:
                priorProbGivenLabel = (columnarTrainingVector[-1].getDiscreteSet()[eachClassLabel] + 1)/ float(len(columnarTrainingVector[-1].getData()) + len(columnarTrainingVector[-1].getDiscreteSet().keys()))
            else:
                priorProbGivenLabel = columnarTrainingVector[-1].getDiscreteSet()[eachClassLabel] / float(len(columnarTrainingVector[-1].getData()))
            
            probXGivenY *= priorProbGivenLabel
            for eachFeature in columnarTestVector[:-1]:
                 
                key = 'P(%s=%s|%s=%s)' % (eachFeature.getName(), eachFeature.getData()[i], columnarTestVector[-1].getName(), eachClassLabel)
                if key in conditionalProbabilityDict.keys():
                    probXGivenY *= conditionalProbabilityDict[key]
                else:                                   
                    # get laplacian correction for the same feature in training set
                    lapCorrectionValue = 0
                    for feature in columnarTrainingVector:
                        if feature.getName() == eachFeature.getName():                            
                            lapCorrectionValue = 1 / float(len(feature.getDiscreteSet().keys()))
                    probXGivenY *= lapCorrectionValue
            likelihoodList[eachClassLabel] = probXGivenY
        
        predictedResults.append(max(likelihoodList.iteritems(), key=operator.itemgetter(1))[0])
        formatPrintCount +=1
        if formatPrintCount % 10 == 0:
            print ".",
            
    print "\n"
    
    return predictedResults

if __name__ == '__main__':
    
    print "=" * 90
    print "\t\t\tWelcome to Naive Bayes Modeler"
    print "=" * 90

    #training_data = "BuyCondodataSet.csv"
    #test_data = "BuyCondodataSetTest.csv"
    training_data = "zoo-train.csv"
    test_data = "zoo-test.csv"
    trainingVector = readFileAsVector(training_data)
    columnarTrainingVector = convertVectorToColumnar(trainingVector)
    testVector = readFileAsVector(test_data)    
    columnarTestVector = convertVectorToColumnar(testVector)
    
    
    condPList = trainModel(trainingVector)
    expectedResults = columnarTestVector[-1].getData()
    predictedResults = testModel(condPList, columnarTrainingVector, columnarTestVector, True)
    
    ############ Evaluation ########### 
        
    #Accuracy
    findAccuracy(predictedResults, expectedResults)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedResults, expectedResults)
    printConfusionMatrix(confusionMatrix)
    
    print "\n\n","-" * 90
    print "\t\tThank you for using the Naive Bayes Modeler"
    print "-" * 90,"\n\n"
    
    