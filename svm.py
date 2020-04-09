# svm.py
# -------------

# svm implementation
import util
import samples
import numpy as np
from sklearn.svm import SVC
PRINT = True

class SVMClassifier:
  """
  svm classifier
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    print ("Starting iteration...")#global iteration not defined, iteration, "..."
    tData = []
    vData = []

    # Converting the training and validation data to the input format
    for i in range(len(trainingData)):
        sample = trainingData[i].items()
        if i < len(validationData):
            validation = validationData[i].items()
        temp1 = []
        temp2 = []
        for j in range(len(sample)):
            # Only appending the feature at that point
            # Number of features should be the same for all training samples
            temp1.append(sample[j][1])
            if i < len(validationData):
                temp2.append(validation[j][1])
        sample = np.asarray(temp1)
        sample = sample.flatten()
        tData.append(sample)
        if i < len(validationData):
            validation = np.asarray(temp2)
            validation = validation.flatten()
            vData.append(validation)
    tData = np.asarray(tData)
    vData = np.asarray(vData)
    tLabels = np.asarray(trainingLabels)
    vLabels = np.asarray(validationLabels)

    # Creating a SVM model
    clf = SVC(decision_function_shape='ocr', kernel='linear')
    # Fitting the training data
    clf.fit(tData, tLabels)
    # Cross validation (unhighlight below two lines)
    # score = clf.score(vData, vLabels)
    # print "Cross validation score = %.4f" % (score)
    # Save the model
    self.model = clf

  def classify(self, data ):
    guesses = []
    clf = self.model
    processedData = []

    # Processing data into correct data input
    for datum in data:
        datum = datum.items()
        temp1 = []
        for j in range(len(datum)):
            # Only appending the feature at that point
            # Number of features should be the same for all training samples
            temp1.append(datum[j][1])
        sample = np.asarray(temp1)
        sample = sample.flatten()
        processedData.append(sample)

    # Predicting and reformating
    processedData = np.asarray(processedData)
    # print(processedData.shape)
    guesses = clf.predict(processedData)
    guesses = np.array(guesses)
    return guesses
