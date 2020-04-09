# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.probXBlackGivenY = util.Counter()
        self.probXWhiteGivenY = util.Counter()
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 10  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        # probabilities p(y)
        # p(y=0) = number of no face/ total labels
        # p(y=1) = number of face/ total labels
        self.prob = {}
        counts = util.Counter()
        for label in trainingLabels:
            counts[label] += 1
        self.prob = util.normalize(counts)
        # black_feature_label : number of times a feature is black pixel in all images
        # (training label for image x, feature) : #times black
        # white_feature_label : number of times a feature is white pixel in all images
        # (training label for image x, feature) : #times white
        # count_features_labels : number of times a feature occurred

        black_feature_label = util.Counter()
        white_feature_label = util.Counter()
        count_features_labels = util.Counter()
        index = 0
        for image in trainingData:
            for feature, color in image.items():
                    count_features_labels[(feature, trainingLabels[index])] += 1
                    if color == 1:
                        black_feature_label[(feature, trainingLabels[index])] += 1
                    else:
                        white_feature_label[(feature, trainingLabels[index])] += 1
            index += 1

        # smoothing
        for label in self.legalLabels:
            for feature in self.features:
                black_feature_label[(feature, label)] += self.k

        # conditional probability p(x|y=true), p(x|y=false) x= count for a feature when it's a black pixels
        for feature_label, count in black_feature_label.items():
            self.probXBlackGivenY[feature_label] = float(count) / float(count_features_labels[feature_label])

        # conditional probability p(x|y=true), p(x|y=false) x= count for a feature when it's a white pixels
        for feature_label, count in white_feature_label.items():
            self.probXWhiteGivenY[feature_label] = float(count) / float(count_features_labels[feature_label])

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):

        "*** YOUR CODE HERE ***"

        log_joint_prob = util.Counter()
        for label in self.legalLabels:
            log_joint_prob[label] = math.log(self.prob[label])
            for feature, value in datum.items():
                if value == 1:
                    probability = self.probXBlackGivenY[feature, label]
                elif self.probXWhiteGivenY[feature, label]>0:
                    probability = self.probXWhiteGivenY[feature, label]
                else:
                    probability = 1
                log_joint_prob[label] += math.log(probability)
        return log_joint_prob
