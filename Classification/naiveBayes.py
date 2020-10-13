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
import numpy as np


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
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
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
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

    distinct_label = self.legalLabels
    label_total_num = len(trainingLabels)
    label_distinct_num = len(distinct_label)

    feature_len = len(self.features)
    trainingData = np.asarray(trainingData)
    
    label_freq = util.Counter()
    for label in distinct_label:
        label_freq[label] = (trainingLabels.count(label))
    self.label_freq = label_freq
    self.label_ratio = label_freq.copy()
    self.label_ratio.divideAll(label_total_num)


    feature_count = util.Counter()
    for index, label in enumerate(trainingLabels):
        if(feature_count[label] == 0):
            feature_count[label] = trainingData[index]
        else:
            feature_count[label] += trainingData[index]

    self.feature_count = feature_count


    max_val = 0
    best_k = 0

    self.feature_probs = util.Counter()
    density = 1
    for k in kgrid:
        for label in self.legalLabels:
            freq = label_freq[label]
            try:
                feature_probs = feature_count[label].copy()
            except:
                print("Training data is not representative")
            feature_probs.incrementAll(self.features, k)
            feature_probs.divideAll(freq + 2 * k)
            self.feature_probs[label] = feature_probs.copy()
            
        self.k = k
        pred_label = self.classify(validationData)
        accuracy_list = [int(p == d) for p, d in zip(pred_label, validationLabels)]
        accuracy = accuracy_list.count(1) / (len(accuracy_list) * 1.0)
        if(max_val < accuracy):
            max_val = accuracy
            best_k = k

    for label in distinct_label:
        freq = label_freq[label]
        feature_probs = feature_count[label].copy()
        feature_probs.incrementAll(self.features,  best_k)
        feature_probs.divideAll(freq  +  2 * best_k) 
        self.feature_probs[label] = feature_probs.copy()


    print("Best k: ", best_k)
    self.k = best_k

        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    distinct_feature = self.features
    distinct_label = self.legalLabels

    for i in distinct_label:
        logJoint[i] = math.log(self.label_ratio[i])
        for j in distinct_feature:
            if(datum[j] != 0):
                logJoint[i] += math.log(self.feature_probs[i][j])
             
            else:
                logJoint[i] += math.log((self.label_freq[i] - self.feature_count[i][j] + self.k) / (self.label_freq[i] + 2 * self.k) )
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
    total_featuresOdds = util.Counter()
    feature_label1 = self.feature_probs[label1]
    feature_label2 = self.feature_probs[label2]

    for i in self.features:
        if(feature_label1[i] != 0 and feature_label2[i] == 0):
            total_featuresOdds[i] = sys.float_info.max
            continue
        total_featuresOdds[i] = (feature_label1[i] / feature_label2[i])
    total_featuresOdds.sortedKeys()

    for i in range(100):
        featuresOdds.append(list(total_featuresOdds.items())[i][0])
    print(featuresOdds)
    return featuresOdds
    

    
      
