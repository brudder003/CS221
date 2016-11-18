#!/usr/bin/python

import random
import collections
import math
import sys
import util
from collections import Counter
from util import *
from numpy import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    words = x.split(' ')
    wordFeatures = collections.Counter(words)
    return wordFeatures
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    n = 2; 
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    for j in range(numIters):
        # predictor for train loss and dev loss
        print 'Train loss: ', evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print 'Dev loss: ', evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        for x, y in trainExamples:
            feature = featureExtractor(x)
            # if hinge loss less than 1.0 updates w
            if dotProduct(weights, feature)*y <= 1.0:
                # schoastic gradient decent
                for singleElementX in feature:
                    if singleElementX not in weights:
                        weights[singleElementX] = 0
                    weights[singleElementX] += eta*feature[singleElementX]*y
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        phi = {}
        for keys in weights.keys():
            phi[keys] = random.random() * 10.00
        y = 1 if dotProduct(phi,weights) >= 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        xWithoutSpace = ''.join(x.split(' '))
        splitLetters = [xWithoutSpace[i:i+n] for i in range(len(xWithoutSpace)-n+1)]
        splitLettersCounter = {}
        for i in range(len(splitLetters)):
            splitLettersCounter[splitLetters[i]] = 1 if splitLetters[i] not in splitLettersCounter else +1
        return splitLettersCounter
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    def distance(example, mu, a, b):
        #print 'example, mu:', example, mu
        distance = dataSquare[a] - 2*dotProduct(example, mu) + muSquare[b]
        return distance

    # cluster centriods
    mu = [examples[random.randint(0, len(examples) - 1)] for k in range(K)]
    print 'K: ', K
    print 'mu: ', mu

    # cluster assignments 
    z = [None] * len(examples)

    # precompute on certain values to make the code running more efficient
    dataSquare = {}
    muSquare = {}
    index = 0
    for _ in examples:
        dataSquare[index] = dotProduct(_ , _)
        index += 1
    index = 0
    for _ in mu:
        muSquare[index] = dotProduct(_ , _)
        index += 1

    for iter in range(maxIters):
        # step 1: set z
        totalCost = 0
        for i in range(len(examples)):
            cost, z[i] = min([(distance(examples[i], mu[k], i, k), k) for k in range(K)])
            totalCost += cost
            # print 'totalCost = %s, mu = %s' % (totalCost, mu)

        # step 2: set mu
        for k in range(K):
            l = 0
            myPoints = collections.Counter()
            for i in range(len(examples)):
                if z[i] == k:
                    l += 1
                    for x in list(examples[i]):
                        myPoints[x] = (myPoints[x] + examples[i][x])
            if len(myPoints) > 0: 
                averagemuSet = {}
                for key in myPoints:
                    averagemuSet[key] = myPoints[key] / l
                mu[k] = averagemuSet[key]

    return mu, z, totalCost


    # END_YOUR_CODE

print extractWordFeatures('I am what I am')
