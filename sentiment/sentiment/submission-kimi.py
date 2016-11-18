#!/usr/bin/python
import random
import collections
import math
import sys
from collections import Counter
from util import *
import numpy as np
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
    # split out words
    wordList = x.split()
    # create empty dictionary
    wordDict = dict()
    # loop through eachword in wordlist
    for eachWord in wordList:
        # if wordDict does not have this word, initialize its value to 1
        if eachWord not in wordDict:
            wordDict[eachWord] = 1
        # else increment by 1
        else:
            wordDict[eachWord] += 1
    # return the final dictionary
    return wordDict
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
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    for i in range(numIters):
        for x,y in trainExamples:
            features = featureExtractor(x)
            hinge = 1.00 - dotProduct(features,weights) * y
            if hinge > 0:
                for eachx in features:
                    hingeGradient = - features[eachx] * y
                    if eachx not in weights:
                        weights[eachx] = 0
                    weights[eachx] -= eta * hingeGradient
    # END_YOUR_CODE
    # print weights
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
        # raise Exception("Not implemented yet")
        phi = {}
        y = {}
        for eachWord in weights:
            phi[eachWord] = random.randint(1,10)
        if (dotProduct(weights,phi) > 0):
            y = 1
        else:
            y = -1
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
        # raise Exception("Not implemented yet")
        features = {}
        wordsList = x.split()
        wordsCombine = ''.join(wordsList)
        for i in range(0,(len(wordsCombine)-n+1)):
            charactes = wordsCombine[i:(i+n)]
            if (charactes not in features):
                features[charactes] = 1
            else:
                features[charactes] += 1
        return features
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
    # sparse vector minus
    def sparseMinus(v1,v2):
        k = {}
        print v1
        print v2
        for i in list(v1):
            k[i] = v1[i] - v2[i]
        return k

    # create random mu centroid and z assignemnt
    n = len(examples)
    mu = []
    totalCost = 0
    for k in range(K):
        mu.append(examples[random.randint(0,n-1)])
    # Cluster assignments
    z = [None] * n

    for t in range(maxIters):
        # Step 1: set z
        totalCost = 0
        for i in range(n):
            cost, z[i] = min([(dotProduct(sparseMinus(examples[i],mu[k]),sparseMinus(examples[i],mu[k])), k) for k in range(K)])
            totalCost += cost
        # print 'totalCost = %smu = %s' %(totalCost, mu)

        # Step 2: set mu
        for k in range(K):
            myExamples = [examples[i] for i in range(n) if z[i] == k]
            if len(myExamples) > 0:
                mu[k] = sum(myExamples) / len(myExamples)
    return mu, z, totalCost



    # END_YOUR_CODE
