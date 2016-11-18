import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return self.query
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return state == ''
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # Return list of (action, newState, cost)
        results = []
        for index in range(len(state)+1):
            results.append((state[0:index], state[index:], self.unigramCost(state[0:index])))
        return results
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    if len(ucs.actions) > 0:
        #print 'Problem 1-b: ', ' '.join(ucs.actions)
        return ' '.join(ucs.actions) 

    else:
        return ''
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # next word index, and current word
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        results = []
        # generate all possible words
        nextFilledWords = set()
        nextFilledWords = self.possibleFills(self.queryWords[state[0]])
        if len(nextFilledWords) == 0:
            nextFilledWords.add(self.queryWords[state[0]])

        for eachFilledWord in nextFilledWords:
            if state[0] < len(self.queryWords):
                newState = state[0] + 1, eachFilledWord
                results.append((eachFilledWord, newState, self.bigramCost(state[1], eachFilledWord)))

        return results 
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    if len(ucs.actions) > 0:  
        #print 'Problem 2-b: ', ' '.join(ucs.actions)
        return ' '.join(ucs.actions)
    else:
        return ''

    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return state[0] == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        results = []
        firstWord = state[1]
        #print 'firstWord:', firstWord
        nextFilledWords = set()
        for index in range(state[0], len(self.query) + 1):
            secondWord = self.query[state[0]:index]
            nextFilledWords = self.possibleFills(secondWord)
            #print 'nextFilledWords: ', nextFilledWords
            for eachWord in nextFilledWords:
                results.append((eachWord, (index, eachWord), self.bigramCost(firstWord, eachWord)))

        return results

        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    if len(ucs.actions) > 0:  
        #print 'Problem 3-b: ', ' '.join(ucs.actions)
        return ' '.join(ucs.actions)
    else:
        return ''
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
