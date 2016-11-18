import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(sorted(text.split(' '), key=str.lower, reverse=True), key=len)
    # END_YOUR_CODE

############################################################
# Problem 3b

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    High-level idea: generate sentences similar to a given sentence.
    Given a sentence (sequence of words), return a list of all possible
    alternative sentences of the same length, where each pair of adjacent words
    also occurs in the original sentence. (The words within each pair should appear 
    in the same order in the output sentence as they did in the orignal sentence.)
    Notes:
    - The order of the sentences you output doesn't matter.
    - You must not output duplicates.
    - Your generated sentence can use a word in the original sentence more than
      once.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    #print "Test - 3c"
    # Split the sentence into words
    words = sentence.split()
    mutateSentences = []
    wordPair = {}
    # only one possible sentence combination exist if the number of words in the sentence is less than three
    if len(words) < 3:
        mutateSentences.append(sentence)
        return mutateSentences
    # if the length of sentence is more than three words. then permutation exists.
    else:
    # create set of each pair of adjacent words
        for i in range(0,len(words)-1):
            word = words[i]
            if word not in wordPair:
                wordPair[word] =  set()
            wordPair[word].add(words[i+1])
            if words[-1] not in wordPair:
                wordPair[words[-1]] = set()

    # now generate sentences similar to a give sentence.
    mutateSentences = [[word] for word in wordPair]
    for i in range(1, len(words)):
        newSentences = []
        for w in mutateSentences:
            lastSet = w[-1]
            if lastSet in wordPair:
                wordPairCombination = []
                for nextSet in wordPair[lastSet]:
                    wordPairCombination = w + [nextSet]
                    newSentences.append(wordPairCombination)

        mutateSentences = newSentences
    
    mutateSentences = [ ' '.join(w) for w in mutateSentences ]
    return mutateSentences
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    DotProduct = 0
    for i in list(v1):
        DotProduct += v1[i] * v2[i]
    return DotProduct
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for i in list(v2):
        v1[i] += scale*v2[i]
    return v1
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
    their count, i.e.
    (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(float).
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    words = text.split()
    # Counts the number of apperance of a certain words, return empty if its zero
    countWords = collections.Counter(words)
    # return the the set of words that occur the maximum number of times, and their count
    mostCommonWords = set(word[0] for word in countWords.items() if word[1] == countWords.most_common(1)[0][1])
    return (mostCommonWords, countWords.most_common(1)[0][1])
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    cache = {}
    letterCount = collections.Counter(text)
    length = 0 
    def determineLength(text):
        if text in cache:
            return cache[text]
        if len(text) < 3:
            if len(text) == 0: 
                length = 0
            elif len(text) == 1:
                length = 1
            elif len(text) == 2:
                for word in text:
                    if letterCount[word] == 2:
                        length =  2
                    else:
                        length =  1
        elif text[0] == text[-1]:
            length = 2 + determineLength(text[1:-1])
        else:
            removeBack = determineLength(text[0:-1])
            removeFront = determineLength(text[1:len(text)])
            length = max(removeBack,removeFront)
        cache[text] = length
        return length

    return determineLength(text)
    # END_YOUR_CODE