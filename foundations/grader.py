#!/usr/bin/env python

import graderUtil, collections, random

grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Problems 1 and 2

grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf'))


############################################################
# Problem 3a: computeMaxWordLength

grader.addBasicPart('3a-0-basic', lambda :
        grader.requireIsEqual('longest', submission.computeMaxWordLength('which is the longest word')))

grader.addBasicPart('3a-1-basic', lambda : grader.requireIsEqual('sun', submission.computeMaxWordLength('cat sun dog')))
grader.addBasicPart('3a-2-basic', lambda : grader.requireIsEqual('99999', submission.computeMaxWordLength(' '.join(str(x) for x in range(100000)))))

############################################################
# Problem 3b: manhattanDistance

grader.addBasicPart('3b-0-basic', lambda : grader.requireIsEqual(6, submission.manhattanDistance((3, 5), (1, 9))))

def test():
    random.seed(42)
    for _ in range(100):
        x1 = random.randint(0, 10)
        y1 = random.randint(0, 10)
        x2 = random.randint(0, 10)
        y2 = random.randint(0, 10)
        ans2 = submission.manhattanDistance((x1, y1), (x2, y2))
grader.addHiddenPart('3b-1-hidden', test, 2)

############################################################
# Problem 3c: mutateSentences

def test():
    grader.requireIsEqual(sorted(['a a a a a']), sorted(submission.mutateSentences('a a a a a')))
    grader.requireIsEqual(sorted(['the cat']), sorted(submission.mutateSentences('the cat')))
    grader.requireIsEqual(sorted(['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']), sorted(submission.mutateSentences('the cat and the mouse')))
grader.addBasicPart('3c-0-basic', test)

def genSentence(K, L): # K = alphabet size, L = length
    return ' '.join(str(random.randint(0, K)) for _ in range(L))
def test():
    random.seed(42)
    for _ in range(10):
        sentence = genSentence(3, 5)
        ans2 = submission.mutateSentences(sentence)
grader.addHiddenPart('3c-1-hidden', test, 1)

def test():
    random.seed(42)
    for _ in range(10):
        sentence = genSentence(25, 10)
        ans2 = submission.mutateSentences(sentence)
grader.addHiddenPart('3c-2-hidden', test, 2)

############################################################
# Problem 3d: dotProduct

grader.addBasicPart('3d-0-basic', lambda : grader.requireIsEqual(15, submission.sparseVectorDotProduct(collections.defaultdict(float, {'a': 5}), collections.defaultdict(float, {'b': 2, 'a': 3}))))

def randvec():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) - 5
    return v
def test():
    random.seed(42)
    for _ in range(10):
        v1 = randvec()
        v2 = randvec()
        ans2 = submission.sparseVectorDotProduct(v1, v2)
grader.addHiddenPart('3d-1-hidden', test, 2)

############################################################
# Problem 3e: incrementSparseVector

def test():
    v = collections.defaultdict(float, {'a': 5})
    submission.incrementSparseVector(v, 2, collections.defaultdict(float, {'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.defaultdict(float, {'a': 11, 'b': 4}), v)
grader.addBasicPart('3e-0-basic', test)

def test():
    random.seed(42)
    for _ in range(10):
        v1a = randvec()
        v1b = v1a.copy()
        v2 = randvec()
        submission.incrementSparseVector(v1b, 4, v2)
        for key in list(v1b):
          if v1b[key] == 0:
            del v1b[key]
grader.addHiddenPart('3e-1-hidden', test, 2)

############################################################
# Problem 3f: computeMostFrequentWord

def test3f():
    grader.requireIsEqual((set(['the', 'fox']), 2), submission.computeMostFrequentWord('the quick brown fox jumps over the lazy fox'))
grader.addBasicPart('3f-0-basic', test3f)

def test3f(numTokens, numTypes):
    import random
    random.seed(42)
    text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
grader.addHiddenPart('3f-1-hidden', lambda : test3f(1000, 10), 1)
grader.addHiddenPart('3f-2-hidden', lambda : test3f(10000, 100), 1)

############################################################
# Problem 3g: computeLongestPalindrome

def test3g():
    # Test around bases cases
    grader.requireIsEqual(0, submission.computeLongestPalindrome(""))
    grader.requireIsEqual(1, submission.computeLongestPalindrome("a"))
    grader.requireIsEqual(2, submission.computeLongestPalindrome("aa"))
    grader.requireIsEqual(1, submission.computeLongestPalindrome("ab"))
    grader.requireIsEqual(3, submission.computeLongestPalindrome("animal"))
grader.addBasicPart('3g-0-basic', test3g)

def test3g(numChars, length):
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = ' '.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
    ans2 = submission.computeLongestPalindrome(text)
    print text
    print ans2
grader.addHiddenPart('3g-2-hidden', lambda : test3g(2, 10), 1, maxSeconds=1)
grader.addHiddenPart('3g-3-hidden', lambda : test3g(10, 10), 1, maxSeconds=1)
grader.addHiddenPart('3g-4-hidden', lambda : test3g(5, 20), 1, maxSeconds=1)
grader.addHiddenPart('3g-5-hidden', lambda : test3g(5, 400), 1, maxSeconds=2)

grader.grade()
