import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return ['move']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return [(0, 0.45, 15), (1, 0.35, 5), (2, 0.20, 0)]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")

        # state is defined as triples: 
        # (totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts)
        totalCardValueInHand = state[0]
        nextCardIndexIfPeeked = state[1]
        deckCardCounts = state[2]

        # initialize the return parameter 
        result = []

        # check if there still have cards left in the deck, if not return result
        if deckCardCounts == None:
            return result

        # if the action is Take
        elif action == 'Take':
            # if the last action is peek 
            if nextCardIndexIfPeeked != None:
                totalCardValueInHand += self.cardValues[nextCardIndexIfPeeked]
                # check if the game is busted
                # if its busted
                if totalCardValueInHand > self.threshold:
                    newState = (totalCardValueInHand, None, None)
                    reward = 0
                # if not busted
                else:
                    deck = list(deckCardCounts)
                    deck[nextCardIndexIfPeeked] -= 1
                    deck = tuple(deck)
                    # run out of cards
                    if sum(deck) == 0:
                        newState = (totalCardValueInHand, None, None)
                        reward = totalCardValueInHand
                    # still have cards left in the deck
                    else:
                        newState = (totalCardValueInHand, None, deck)
                        reward = 0
                result.append((newState,1,reward))
                return result

            # if the last action is not peek
            else:
                results = []
                rewards = []
                totalCards = 0
                for v in range(len(self.cardValues)):
                    if deckCardCounts[v] != 0:
                        totalCardValueInHand += self.cardValues[v]
                        reward = 0
                        totalCards += 1
                        # if not busted 
                        if totalCardValueInHand <= self.threshold:
                            deck = list(deckCardCounts)
                            deck[v] -= 1
                            deck = tuple(deck)
                            # run out of cards
                            if sum(deck) == 0:
                                newState = (totalCardValueInHand, None, None)
                                reward = totalCardValueInHand
                            else:
                                newState = (totalCardValueInHand, nextCardIndexIfPeeked, deck)
                                reward = 0
                        # if busted
                        else:
                            newState = (totalCardValueInHand, None, None)
                            reward = 0
                        results.append(newState)
                        rewards.append(reward)
                        # reset the totalCardValueInHand value
                        totalCardValueInHand = state[0]
                return ([ (r, 1/float(totalCards),rewards[i]) for i,r in enumerate(results)])


        # if the action is Quit
        elif action == 'Quit':
            if ( totalCardValueInHand > self.threshold ):
                return [((totalCardValueInHand, None, None), 1, 0)]
            else:
                return [((totalCardValueInHand, None, None), 1, totalCardValueInHand)]

        # if the action is Peek
        elif action == 'Peek':
            if nextCardIndexIfPeeked != None:
                results = []
            else:
                results = []
                for v in range(len(self.cardValues)):
                    if deckCardCounts[v] > 0:
                        results.append((totalCardValueInHand,v,deckCardCounts))
            return [ (r, 1/float(len(results)), -self.peekCost) for r in results]
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    return BlackjackMDP(cardValues = [16, 5, 4], multiplicity = 3, threshold = 20, peekCost = 1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # get the value for Vopt
        if newState == None:
            Vopt = 0
        else:
            Vopt = max(self.getQ(newState, newActions) for newActions in self.actions(newState))

        # have the predication Q-values
        qOptHat = self.getQ(state,action)

        # update the weight
        for index, feature in self.featureExtractor(state,action):
            self.weights[index] -= self.getStepSize()*(qOptHat - (reward + self.discount*Vopt))*feature

        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
# smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
# valueIter = ValueIteration()
# valueIter.solve(smallMDP, .001)

# learning
# learning algorithm
# qLearning = QLearningAlgorithm(smallMDP.actions, smallMDP.discount(), identityFeatureExtractor, 0.2)
# util.simulate(smallMDP, qLearning, numTrials=30000, maxIterations=1000, verbose=False, sort=False)

# policy from Q-learning
# print '\nPolicy from smallMDP:'
# qLearning.explorationProb = 0.0
# util.simulate(smallMDP, qLearning, numTrials=5, maxIterations=10, verbose=True, sort=True)

# value iteration method
# for states in valueIter.pi:
#    print 'The smallMDP state:  %s action: %s' % (states, valueIter.pi[states])
########################

# Large test case
# largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
# largeMDP.computeStates()
# valueIter = ValueIteration()
# valueIter.solve(largeMDP, .001)

# learning
# learning algorithm
# qLearning = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), identityFeatureExtractor, 0.2)
# util.simulate(largeMDP, qLearning, numTrials=30000, maxIterations=1000, verbose=False, sort=False)

# policy from Q-learning
# print '\nPolicy from largeMDP:'
# qLearning.explorationProb = 0.0
# util.simulate(largeMDP, qLearning, numTrials=5, maxIterations=10, verbose=True, sort=True)


# value iteration method
# for states in valueIter.pi: 
#    print 'The largeMDP state:  %s action: %s' % (states, valueIter.pi[states]) 



############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    feature = []
    # beiginning 
    feature.append(((total,action),1))
    if counts != None:
        # second
        feature.append(((tuple(int(bool(item)) for item in counts), action), 1))
        # rest 
        for i,j in enumerate(counts):
            feature.append(((('card'+str(i),j,counts[i]), action), 1))
    return feature

    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
# originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
# solve using value iteration
# valueIter = ValueIteration()
# valueIter.solve(originalMDP, .001)
# fixed reinforcement learning
# fixed = util.FixedRLAlgorithm(valueIter.pi)
# determine reward
# originalReward = util.simulate(originalMDP, fixed, numTrials=30000, maxIterations=5, verbose=False, sort=False)
# print '\nAverage utility for originalMDP is: ', float(sum(originalReward))/float(len(originalReward))

# New threshold
# newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
# determine reward
# newThresholdReward = util.simulate(newThresholdMDP, fixed, numTrials=30000, maxIterations=5, verbose=False, sort=False)
# print '\nAverage utility for newThresholdMDP is: ', float(sum(newThresholdReward))/float(len(newThresholdReward))

# Q-learning algorithm
# qLearning = QLearningAlgorithm(newThresholdMDP.actions, newThresholdMDP.discount(), identityFeatureExtractor, 0.2)
# qLearning.explorationProb = 0.0
# qLearningReward = util.simulate(newThresholdMDP, qLearning, numTrials=5, maxIterations=10, verbose=True, sort=True)
# print '\nAverage utility for qLearningReward is: ', float(sum(qLearningReward))/float(len(qLearningReward))