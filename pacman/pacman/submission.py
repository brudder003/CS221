from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successors
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    # Higher level legal movements
    legalMovesG = gameState.getLegalActions()

    # Return (minimax value of state, optimal action) 
    def recurse(gameState, gameDepth, agentIndex):
      # print'\nlegalMoves: ', legalMoves
      # Collect legal moves and successor states
      legalMoves = gameState.getLegalActions(agentIndex)
      # If the game ends either on win state or lose state or no legal movements, return the final score
      if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
          # print '------------------- Game Ends -------------------'
          return gameState.getScore()

      # Depth equals to 0, call evaluation function 
      elif gameDepth == 0:
          # Choose one of the best actions
          scores = self.evaluationFunction(gameState)
          return scores

      else: 
        # Update depth and index based on current agentIndex
        if agentIndex == (gameState.getNumAgents() - 1):
            newDepth = gameDepth - 1
            newAgentIndex = 0
        # if the agent is pacman, return the maximum score
        else:
            newDepth = gameDepth
            newAgentIndex = agentIndex + 1
        # Recurse 
        choices = [(recurse(gameState.generateSuccessor(agentIndex, action), newDepth, newAgentIndex), action) for action in legalMoves]
        
        # if the agent is not pacman, return the minimum score
        if agentIndex != 0:
            return min(choices)[0]
        # if the agent is pacman, return the maximum score
        else: 
            return max(choices)[0]
      
    globalChoice = [(recurse(gameState.generateSuccessor(self.index, action), self.depth, self.index+1), action) for action in legalMovesG]
    bestScore = max(globalChoice)[0]
    bestIndices = [index for index in range(0, len(globalChoice)) if globalChoice[index][0] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMovesG[chosenIndex]

    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    # Higher level legal movements
    legalMovesG = gameState.getLegalActions()

    # Return (minimax value of state, optimal action) 
    def recurse(gameState, gameDepth, agentIndex, alpha, beta):
      # print'\nlegalMoves: ', legalMoves
      # Collect legal moves and successor states
      legalMoves = gameState.getLegalActions(agentIndex)
      # If the game ends either on win state or lose state or no legal movements, return the final score
      if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
          # print '------------------- Game Ends -------------------'
          return gameState.getScore()

      # Depth equals to 0, call evaluation function 
      elif gameDepth == 0:
          # Choose one of the best actions
          scores = self.evaluationFunction(gameState)
          return scores

      else: 
        # Update depth and index based on current agentIndex
        if agentIndex == (gameState.getNumAgents() - 1):
            newDepth = gameDepth - 1
            newAgentIndex = 0
        # if the agent is pacman, return the maximum score
        else:
            newDepth = gameDepth
            newAgentIndex = agentIndex + 1

        # if the agent is not pacman update beta value
        if agentIndex != 0:
            bar = float("+inf")
            for action in legalMoves:
                choices = recurse(gameState.generateSuccessor(agentIndex, action), newDepth, newAgentIndex, alpha, beta)
                if choices < bar:
                    bar = choices
                beta = min(beta, bar)
                if beta <= alpha:
                    break
            return bar

        # if the agent is pacman, update alpha value
        elif agentIndex == 0:
            bar = float("-inf")
            for action in legalMoves:
                choices = recurse(gameState.generateSuccessor(agentIndex, action), newDepth, newAgentIndex, alpha, beta)
                if choices > bar:
                  bar = choices
                alpha = max(alpha, bar)
                if beta <= alpha:
                    break
            return bar
        # Recurse 

    globalChoice = [(recurse(gameState.generateSuccessor(self.index, action), self.depth, self.index+1, float('-inf'), float('+inf')), action) for action in legalMovesG]
    bestScore = max(globalChoice)[0]
    print '\nglobalChoice: ', globalChoice
    print '\nbestScore: ', bestScore
    bestIndices = [index for index in range(0, len(globalChoice)) if globalChoice[index][0] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    print '\nbestIndices: ', bestIndices
    print '\nchosenIndex: ', chosenIndex
    return legalMovesG[chosenIndex]

    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    # Higher level legal movements
    legalMovesG = gameState.getLegalActions()

    # Return (minimax value of state, optimal action) 
    def recurse(gameState, gameDepth, agentIndex):
      # print'\nlegalMoves: ', legalMoves
      # Collect legal moves and successor states
      legalMoves = gameState.getLegalActions(agentIndex)
      # If the game ends either on win state or lose state or no legal movements, return the final score
      if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
          # print '------------------- Game Ends -------------------'
          return gameState.getScore()

      # Depth equals to 0, call evaluation function 
      elif gameDepth == 0:
          # Choose one of the best actions
          scores = self.evaluationFunction(gameState)
          return scores

      else: 
        # Update depth and index based on current agentIndex
        if agentIndex == (gameState.getNumAgents() - 1):
            newDepth = gameDepth - 1
            newAgentIndex = 0
        # if the agent is pacman, return the maximum score
        else:
            newDepth = gameDepth
            newAgentIndex = agentIndex + 1
        # Recurse 
        choices = [(recurse(gameState.generateSuccessor(agentIndex, action), newDepth, newAgentIndex), action) for action in legalMoves]
        
        # if the agent is not pacman, return random action and average score
        if agentIndex != 0:
            return random.choice(choices)[0]
        # if the agent is pacman, return the maximum score
        else: 
            return max(choices)[0]
      
    globalChoice = [(recurse(gameState.generateSuccessor(self.index, action), self.depth, self.index+1), action) for action in legalMovesG]
    bestScore = max(globalChoice)[0]
    bestIndices = [index for index in range(0, len(globalChoice)) if globalChoice[index][0] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMovesG[chosenIndex]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
