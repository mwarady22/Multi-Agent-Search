# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print("successorGameState", successorGameState)
        # print("newPos", newPos)
        # print("newFood", newFood)
        # print("ghostpos")
        # for i in range(0, len(newGhostStates)):
        #     print(newGhostStates[i].configuration.getPosition())
        # print("newScaredTimes", newScaredTimes)
        "*** YOUR CODE HERE ***"
        # print(newPos)

        ret = 0
        listfood = list(newFood)
        lenlistfood = len(listfood)
        otherdist = len(listfood[0])
        mindist = lenlistfood * otherdist


        for i in range(0, len(newGhostStates)):
            disttoghost = manhattanDistance(newPos, newGhostStates[i].configuration.getPosition())
            if disttoghost < 2:
                ret += -1000
                return ret
            elif disttoghost < 3:
                ret += -100
            elif disttoghost < 4:
                ret += -50
            elif disttoghost < 5:
                ret += -25

        x = newPos[0]
        y = newPos[1]
        # print(listfood[x][y])
        # if listfood[x][y]:
        if list(currentGameState.getFood())[x][y]:
            # print("pos", newPos, ((lenlistfood * otherdist) * 1000000))
            # print("pos", newPos, (1000000))
            ret += 1000
            return ret


        for col in range(0, lenlistfood):
            for row in range(0, len(listfood[col])):
                if listfood[col][row] == True:
                    mandist = manhattanDistance(newPos, (col, row))
                    if mandist < mindist:
                        mindist = mandist


        # print("pos", newPos, (lenlistfood * otherdist) - mindist)
        # print("pos", newPos, (1 / mindist))
        return ret + (lenlistfood * otherdist) - mindist
        # return 1 / mindist

        #successorGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        #gameState.getNumAgents() 1 pac + ghosts
        #gameState._agentMoved which agent it is 0 pac other ghost

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        numagents = gameState.getNumAgents()

        def value(state, action, depth, index):

            if index % numagents == 0:
                depth += 1

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if depth == self.depth:
                return self.evaluationFunction(state)

            if not index:

                return maximizer(state, depth, index)
            return minimizer(state, depth, index)


        def maximizer(state, depth, index):

            v = float("-inf")

            acts = state.getLegalActions(index)

            thissuccs = [(state.generateSuccessor(index, a), a) for a in acts]
            for succ in thissuccs:
                v = max(v, value(succ[0], succ[1], depth, ((index + 1) % numagents)))
            return v


        def minimizer(state, depth, index):
            v = float("inf")

            acts = state.getLegalActions(index)
            thissuccs = [(state.generateSuccessor(index, a), a) for a in acts]
            for succ in thissuccs:
                v = min(v, value(succ[0], succ[1], depth, ((index + 1) % numagents)))
            return v


        maxval = float("-inf")
        maxvalaction = None
        for a in gameState.getLegalActions(self.index):
            val = value(gameState.generateSuccessor(0, a), a, 0, self.index + 1)
            if val > maxval:
                maxval = val
                maxvalaction = a
        return maxvalaction

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        numagents = gameState.getNumAgents()

        def value(state, action, depth, index, alpha, beta):

            if index % numagents == 0:
                depth += 1

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), alpha, beta

            if depth == self.depth:
                return self.evaluationFunction(state), alpha, beta

            if not index:

                return maximizer(state, depth, index, alpha, beta), alpha, beta
            return minimizer(state, depth, index, alpha, beta), alpha, beta


        def maximizer(state, depth, index, alpha, beta):
            v = float("-inf")

            acts = state.getLegalActions(index)

            for act in acts:
                thissucc = (state.generateSuccessor(index, act), act)
                val, alpha, beta = value(thissucc[0], thissucc[1], depth, ((index + 1) % numagents), alpha, beta)
                v = max(v, val)
                if v > beta:
                    return v
                else:
                    alpha = max(alpha, v)
            return v


        def minimizer(state, depth, index, alpha, beta):
            v = float("inf")

            acts = state.getLegalActions(index)
            for act in acts:
                thissucc = (state.generateSuccessor(index, act), act)
                val, alpha, beta = value(thissucc[0], thissucc[1], depth, ((index + 1) % numagents), alpha, beta)
                v = min(v, val)
                if v < alpha:
                    return v
                else:
                    beta = min(beta, v)
            return v


        maxval = float("-inf")
        maxvalaction = None
        alpha = float("-inf")
        beta = float("inf")
        for a in gameState.getLegalActions(self.index):
            val, alpha, beta = value(gameState.generateSuccessor(0, a), a, 0, (self.index + 1), alpha, beta)
            if val > maxval:
                maxval = val
                maxvalaction = a
                alpha = val
        return maxvalaction

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        numagents = gameState.getNumAgents()

        def value(state, action, depth, index):

            if index % numagents == 0:
                depth += 1

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if depth == self.depth:
                return self.evaluationFunction(state)

            if not index:
                return maximizer(state, depth, index)
            return minimizer(state, depth, index)


        def maximizer(state, depth, index):

            v = float("-inf")

            acts = state.getLegalActions(index)

            thissuccs = [(state.generateSuccessor(index, a), a) for a in acts]
            for succ in thissuccs:
                v = max(v, value(succ[0], succ[1], depth, ((index + 1) % numagents)))
            return v


        def minimizer(state, depth, index):
            v = float("inf")
            totalv = 0

            acts = state.getLegalActions(index)
            thissuccs = [(state.generateSuccessor(index, a), a) for a in acts]
            for succ in thissuccs:
                v = value(succ[0], succ[1], depth, ((index + 1) % numagents))
                totalv += v
                # print("totalv: ", totalv)
            # print("totalv, lenthissuccs, (totalv / lenthissuccs): ", totalv, len(thissuccs), (totalv / len(thissuccs)))
            return totalv / len(thissuccs)


        maxval = float("-inf")
        maxvalaction = None
        for a in gameState.getLegalActions(self.index):
            # print(gameState, a)
            val = value(gameState.generateSuccessor(0, a), a, 0, self.index + 1)
            if val >= maxval:
                maxval = val
                maxvalaction = a
        return maxvalaction

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    ret = 0
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newPos = currentGameState.getPacmanPosition()
    
    listfood = list(newFood)
    lenlistfood = len(listfood)
    otherdist = len(listfood[0])
    mindist = lenlistfood * otherdist
    area = mindist

    score = currentGameState.getScore()

    for i in range(0, len(newGhostStates)):
        disttoghost = manhattanDistance(newPos, newGhostStates[i].configuration.getPosition())
        if disttoghost < 2:
            ret += -1000
            # return ret
        elif disttoghost < 3:
            ret += -100
        elif disttoghost < 4:
            ret += -50
        elif disttoghost < 5:
            ret += -25

    for col in range(0, lenlistfood):
            for row in range(0, len(listfood[col])):
                if listfood[col][row] == True:
                    mandist = manhattanDistance(newPos, (col, row))
                    if mandist < mindist:
                        mindist = mandist

    caps = currentGameState.getCapsules()
    numcaps = len(caps)
    # print("listcaps: ", caps)
    for cap in caps:
        # print(cap, newPos)
        if (cap[0] == newPos[0]) and (cap[1] == newPos[1]):
            return 10000

    return ret + (score * 10) - mindist - (numcaps * 80)

   
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
