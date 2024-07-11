# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.QValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # the Q-value of the given state-action pair from the QValues dictionary and returns it.
        return self.QValues[(state, action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions for the current state
        allowedActions = self.getLegalActions(state)

        # If there are no legal actions for the current state, return 0.0
        if len(allowedActions) == 0:
            return 0.0

        # Find the maximum Q-value among all legal actions
        # Set the initial max Q-value to negative infinity
        maxQValue = float('-inf')

        # we loop over all legal actions for the current state
        for action in allowedActions:

            # Get the Q-value for the current state-action pair
            qValue = self.getQValue(state, action)

            # if the current Q-value is greater than the current max Q-value, update the max Q-value
            if qValue > maxQValue:
                maxQValue = qValue

        # Return the maximum Q-value over all legal actions for the current state
        return maxQValue

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get legal actions for the given state
        legalActions = self.getLegalActions(state)

        # If there are no legal actions, return None
        if len(legalActions) == 0:
            return None

        # Choosing the the action with the maximum Q-value
        bestAction = legalActions[0]

        # then we nitially set best Q value to negative infinity
        maxQValue = float('-inf')
        for action in legalActions:
            # Get the Q-value for the action in the current state
            qValue = self.getQValue(state, action)

            # Check if the  Q-value is greater than the current maximum, 
            # update the maximum and best action
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action
        
        # Return the action with the maximum Q-value
        return bestAction       
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"


        # Get the legal actions in the current state
        legalActions = self.getLegalActions(state)
        action = None

        # Check if there are no legal actions in the current state
        if len(legalActions) == 0:
            return None

        # Randomly decide whether to take a random action or best policy action
        if not util.flipCoin(self.epsilon):
            # Take the best policy action
            action = self.getPolicy(state)
        else:
            # Take a random action
            action = random.choice(legalActions)

        return action

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Get the current Q-value estimate for the (state, action) pair.
        current_q_value = self.QValues.get((state, action), 0)

        # Compute the expected future reward based on the nextState.
        next_q_value = self.getValue(nextState)
        new_q_value = current_q_value + self.alpha * \
            (reward + self.discount * next_q_value - current_q_value)
        
        # Update the Q-value estimate for the (state, action) pair.
        self.QValues[(state, action)] = new_q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # Extract the feature vector for the state-action pair
        features = self.featExtractor.getFeatures(state, action)

        # Here is calculates the dot product between the feature vector amd the weight vector,
        # which gies us the Q-value for the gien state-action pair.
        # like calculate the dot product between the feature vector and the weights
        # Then using a generator expression and the sum() function to compute the dot product
        # For each (feature, value) pair in the features dictionary, we multiply the weight associated with the feature by the value,
        # and then sum up all these products to get the dot product
        q_value = sum(self.weights[feature] * value for feature, value in features.items())

        # Return the dot product as the Q-value for the given state-action pair
        return q_value

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # we Calculate the difference between the observed reward and the expected reward for the current state and action
        diff = (reward + self.discount *
                    self.getValue(nextState)) - self.getQValue(state, action)

        # Get the features for the current state and action
        features = self.featExtractor.getFeatures(state, action)

        # Then For each feature,we update its weight by the difference multiplied by the feature's value,
        # and then multiplied by the learning rate alpha
        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass
