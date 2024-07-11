# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(iterations):

            # Create a copy of the current values for iteration
            values_copy = self.values.copy()
            for state in mdp.getStates():  # Iterate over all states in the MDP
                
                if mdp.isTerminal(state):   # Skip terminal states
                    continue
                # we Find the maximum Q-value for all possible actions in the current state
                max_q_value = float('-inf')
                for action in mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
                # Update the value of the state with the maximum Q-value
                values_copy[state] = max_q_value
            self.values = values_copy  # Set the updated values for the next iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #used the bellman equation to help in this
        q_value = sum([prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
        return q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get all possible actions for the given state
        possible_actions = self.mdp.getPossibleActions(state)

        # Check if there are any possible actions, otherwise return None
        if not possible_actions:
            return None

        # Calculate the Q-value for each possible action
        q_values = [self.getQValue(state, action)
                    for action in possible_actions]

        # Find the index of the action with the highest Q-value
        best_action_index = q_values.index(max(q_values))

        # Return the action with the highest Q-value
        return possible_actions[best_action_index]
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
