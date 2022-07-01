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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            temp_values = util.Counter()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                maxvalue = -99999
                flag = False
                for action in actions:
                    transition = self.mdp.getTransitionStatesAndProbs(state, action)
                    sum_of_values = 0.0
                    for state_prob in transition:
                        temp_val = self.discount * self.values[state_prob[0]]
                        sum_of_values += state_prob[1] * (self.mdp.getReward(state, action, state_prob[0]) + temp_val)
                    maxvalue = max(maxvalue, sum_of_values)
                if maxvalue != -99999:
                    temp_values[state] = maxvalue
                    flag = True

            for state in states:
                self.values[state] = temp_values[state]

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
        value = 0.0
        for stateProb in self.mdp.getTransitionStatesAndProbs(state, action):
            temp_value = self.discount * self.values[stateProb[0]]
            value += stateProb[1] * (self.mdp.getReward(state, action, stateProb[0]) + temp_value)

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        max_value = -99999
        max_action = None

        for action in self.mdp.getPossibleActions(state):
            action_value = self.computeQValueFromValues(state, action)
            if action_value > max_value:
                max_value = action_value
                max_action = action
        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        get_q_values_of_state_action = lambda state: [self.getQValue(state, action) for action in
                                                      self.mdp.getPossibleActions(state)]
        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if self.mdp.isTerminal(state):
                continue
            self.values[state] = max(get_q_values_of_state_action(state))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # first we create the initialized variables for our algorithm
        queue = util.PriorityQueue()
        predecessors = {}

        # creating predecessors dictionary
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    # for every action we get the transition states and probabilities
                    for stateProb in self.mdp.getTransitionStatesAndProbs(state, action):
                        # if the state is not in the predecessors dictionary we add it
                        if stateProb[0] in predecessors:
                            predecessors[stateProb[0]].add(state)
                        else:  # if the state is not in the predecessors dictionary we add it
                            predecessors[stateProb[0]] = {state}

        get_values = lambda state: [self.computeQValueFromValues(state, action_in_state) for action_in_state in
                                    self.mdp.getPossibleActions(state)]
        # for non-terminal states we add them to the queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # we find the max q value for the state and calculate the error
                diff = abs(self.values[state] - max(get_values(state)))
                # we add the state to the queue with the negative error
                queue.update(state, -diff)

        # for the number of iterations we run the algorithm
        for iteration in range(self.iterations):
            if queue.isEmpty():
                break
            # we get the state with the lowest error
            state = queue.pop()
            # first we set the init state value
            if not self.mdp.isTerminal(state):
                # finding the max q value for the state and set it as the state value
                self.values[state] = max(get_values(state))

            # calculating the diff again, and then we will update the value
            # if it was more than theta
            for pred in predecessors[state]:
                if not self.mdp.isTerminal(pred):
                    diff = abs(self.values[pred] - max(get_values(pred)))
                    if diff > self.theta:
                        queue.update(pred, -diff)
