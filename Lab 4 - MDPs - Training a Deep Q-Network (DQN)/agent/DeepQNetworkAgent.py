from random import random, randrange, gauss
from sklearn.neural_network import MLPRegressor
import numpy as np
from agent.AgentBase import AgentBase, Verbosity
from ParkingDefs import StateType, Act
import pickle

class DeepQNetworkAgent(AgentBase):
    def __init__(self, name, numActions, numStates, probGreedy=.9, discountFactor=.7, learningRate=.0001, layer_sizes=[128], regularization=.0001, seed=10, verbosity=Verbosity.SILENT):
        super().__init__(name, verbosity)
        self.probGreedy = probGreedy
        self.evaluating = False
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.numActions = numActions
        self.numStates = numStates

        # Initialize the neural network model
        self.model = MLPRegressor(hidden_layer_sizes=layer_sizes, learning_rate_init=self.learningRate,warm_start=True, max_iter=1, alpha=regularization,random_state=seed)

        # Initial training to set up the model
        dummy_state = np.zeros((1, self.numStates))
        dummy_qvalues = np.zeros((1, self.numActions))
        self.model.fit(dummy_state, dummy_qvalues)

    def predict_qvalues(self, state):
        state_onehot = np.zeros((1, self.numStates))
        state_onehot[0][state] = 1
        return self.model.predict(state_onehot)[0]

    def findHighestValuedAction(self, state):
        q_values = self.predict_qvalues(state)
        return np.argmax(q_values)

    def selectAction(self, state, iteration, mdp):
        if self.evaluating or random() < self.probGreedy:
            return self.findHighestValuedAction(state)
        else:
            return randrange(self.numActions)

    def observeReward(self, iteration, currentState, nextState, action, totalReward, rewardHere):
        if not self.evaluating:
            current_q_values = self.predict_qvalues(currentState)
            next_q_values = self.predict_qvalues(nextState)

            # Update Q-value using Bellman equation
            bestNextAction = np.argmax(next_q_values)
            target = rewardHere + self.discountFactor * (next_q_values[bestNextAction] - current_q_values[action])
            current_q_values[action] = target

            # Train the model
            state_onehot = np.zeros((1, self.numStates))
            state_onehot[0][currentState] = 1
            self.model.partial_fit(state_onehot, [current_q_values])

        return super().observeReward(iteration, currentState, nextState, action, totalReward, rewardHere)

    #FIXME evaluate if this is breaking things when we try to switch the optimizers to SGD
    def setLearningRate(self, iteration, totalIterations):
        pass

    def loadModelFromPickle(self, filename):
        theFile = open(filename, "rb")
        self.model = pickle.load(theFile)
        # Initial training to set up the model
        #dummy_state = np.zeros((1, self.numStates))
        #dummy_qvalues = np.zeros((1, self.numActions))
        #self.model.fit(dummy_state, dummy_qvalues)

    def saveModelToPickle(self, filename):
        theFile = open(filename, "wb")
        pickle.dump(self.model, theFile)

    def determine_criticalities_huang(self):
        criticalities = []
        for state in range(self.numStates):
            q_values = self.predict_qvalues(state)
            crit = max(q_values) - np.average(q_values)
            criticalities.append((state, crit))
        return criticalities

    def determine_criticalities_amir(self):
        criticalities = []
        for state in range(self.numStates):
            q_values = self.predict_qvalues(state)
            crit = max(q_values) - min(q_values)
            criticalities.append((state, crit))
        return criticalities

    def printCriticalities(self, criticalities, parkingMode=False, numRows=2, numSpacesPerRow=10):
        if not parkingMode:
            print(criticalities)
        else:
            criticalities.sort(key=lambda tup: tup[1], reverse=True)
            for state, crit in criticalities:
                row, space = StateType.getIndices(state, numSpacesPerRow)
                print(row, "\t", space, "\t", StateType.get(state), "\t", crit)

    #FIXME identify a critical state in the toy MDP and parking MDP, do you agree (same task for non-critical)
    #FIXME have them write their own criticality function for the Q-learning agent
    #FIXME have them use criticality to "test" by ranking a set of agents arising from too-early stopping, mutation, and principled

    def mutate(self, variance):
        for layer in range(len(self.model.coefs_)):
            for listIdx in range(len(self.model.coefs_[layer])):
                for coefIdx in range(len(self.model.coefs_[layer][listIdx])):
                    self.model.coefs_[layer][listIdx][coefIdx] = gauss(0, variance)*self.model.coefs_[layer][listIdx][coefIdx]

    # This function is peculiar to the parking MDP domain, everything else provided here is general to table-based Q-learning
    def analyzeQfn(self):
        numSpacesDeclined = 0
        for state in range(self.numStates):
            stateClass = StateType.get(state)
            if StateType.DRIVING_AVAILABLE == stateClass:
                q_values = self.predict_qvalues(state)
                if q_values[Act.PARK.value] < q_values[Act.DRIVE.value]:
                    numSpacesDeclined += 1
        result = self.name + "'s final Q function declines to park in unoccupied spaces " + str(numSpacesDeclined) + " times"
        return result

    def __repr__(self):
        result = "\nLearning Rate: " + str(self.learningRate) + "\nValue Function\n\t\t"

        for i in range(self.numActions):
            result += 'Action{:>3}'.format(chr(i+ord('A'))) + "\t"
        result += "\n"

        for i in range(self.numStates):
            result += 'State{:>3}'.format(i) + ":\t"
            q_values = self.predict_qvalues(i)
            for q_value in q_values:
                formattedNum = "{: 7.1f}".format(q_value)
                if len(formattedNum) > 8:  # string length is equal to the length of the scientific notation formatting string on next lines
                    formattedNum = "{:.1E}".format(q_value)
                result += '{:>8}'.format(formattedNum) + "\t"
            result += "\n"

        result += "\nPolicy\n"
        for i in range(self.numStates):
            result += 'State{:>3}'.format(i) + ":\t"
            result += chr(self.findHighestValuedAction(i) + ord('A'))
            result += "\n"
        return result


