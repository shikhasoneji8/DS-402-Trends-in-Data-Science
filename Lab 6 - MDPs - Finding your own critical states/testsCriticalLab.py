from random import seed
from ParkingLotFactory import createParkingMDP
from agent.DeepQNetworkAgent import DeepQNetworkAgent, Verbosity
from agent.QLearningAgent import QLearningAgent, Verbosity
from MDP import MDP, generate_random_mdp
from statistics import stdev

def trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, pickle=False, maxTrajectoryLen=50):
    output = []
    for epoch in range(numEpochs):
        # first, learn for <numSamples> trajectories
        for trajectory in range(numSamples):
            _ = mdp.simulateTrajectory(agent, startState=start)
            if agent.verbosity == Verbosity.VERBOSE:
                print("\n **************************Epoch {}, sample trial: {} Completed**************************".format(epoch + 1, trajectory + 1))

        if pickle:
            agent.saveModelToPickle("output/pickle"+str(epoch))

        # second, evaluate for <numSamples> trajectories
        agent.evaluating = True
        rewardList = []
        for _ in range(numSamples):
            reward = mdp.simulateTrajectory(agent, startState=start, MAX_ACTIONS=maxTrajectoryLen)
            rewardList.append(reward)
        agent.evaluating = False

        rewardStdev = 0
        if len(rewardList) > 1:
            rewardStdev = stdev(rewardList)
        # print to console
        print("At Epoch\t{}\tOn\t{}\t, Over {} Trajectories, Agent\t{}\t achieved a Grand Total Reward:\t{:.4f}\t, for an average of\t{:.4f}\t with std\t {:.4f}.".format(
                epoch, mdp.name, len(rewardList), agent.name, sum(rewardList), sum(rewardList) / len(rewardList),
                rewardStdev))
        output.append(sum(rewardList) / len(rewardList))

    print()
    return output

# Task 1's test
def CriticalLabTest1(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 1 - Criticalities from DQN agent on the toy MDP, with Seed " + str(seedValue))

    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")
    start = 0

    # configure the DQN
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization
    agent = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)

    numSamples = 50
    numEpochs = 50
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=20)
    print("Criticalities, Huang measure [(state, criticality),...]:\n ", agent.determine_criticalities_huang())
    print()
    print("Criticalities, Amir measure [(state, criticality),...]:\n ", agent.determine_criticalities_amir())

# Task 2's test
def CriticalLabTest2(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 2 - Criticalities from Q-Learning agent on the toy MDP, with Seed " + str(seedValue))

    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")
    start = 0

    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates)

    numSamples = 50
    numEpochs = 50
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=20)

    #TODO this part will break if you haven't written these functions yet
    print("Criticalities, Huang measure [(state, criticality),...]:\n ", agent.determine_criticalities_huang())
    print()
    print("Criticalities, Amir measure [(state, criticality),...]:\n ", agent.determine_criticalities_amir())

# Task 3's test
def CriticalLabTest3(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 3 - Criticalities from DQN agent on a harder Parking MDP, with Seed " + str(seedValue))

    busyRate = .9
    numRegularSpacesPerRow = 5
    numHandicappedSpacesPerRow = 3
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate, numHandicappedSpacesPerRow=numHandicappedSpacesPerRow, numRegularSpacesPerRow=numRegularSpacesPerRow)

    # configure the DQN
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization
    agent = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)

    numSamples = 50
    numEpochs = 100
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=20)
    print(agent.analyzeQfn())
    print("\nCriticalities, Huang measure")
    agent.printCriticalities(agent.determine_criticalities_huang(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)
    print("\nCriticalities, Amir measure")
    agent.printCriticalities(agent.determine_criticalities_amir(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)


# Task 4's test
def CriticalLabTest4(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 4 - Criticalities from Q-Learning agent on a harder Parking MDP, with Seed " + str(seedValue))

    busyRate = .9
    numRegularSpacesPerRow = 5
    numHandicappedSpacesPerRow = 3
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate, numHandicappedSpacesPerRow=numHandicappedSpacesPerRow, numRegularSpacesPerRow=numRegularSpacesPerRow)

    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates)

    numSamples = 50
    numEpochs = 100
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=20)
    print(agent.analyzeQfn())

    print("\nCriticalities, Huang measure")
    agent.printCriticalities(agent.determine_criticalities_huang(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)
    print("\nCriticalities, Amir measure")
    agent.printCriticalities(agent.determine_criticalities_amir(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)

def CriticalLabTest5(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - Criticalities from both agents on a random MDP, with Seed " + str(seedValue))

    numStates = 10
    numActions = 40
    start = 0
    mdp = generate_random_mdp(numStates, numActions)

    # configure the DQN
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization
    learningRate = .0001
    agent1 = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, learningRate=learningRate, regularization=regularization, seed=seedValue)

    learningRate = .01
    agent2 = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, learningRate=learningRate)

    numSamples = 50
    numEpochs = 50

    agents = [agent1, agent2]
    for agent in agents:
        print(agent.name + "*************************************************")
        trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=40)
        print("Criticalities, Huang measure [(state, criticality),...]:\n ", agent.determine_criticalities_huang())
        print()
        print("Criticalities, Amir measure [(state, criticality),...]:\n ", agent.determine_criticalities_amir())
        print()


def CriticalLabTestSECRET(seedValue):
    pass

#task 6's test
def CriticalLabTest6(seedValue):
    seed(seedValue)
    print(
        "\n\n\n----------------------- BEGIN Test 6 - Comparing DQNs via criticality, with Seed " + str(
            seedValue))

    busyRate = .9
    numRegularSpacesPerRow = 20
    numHandicappedSpacesPerRow = 8
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate, numRegularSpacesPerRow=numRegularSpacesPerRow, numHandicappedSpacesPerRow=numHandicappedSpacesPerRow)

    # configure the DQN
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization

    agent1 = DeepQNetworkAgent("DQN_pickleBLUE", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent1.loadModelFromPickle("data/pickleBLUE")

    agent2 = DeepQNetworkAgent("DQN_pickleGREEN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent2.loadModelFromPickle("data/pickleGREEN")

    agent3 = DeepQNetworkAgent("DQN_pickleORANGE", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent3.loadModelFromPickle("data/pickleORANGE")

    agent4 = DeepQNetworkAgent("DQN_picklePURPLE", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent4.loadModelFromPickle("data/picklePURPLE")

    agent5 = DeepQNetworkAgent("DQN_pickleRED", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent5.loadModelFromPickle("data/pickleRED")

    agent6 = DeepQNetworkAgent("DQN_pickleYELLOW", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue)
    agent6.loadModelFromPickle("data/pickleYELLOW")

    agents = [agent1, agent2, agent3, agent4, agent5, agent6]
    for agent in agents:
        print(agent.name)
        agent.printCriticalities(agent.determine_criticalities_huang(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)
        print()
        agent.printCriticalities(agent.determine_criticalities_amir(), parkingMode=True, numSpacesPerRow=numRegularSpacesPerRow + numHandicappedSpacesPerRow)
        print()
