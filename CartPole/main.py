import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

# taking inputs and creating environment
LearningRate = 1e-3  # Learning rate : This can be experimented with.
env = gym.make("CartPole-v0")  # Making your environment
env.reset()
# These can be changed
goal_episode = 200
score_requirement = 50
initial_games = 1500


# making a random agent for trial
def makeRandomAgent():
    for episode in range(5):
        env.reset()
        # each frame upto 200.
        for t in range(200):
            # This will display the environment
            env.render()

            # This will just create a sample action.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break


# makeRandomAgent() I commented this out because it is not needed in the code, it just creates a temporary env to test it for the first time


# Learning from previous losses.
def initialRun():
    # OBSERVATION, ACTIONS
    training_data = []
    # FINDING SCORES
    scores = []
    # APRROXIMATE SCORES ACHEIVED
    accepted_scores = []
    # ITERATING THROUGH GAMES
    for _ in range(initial_games):
        score = 0

        game_memory = []  # sTORING THE RESULTS OF GAME
        # PREVIOUS OBSERVATION
        previous_observation = []
        # for each frame in 200
        for _ in range(goal_episode):
            # CHOOSING A RANDOM ACTION (0 OR 1) FOR (lEFT,RIGHT)
            action = random.randrange(0, 2)

            observation, reward, done, info = env.step(action)

            # STORING PREVIOUS ACTIONS AND OBSERVATIONS, AND USING PREVIOUS OBSERVATION TO LEARN.
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            previous_observation = observation
            score = score + reward
            if done: break
        # IF SCORE IS MET, METHODOLOGY IS BEING REINFORCED.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # OUTPUT LAYER FOR NEURAL NETWORK
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                # TRAINING DATA BEING SAVED
                training_data.append([data[0], output])

        # ENVIRONMENT BEING RESET
        env.reset()
        # SAVING SCORES
        scores.append(score)

    # SCORES CAN BE REFERED TO LATER
    training_data_saved = np.array(training_data)
    np.save('saved.npy', training_data_saved)

    # Displaying current average and median scores.
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    # RETURNING TRAINING DATA
    return training_data


# Creating a neural network with hyperparameters.
def neural_network_model(input_size):
    # Input Layer
    network = input_data(shape=[None, input_size, 1], name='input')
    # rectified linear unit (relu)
    # All 5 can satisfy a 128, but I just tested it to see what happens, there's no change
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    # Output Layer
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='Adam', learning_rate=LearningRate, loss='categorical_crossentropy',
                         name='targets')  # Adam is a type of optimizer
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def training_model(training_data, model=False):  # Training the model

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))
    # learning: one epoch = one pass of the full training set.
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


training_data = initialRun()
model = training_model(training_data)
scores = []  # Empty ARRAY
choices = []
for each_episode in range(10):
    score = 0
    game_memory = []
    prev_obs = []  # Different from previous_observation array
    env.reset()
    for _ in range(goal_episode):
        env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)  # Append the most recent choice to the end of the list

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation  # Making previous observation the new observation
        game_memory.append([new_observation, action])
        score += reward
        if done: break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))  # Printing average scores
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)  # Printing score requirement.