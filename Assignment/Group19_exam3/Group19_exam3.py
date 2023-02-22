import pandas as pd
import numpy as np
import gym
import random

#----------------------
#   1
# 3   2 pickup: 4, dropout: 5
#   0
#----------------------

# Main
def main():
    agent = Agent()
    x, y, pick, drop = 1, 1, 'Y', 'B'

    print(f"taxi_row: {x}, taxi_col: {y}, pickup: {pick}, dropoff: {drop}")
    pickup, dropoff = P_D[pick], P_D[drop]
    agent.Train_agent()
    print("Reward + Action: ",agent.Strategy(taxi_row=x, taxi_col=y, pickup=pickup, dropoff=dropoff))
    env.close()
#------------------------------------

# dict pick, drop
P_D = {'R': 0, 'G': 1, 'Y': 2, 'B' : 3, 'in': 4}
#----------------------

# Hyper parameters
total_episodes = 99999
total_test_episodes = 99
max_steps = 99

epsilon = 0.632
learning_rate = 0.7
gamma = 0.618

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01
#-------------------------

# create env
env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
#------------------------

#create qtable
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
#-------------------------

# class
class Agent:
    def __init__(self):
        print('Initialization Successful Agent "Taxi - V3:"')
    def convert_state(self, old_state):
        return old_state[0]

    # create values for qtable
    def Train_agent(self):
        global epsilon
        for episode in range(total_episodes):
            State = env.reset()
            state = self.convert_state(State)
            done = False
            while not done:
                exp_exp_tradeoff = random.uniform(0, 1)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(qtable[state, :])
                else:
                    action = env.action_space.sample()
                new_state, reward, done, _, _ = env.step(action=action)
                qtable[state, action] = (1 - learning_rate) * qtable[state, action] + learning_rate * \
                                        (reward + gamma * np.max(qtable[new_state, :]))
                state = new_state
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

    # strategy
    def Strategy(self, taxi_row, taxi_col, pickup, dropoff):
        reward_main = 0
        list_main = []
        for epsilon in range(1):
            rewards = 0
            done = False
            action_list = []
            env.reset()
            state = env.encode(taxi_row, taxi_col, pickup, dropoff)
            env.s = state
            for step in range(max_steps):
                action = np.argmax(qtable[state,:])
                action_list.append(action)
                new_state, reward, done, _, _ = env.step(action=action)
                rewards += reward
                # print(env.render())
                if done:
                    print(rewards)
                    if rewards >= reward_main:
                        reward_main = rewards
                        list_main = action_list
                    # print("reward: ", reward_main)
                    # print("action: ",  list_main)
                    break
                state = new_state
        return (reward_main, list_main)

#----------------------------------
main()

