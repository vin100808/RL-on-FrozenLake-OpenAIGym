# testing SARSA on FrozenLake-v0
# 2022/02/08
# sample code from https://gym.openai.com/docs/

import gym
import random
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from gym import spaces
from gym.envs.registration import register
from IPython.display import clear_output


# GLOBAL ENV
ENV1 = 'CartPole-v0'
ENV2 = 'Acrobot-v1'
ENV3 = 'MountainCar-v0'
ENV4 = 'MountainCarContinuous-v0'
ENV5 = 'Pendulum-v0'

TOY1 = 'FrozenLake-v0'
TOY2 = 'FrozenLakeNoSlip-v1'


# reregister the game to toggle off is_slippery
try:
    register(
        id="FrozenLakeNoSlip-v1",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name": "4x4", 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.70,  # optimum = 0.74
    )
except:
    pass

def clear():
    os.system( 'clear' )

class agent():
    def __init__(self, env):
        self.action_size = env.action_space.n # possible moves
        print("Action size:", self.action_size)
    def get_action(self):
        action = random.choice(range(self.action_size)) # choose a random action from action pool
        return action
    
class QAgent(agent):
    '''Off-policy'''
    def __init__(self, env, discount_rate = 0.97, learning_rate = 0.01):
        super().__init__(env) # init via parent
        self.state_size = env.observation_space.n # possible states
        print("State size:", self.state_size)

        self.eps = 1.0 # e-greedy
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        '''
        [Q table]
        values: expected future reward at x state taking y action
        '''
        #self.q_table = 1e-4*np.random.random([self.state_size, self.action_size]) # rows -> states, columns -> actions
        self.q_table = np.full([self.state_size, self.action_size], 5e-5)

    def get_action(self, state):
        '''e-greedy'''

        # match state:
        #     case 0:

        q_state = self.q_table[state]
        
        action_greedy = np.argmax(q_state)
        action_random = super().get_action()

        return action_random if random.random() < self.eps else action_greedy 
        
    def train(self, state, action, next_state, next_action, reward, done):
        '''
        [Q LEARNING STEP]
        update: gets called after every move
        '''

        q_next = self.q_table[next_state, next_action]

        # Q value of the terminal state (in other words current state) is zero
        q_next = np.zeros([self.action_size]) if done else q_next

        # training the previous state
        # self.q_table[state, action] = self.q_table[state, action] \
        #                               + self.learning_rate * (reward + self.discount_rate * (np.max(q_next) - self.q_table[state, action]))

        self.q_table[state, action] = self.q_table[state, action] \
                                      + self.learning_rate * (reward + self.discount_rate * (self.q_table[next_state, next_action]) - self.q_table[state, action])
        

        if done:
            self.eps = self.eps * 0.99 # expoential decay to random action rate

def average_cal(array, start_index, end_index):
    '''
    used to calculate averages from a slice of an array
    input: array, starting index, ending index
    result: average
    '''
    temp = array[start_index: end_index]
    total_reward = 0
    total_number = end_index - start_index

    for i in range(len(temp)):
        total_reward += temp[i]

    return total_reward/total_number

def run(game):
    
    episodes = 1000
    plot_points = int(episodes/50) # 4
    # graph init
    x = [0]
    for i in range(plot_points):
        i += 1
        x.append(i*50 - 1)
    
    #x = [0, 49, 99, 149, 199, 249, 299, 349, 399, 449, 599, 649, 699, 749, 799, 849, 899, 949, 999] # nubmer of episodes (x-axis)
    y = np.zeros([plot_points+1]) # plotting every 50 episodes
    rewards = [] # rewards for each episodes (y-axis)

    # environment & agent init
    env = gym.make(game)
    agent = QAgent(env)
    total_reward = 0
    success = 0

    for ep in range(episodes):
        state = env.reset()
        done = False

        action = agent.get_action(state)

        while not done:
            print("s:", state, "a:", action)
            print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
            print(agent.q_table)
            env.render()
            time.sleep(0.001)

            next_state, reward, done, info = env.step(action)

            # choosing the next action already so that it can be used to update the Q value
            next_action = agent.get_action(next_state)
            
            agent.train(state, action, next_state, next_action, reward, done)

            state = next_state
            action = next_action
            total_reward += reward

            clear()        

        if reward == 1:
            success += 1
        # append the reward for this episode
        rewards.append(reward)

    # graphs
    
    for i in range(len(x)):
        if i == 0:
            y[i] = rewards[0]
        y[i] = average_cal(rewards, x[i-1], x[i])
    print("x:", x)
    print("size of x:", len(x))
    print("y:", y)
    print("size of y:", len(y))
    print("performance:", total_reward/episodes)

    plt.xlabel('ith episodes')
    plt.ylabel('the average rewards from previous 50 episodes')
    plt.title('SARSA on FrozenLake-v0 non-slippery')
    plt.axis([0, episodes, 0, 1.1]) # range of x and y axis
    plt.scatter(x, y)
    plt.show()
        




run(TOY2)


