# testing cartpole 
# 2022/02/08
# sample code from https://gym.openai.com/docs/

import gym
import random
import numpy as np
import time
import os
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
    '''Off policy'''
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
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size]) # rows -> states, columns -> actions
    
    def get_action(self, state):
        '''e-greedy'''

        # match state:
        #     case 0:

        q_state = self.q_table[state]
        
        action_greedy = np.argmax(q_state)
        action_random = super().get_action()

        return action_random if random.random() < self.eps else action_greedy 
        
    def train(self, experience):
        '''
        [Q LEARNING STEP]
        update: gets called after every move
        '''
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]

        # Q value of the terminal state (in other words current state) is zero
        q_next = np.zeros([self.action_size]) if done else q_next

        # training the previous state
        self.q_table[state, action] = self.q_table[state, action] \
                                      + self.learning_rate * (reward + self.discount_rate * (np.max(q_next) - self.q_table[state, action]))
                                       
        if done:
            self.eps = self.eps * 0.99 # expoential decay to random action rate

def run(game):
    
    env = gym.make(game)
    agent = QAgent(env)
    total_reward = 0

    for ep in range(1000):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.train((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

            print("s:", state, "a:", action)
            print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
            print(agent.q_table)
            env.render()
            time.sleep(0.05)

            if ep != 99:
                clear()

run(TOY2)


