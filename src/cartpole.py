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

def v1():
    '''
    simple version that runs cartpole and ends after 1000 timesteps regardless of termination checks
    '''
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action

    env.close()

def v2():
    '''
    retrieves returned values after each action and determine if the episode should be ended
    TODO: who defines the end? gym does it based on the game?
    '''
    
    env = gym.make('CartPole-v0') # the env is now a handle for our cartpole game environment

    for i_episode in range(20): # 20 episodes/tries
        observation = env.reset() # obtain an initial observation object via reset
        
        for t in range(100): 
            env.render()
            print(observation)
            # env.action_space has valid actions for this specific game
            # there are also env.observation_space
            action = env.action_space.sample() # random action
            observation, reward, done, info = env.step(action) # after env takes a step, retrieve all 4 returned values

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        env.close()

def v3():
    '''
    outputs information about valid actions and observations specifically to the game of cartpole
    also checks the upper and lower bound of observations
    Box space represetns and n-dimensional box
    '''
    env = gym.make('CartPole-v0')
    print("ACTION SPACE:")
    print(env.action_space)
    #> Discrete(2)
    print("OBSERVATION SPACE:")
    print(env.observation_space)
    #> Box(4,)
    print(env.observation_space.high)
    print(env.observation_space.low)

    space = spaces.Discrete(8) # set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    assert space.contains(x)
    assert space.n == 8
    print(space.n) # number of elements
    #> 8 
    

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n # 2 (only two possible moves)
        print("Action size:", self.action_size)

    def get_action(self, observation):
        action = random.choice(range(self.action_size)) # exclusive so only 0, 1 is possible
        # pole_angle = observation[2] # second field is pole angle
        # action = 0 if pole_angle < 0 else 1 # push to the left if angle is negative (leaning left) and otherwise
        return action

class QAgent(Agent):
    def __init__(self, env, discount_rate = 0.97, learning_rate = 0.01):
        super().__init__(env) # subclass of Agent
        self.state_size = env.observation_space.n
        print('State size:', self.state_size)

        self.eps = 1.0 # eps was introduce to balance the explorative action and the policy action (to get out of the minimal maxima)
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self): # builds the Q table
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size]) # define the Q table where state as rows and action as columns
                                                                                  # this is going to have 16 elements indicating the 16 states  
                                                                                  # each states have 4 elements indicating at that state, what's the Q value of taking that action
        #self.q_table = np.full([self.state_size, self.action_size], 5e-5)
        
    def get_action(self, state): # new get action method depends on where the current state is
        q_state = self.q_table[state] 
        #> [Q value for taking action 0, Q value for taking action 1, ...]
        action_greedy = np.argmax(q_state) # we take the action which has the highest Q value (the most rewarding)
        action_random = super().get_action(state) # directly use the super class's get_action which randomly selects an action
        
        return action_random if random.random() < self.eps else action_greedy # this is similar to 457A where explorative actions are prioritized first then policy actions

    def train(self, experience):
        '''for updating the Q table at each step'''
        state, action, next_state, reward, done = experience # TODO: need to understand this line further in order to grasp the rest of the method

        q_next = self.q_table[next_state] # q value for next state, only non terminal state have a next state
        
        #q_next = np.zeros([self.action_size]) if done else q_next # therefore we set the Q values to all zeros if done
        if done: 
            print("DONEDONEDONE")
            q_next = np.zeros([self.action_size])
        
        # according to equation and this is only for ONE state
        # target: (reward of next state)  +  discount rate * (Q value of next state)
        #q_target = reward + self.discount_rate * np.max(q_next) 
        
        #  learned: q(s,a)          =              q(s,a)         +   learning rate    * (                       target                  -             q(s,a)           ) 
        # self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * ( (reward + self.discount_rate * np.max(q_next) - self.q_table(state, action) ))

        #q_update = q_target - self.q_table[state, action]
        #self.q_table[state, action] += self.learning_rate * q_update # according to equation for the learned Q value

        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.discount_rate * (np.max(q_next) - self.q_table[state, action]))

        if done:
            self.eps = self.eps * 0.99 # apply a exponential decay to the eps 

def v4(game):
    '''
    
    '''
    
    env = gym.make(game)
    agent = QAgent(env) 
    observation = env.reset() # initial observation object

    for t in range(200):
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        
        # if done: # let default done to determine failure (pretty sensitive btw i think because once the pole gets over certain angle its going to fail no matter what)
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break

    env.close()

def v5(game):
    env = gym.make(game)
    agent = QAgent(env)
    
    total_reward = 0
    
    for ep in range(1000): # train for 100 episodes
        state = env.reset() 
        done = False

        while not done: # for each episode thats not done
            action = agent.get_action(state) #state is merely a value from 0-15
            next_state, reward, done, info = env.step(action)

            # TODO: reward small if its not falling off
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
                
v5(TOY2)


#print(1e-4*np.random.random([16, 4])) # define the Q table where state as rows and action as columns
# question:
# boundaries? cant go out of the map
# the equation itself
