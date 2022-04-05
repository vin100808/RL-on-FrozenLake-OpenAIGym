# ECE499 Report : Reinforcement Learning

##### By Vincent Lin

##### Computer Engineering

##### 20674085

## Introduction

### Background

The purpose of this report is to examine various Reinforcement Learning (RL) algorithms on different simulated environments provided by OpenAI gym (OpenAI). The goal is to study the performance of the algorithms depending on the types of environment.

### Reinforcement Learning (RL)

RL is a field from Machine Learning (ML) where it emphasizes on developing its own agents to make a serious of decisions within an environment. RL aims at building the experiences of an agent by trial and error, alongside with a reward and penalty system to determine whether an experience is good or bad. The goal is to transform an agent which takes random actions at first, to making sophisticated decisions after many (typically) trials.

### OpenAI gym

The RL algorithms will be trained on OpenAI. This technology is “a toolkit for developing and comparing reinforcement learning algorithms” (OpenAI, 2022). OpenAI provides a wide range of environments, from the entry level of classic control, toy text to more sophisticated 2D and 3D robots. OpenAI provides a well-documented wiki page from their Github with each environment’s description, observation, action space, episode termination and rewards. The details to these terminologies are:

- Observation: an environment-specific object illustrating the observation of the current environment state, these could include the current coordinate of the car from environment MountainCar-v0, and the velocity of the car. 
- Action space: the possible actions that the agent could take, it could be a discrete action space where agent could only pick a fixed range of non-negative numbers as their actions, or a continuous action space where agent could pick any numbers within a range provided. 
- Episode termination: specifies parameters on what terminates the environment.
- Rewards: the current reward policy built-in by OpenAI, however, users could develop their own rewards policy.

To sum up, OpenAI is a sophisticated toolkit for RL and environment-specific parameters will be discussed in later sections. This report will test various algorithms on FrozenLake-v0 (non-slippery version), and CartPole-v0.

### RL Algorithms

As RL was first introduced in 1965 in an engineering literature (Sutton & Barto, 2014), it has been studied and refined due to its importance and usefulness in the past few decades. There are many great algorithms which performs differently based on the characteristics of the environment. This report will focus on the following algorithms: Q-learning (QL), State-action-reward-state-action (SARSA) and double Q . 

 

The differences, the strength, the weakness (in general), but then make a conclusion later based on results

**actually leave it for the algorithms big section**

### Setup

The results of this report are produced on a MacBook Pro (2016), with a processor of 2.9GHz Dual-Core Intel Core i5, 8GB memory, written in python, and uses various the following libraries:

- gym
- random
- numpy
- time
- os

## Algorithms

### Q-learning (QL)

**sneak double QL in here as well just briefly explain difference**



### State-action-reward-state-action (SARSA)

 





