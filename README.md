# ECE499 Report : Reinforcement Learning

Supervised by Professor. Mark Crowley

By Vincent Lin

Computer Engineering

20674085

[toc]

## Introduction

### Background

The purpose of this report is to examine various Reinforcement Learning (RL) algorithms on simulated environments provided by OpenAI gym (OpenAI). The goal is to study the performance of the algorithms on various domains.

### Reinforcement Learning (RL)

RL is a field from Machine Learning (ML) where it emphasizes on developing its own agents to make a serious of decisions within an environment. RL aims at building the experiences of an agent by trial and error, alongside with a reward and penalty system to determine whether an experience is good or bad. The goal is to transform an agent which takes random actions at first, to making sophisticated decisions after many (typically) trials.

### OpenAI gym

The RL algorithms will be trained on OpenAI. This technology is “a toolkit for developing and comparing reinforcement learning algorithms” (OpenAI, 2022). OpenAI provides a wide range of environments, from the entry level of classic control, toy text to more sophisticated 2D and 3D robots. OpenAI provides a well-documented wiki page from their Github with each environment’s description, observation, action space, episode termination and rewards. The details to these terminologies are:

- Observation: an environment-specific object illustrating the observation of the current environment state, these could include the current coordinate of the car from environment MountainCar-v0, and the velocity of the car. 
- Action space: the possible actions that the agent could take, it could be a discrete action space where agent could only pick a fixed range of non-negative numbers as their actions, or a continuous action space where agent could pick any numbers within a range provided. 
- Episode termination: specifies parameters on what terminates the environment.
- Rewards: the current reward policy built-in by OpenAI, however, users could develop their own rewards policy.

To sum up, OpenAI is a sophisticated toolkit for RL and environment-specific parameters will be discussed in later sections. This report will test various algorithms on the environments FrozenLake-v0 (non-slippery version), and CartPole-v0.

### RL Algorithms

As RL was first introduced in 1965 in an engineering literature, it has been studied and refined due to its importance and usefulness in the past few decades. There are many great algorithms which performs differently based on the characteristics of the environment. This report will focus on the following algorithms: Q-learning (QL), State-action-reward-state-action (SARSA) and double Q . 

There are a few concepts which apply to general RL algorithms, they are:

- Temporal Difference (TD): "TD is an approach to learning how to predict a quantity that depends on future values of a given signal" (Barto, 2007). 
- Q table: generally is a table where row and column represents the state and action respectively, each Q value represents the expected long term reward by taking that action from that state (Generally updated via TD).
- On-policy: meaning the algorithm uses the same policy for both updating the Q values and choosing the next action.
- Off-policy: meaning the algorithm uses different policies for updating the Q values and choosing the next action.
- π-greedy policy: a policy which features both exploitations and explorations, exploitation uses prior knowledge of the agent (Q table) and exploration uses π to determine whether a random action should be taken instead of exploiting.
- μ greedy policy: a policy which always chooses the best Q value.
- γ: discount rate of the future rewards, in other words, how important does the algorithm values the future rewards.
- α: learning rate when updating Q values, the larger it is, the more the sum of immediate reward and discounted estimate of optimal future value is roportionate d in the overall Q value (Venkatachalam, 2021). 

### Setup

The results of this report are produced on a MacBook Pro (2016), with a processor of 2.9GHz Dual-Core Intel Core i5, 8GB memory, written in python, and used the following libraries:

- gym
- random
- numpy
- time
- os
- matplotlib.pyplot

## Algorithms

### State-action-reward-state-action (SARSA)

SARSA is an on-policy TD control method. The algorithm uses π-greedy policy for both choosing an action and updating the Q value, and therefore it is an on-policy algorithm. The pseudo code is:

 ```pseudocode
 Initialize Q(s, a), for all s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0 Repeat (for each episode):
 	Initialize S
 	Choose A from S using policy derived from Q (e.g., ε-greedy) 
 	Repeat (for each step of episode):
 		Take action A, observe R, S′
 		Choose A′ from S′ using policy derived from Q (e.g., ε-greedy) 
 		Q(S, A) ← Q(S, A) + α[R + γQ(S′, A′) − Q(S, A)]
 		S ← S′; A ← A′;
 	until S is terminal
 ```

(Sutton & Barto, 2020, p. 106)

The code initializes Q tables based on the environment-specific spaces (states and actions). Then for each episodes, SARSA resets the game by resetting the states, then immediately choose an action using the π-greedy policy, it could be a random action or the best action derived from the Q table depending on the π value.

After taking an action, the code updates the Q value of the previous state and action by this equation

```pseudocode
Q(S, A) ← Q(S, A) + α[R + γQ(S′, A′) − Q(S, A)]
```

The equation is essentially updating the old Q value with a learning rate and discount rate applied immediate rewards and future rewards. Notice that the future Q value used here is the action taken and the state it arrived, which is the same result that the π-greedy policy produced. 

### Q-learning (QL)

QL is an off-policy TD control method. This algorithm is similar to SARSA, they both choose the next action using the π-greedy policy, however, it updates the Q value differently. Here is the pseudo code: 

```pseudocode
Initialize Q(s, a), for all s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0 Repeat (for each episode):
	Initialize S
	Repeat (for each step of episode):
		Choose A from S using policy derived from Q (e.g., ε-greedy) 
		Take action A, observe R, S′
		Q(S, A) ← Q(S, A) + α[R + γ*(maximum value of)Q(S′, a) − Q(S, A)]
		S ← S′
until S is terminal
 
```

(Sutton & Barto, 2020, p. 106)

The code's initialization is identical to SARSA and therefore will not iterate. One difference which differentiates QL from SARSA is the update function: 

``````pseudocode
Q(S, A) ← Q(S, A) + α[R + γ*(maximum value of)Q(S′, a) − Q(S, A)]
``````

The difference is the future reward used in the equation. Personally I believe the notation from Sutton & Barto is not clear enough, I would like to treat it as:

```pseudocode
Q(S, A) ← Q(S, A) + α[R + γ*(maximum value of)Q(S′) − Q(S, A)]
```

This is because the notation `Q(S', a)` represents the Q value of taking action `a` at state `S'`, however, the algorithm is actually trying to find the action `a` at state `S'` which gives the maximum value of Q. Therefore, I believe the second equation is better to understand. Notice that this is not the same policy as the π-greedy policy, which was used to select the next action. The Q value is updated using the action from the state  `S'` which produces the highest Q value, it may not be the action we actually choose in the state `S'` since there is a possibility that the algorithm chooses to explore other actions rather than expolit on the best action.

## Applications

### FrozenLake-v0

FrozenLake is a toy text environment, the description of the game can be found in https://gym.openai.com/envs/FrozenLake-v0/, it is essentially a game with a 4x4 grid, there exists a preset starting point, ending point, and holes. Player starts from the starting point, and move around the grid until it finds the ending point. The game terminates when the player step on both the ending point or any hole on the grid. Upon arriving at the ending point, the environment gives a reward of 1, and no penalties for stepping on a hole.

Moreover, for the purpose of analysis, the slippery feature is turned off, that is, when a player chooses to move in a direction, it will move exactly 1 box in that direction instead of having the possibility of slipping to further boxes.

#### SARSA

The implementation can be found in `./src/SARSA_Frozen.py`.

The implementation of SARSA on FrozenLake is a little tricky compare to QL, because the updating function requires the next action to be selected using π-greedy policy at the next state, the next action needed to be remember by the program. In other words, when the very first action is chosen and the next state is known, we need to select the next action such that the Q value can be updated. Subsequently, in the next iteration, we use the already selected next action as the action. It could be imagined as "thinking one step ahead". 

Personalized initialization includes:

- learning rate = 0.01 (very low since future rewards are not as important)
- discount rate = 0.97 
- epsilon = 1 (an exponential decay of 0.99 is applied after each step taken to gradually transition from exploration to exploitation)
- number of episodes = 1000
- Q table: initialized to a uniform 5e-5 (i.e. for all states and actions)

The resulting graph is (can be found under `./src/graphs/SARSA_Frozen.png`):

![SARSA_Frozen](/Users/zzzh/Documents/University of Waterloo/Winter2022[4B]/ece499/src/graphs/SARSA_Frozen.png)

This scatter plot graph illustrates the average number of rewards calculated in batches of 50 episodes vs. the ith episodes. To be more specific, for every 50th episode, the average reward from the previous 50 episodes is calculated. Therefore, for 1000 episodes, there should be 20 data points. Note that the maiximum average reward can be achieve is 1, because the implementation did not add any additional rewards (e.g. extra rewards for staying alive). We could observe that SARSA was able to find its first solution quickly and slowly fixate on this solution from around episode 150 to 250. It took around 100 episodes to transition to this solution. This graph achieved a success rate of 80.1%, which means out of 1000 episodes, 801 of them were successful.

#### QL

The implementation can be found in `./src/QL_Frozen.py`.

The implmentation of QL on FrozenLake is identical to the pseudocode discussed earlier. However, if an action will terminate the episode, then the future reward is treated as 0: 

```pseudocode
Q(S, A) ← Q(S, A) + α[R + 0 − Q(S, A)]
```

This is added such that when the agent encounters an undesired action, it will decrease the Q value so that in the future it is less likely to be selected.

Personalized initialization includes:

- learning rate = 0.01 (very low since future rewards are not as important)
- discount rate = 0.97 
- epsilon = 1 (an exponential decay of 0.99 is applied after each step taken to gradually transition from exploration to exploitation)
- number of episodes = 1000
- Q table: initialized to a uniform 5e-5 (i.e. for all states and actions)

The resulting graph is (can be found under `./src/graphs/QL_Frozen.png`):

![QL_Frozen](/Users/zzzh/Documents/University of Waterloo/Winter2022[4B]/ece499/src/graphs/QL_Frozen.png)

We could observe from the graph that once the agent finds a solution successfully, it will converge and fixate to that path relatively quick. From around episode 100 to 300, it only took 200 episodes and after that, the average reward became stable at 1 (with some exceptions). Lastly, the graph achieved a success rate of 88.8%.

#### Conclusion

##### Choices of the factors

The learning rate was selected to be a small factor because the future is uncertain, and therefore should not be too large and cause undesired results. Another explaination to it is that learning rate determines how fast the agent learns, and I believe with a smaller learning rate, the agent will learn slowly but in the right direction. Furthermore, the discount rate determines the how much weight should be put on distant future compare to immediate future, with a low discount rate, the agent tends to update the table with immediate rewards. However, this will not work on this implementation because no additional reward policy was implemented. Using the update function from QL for example:

```pseudocode
Q(S, A) ← Q(S, A) + α[R + γ*(maximum value of)Q(S′) − Q(S, A)]
```

If the discount rate is low, then the update function becomes:

```pseudocode
Q(S, A) ← Q(S, A) + α[R − Q(S, A)]
```

However, `R` is always 0 unless the goal state is reach, and therefore the agent is reducing the Q value when that corresponding action didn't take it to the goal **(that specific action)**. 

Moreover, the Q table was initialized to be all values of a small and identical number. Initially I used randomized small numbers across the table, however, that reduces the performance of the program. I reckon this is because when initialized with random numbers, some actions are already superior than others (if it has a higher random number). However, if the superior actions were the undesired actions, then the agent would take extra time to reduce the Q values of the undesired superior actions, and therefore reduces the performance. I believe an uniform initial Q table introduces more randomness than the randomized one, because in the beginning of the program, the agent treats all actions as the same and therefore could freely explore the options. Lastly, given π-greedy policy which brings intense exploration in the beginning was used, it already serve the purpose of randomization. As the agent slowly transition from exploration to exploitation, the Q table is already refined and ready to be exploit. To further prove my point, this is the resulting graph of QL on FrozenLake-v0 using randomized initial Q table (can be found in `./src/graphs/QL_Random_Frozen.png`):

![QL_Random_Frozen](/Users/zzzh/Documents/University of Waterloo/Winter2022[4B]/ece499/src/graphs/QL_Random_Frozen.png)

As illustrated from the graph, the program has a low success rate in the beginning compare to an uniform initial Q table and the overall success rate is at 78.6% (a 10% decrease compare to an uniform one).

##### Assesment of the results

Given the result presented in graphs, QL is slightly better than SARSA for FrozenLake-v0, however, the results are still at least 80% successful. I believe these two algorithms are an overkill for this particular environment, and that QL is better due to an implementation difference. The difference is at the `train` method of the agent, in the QL impmentation, when an episode terminates due to stepping on a hole, the corresponding update function becomes:

```pseudocode
Q(S, A) ← Q(S, A) + α[R − Q(S, A)]
```

Given stepping on a hole gives a reward of 0, this equation decreases the Q value at state `S` taking action `A`. As a result, in the future iterations, this step is less likely to be chosen and therefore avoids the hole. On the other hand, SARSA's update function remains the same regardless of stepping in a hole or not:

```pseudocode
Q(S, A) ← Q(S, A) + α[R + γQ(S′, A′) − Q(S, A)]
```

## Reflections

### Challenges

Some of the challenges I encountered are:

- Learning OpenAI from scratch, it was difficult to understand the observation object at first, however, I built an understanding watching demostrations videos online. Furthermore, reading other domains from OpenAI also helped.
- I had difficulties understanding the differences between on-policy and off-policy, as the notation on the documentation I read were confusing (this was explained earlier). I was finally able to grasp the idea by finding someone pointing out the same problem online.

### Lesson learned

- As self learner, I believe internet is your biggest ally. There were a few post from StackOverflow, geeksforgeeks that solved a lot of my problems.
- Learning to read documentation was an important aspect in this course. This includes documentation of OpenAI environments, and books regarding RL. To be more specific, OpenAI provides not only their source code, but a wiki page from built in github. This was something I didn't know, and I was surprised how well it was documented. 
- Writting report using Markdown was another interesting thing from this course. After being introduced by Prof. Crowley about Markdown and Typora, I instantly felt that it is way better for engineering students to write reports on Markdown rather than Words. For example, Markdown supports various languages when citing codes, which allows readers to easily read and understand codes. 

## REFERENCES

Barto, A. G. (2007). *Temporal difference learning*. Scholarpedia. Retrieved April 8, 2022, from http://www.scholarpedia.org/article/Temporal_difference_learning 

OpenAI. (2022). *A toolkit for developing and comparing reinforcement learning algorithms*. Gym. Retrieved April 4, 2022, from https://gym.openai.com/docs/ 

*Q-learning in python*. GeeksforGeeks. (2021, November 9). Retrieved April 9, 2022, from https://www.geeksforgeeks.org/q-learning-in-python/ 

*Sarsa Reinforcement Learning*. GeeksforGeeks. (2021, June 24). Retrieved April 9, 2022, from https://www.geeksforgeeks.org/sarsa-reinforcement-learning/ 

*Sarsa Reinforcement Learning*. GeeksforGeeks. (2021, June 24). Retrieved April 9, 2022, from https://www.geeksforgeeks.org/sarsa-reinforcement-learning/ 

Sutton, R. S., & Barto, A. G. (2020). *Reinforcement learning: An introduction*. The MIT Press. 

TheComputerScientist. (2018, October 16). *An introduction to Q-learning - youtube*. Retrieved April 9, 2022, from https://www.youtube.com/watch?v=wN3rxIKmMgE 

Venkatachalam, M. (2021, July 20). *Q-learning - an introduction through a simple table based implementation with learning rate, discount factor and exploration*. gotensor. Retrieved April 9, 2022, from https://gotensor.com/2019/10/02/q-learning-an-introduction-through-a-simple-table-based-implementation-with-learning-rate-discount-factor-and-exploration/#:~:text=The%20learning%20rate%20as%20explained,the%20importance%20of%20future%20rewards. 





