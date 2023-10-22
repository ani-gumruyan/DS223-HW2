# DS223-HW2
DS223 - Marketing Analytics | Homework 2 - A/B Testing

**Scenario**

We have four advertisement options (bandits), and your task is to design an experiment using Epsilon Greedy and Thompson Sampling.

**Design of Experiment**

Bandit_Reward: [1,2,3,4] 

NumberOfTrials: 20000

- Create a Bandit Class
- Create EpsilonGreedy() and ThompsonSampling() classes and methods (inherited
from Bandit()).

Epsilon-greedy:
  1) decay epsilon by 1/t
  2) design the experiment 

Thompson Sampling
  1) design with known precision 
  2) design the experiment

Report:
  1) Visualize the learning process for each algorithm (plot1())
  2) Visualize cumulative rewards from E-Greedy and Thompson Sampling. 
  3) Store the rewards in a CSV file ({Bandit, Reward, Algorithm})
  4) Print cumulative reward
  5) Print cumulative regret
