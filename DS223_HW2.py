from Bandit import *

Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000


def comparison():
    e_greedy = EpsilonGreedy(Bandit_Reward)
    thompson = ThompsonSampling(Bandit_Reward)

    e_greedy_rewards = e_greedy.experiment(NumberOfTrials)
    thompson_rewards = thompson.experiment(NumberOfTrials)

    # Save to CSV
    data = []
    for i, reward in enumerate(e_greedy_rewards):
        data.append([i, reward, "Epsilon-Greedy"])
    for i, reward in enumerate(thompson_rewards):
        data.append([i, reward, "Thompson Sampling"])

    Visualization().save_to_csv(data)

    # Visualization
    viz = Visualization()
    viz.plot1(e_greedy_rewards, thompson_rewards)
    viz.plot2(e_greedy_rewards, thompson_rewards,
              Bandit_Reward, NumberOfTrials)

    # Report
    e_greedy.report()
    thompson.report()


comparison()


# BONUS: Suggest better implementation plan

"""

The observation of the obtained result made me realise that while EpsilonGreedy is performing quite well and choosing optimal bandit, Thompson Sampling is underperforming. 
Thompson Sampling is designed for stochastic (probability based) contexts, and having determinist rewards might be the obstacle for the algorithm to learn and adapt to the problem effectively.

With this in mind, a better implementation plan might include:
- Changing the nature of rewards (e.g.[1,2,3,4] -> [0.1, 0.2, 0.3, 0.4])
- Playing with 'alpha' and 'beta' variables based on rewards
- Referring to other algorithms (e.g UCB (Upper Confidence Bound) to balance exploration and exploitation)

"""
