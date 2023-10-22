"""
  Run this file at first, in order to see what is it printng. 
  Instead of the print() use the respective log level.
"""
import csv
import random
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

# LOGGER
from abc import ABC, abstractmethod
from logs import *

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self, e_greedy_rewards, thompson_rewards):
        # Visualize the learning process for each algorithm (plot1())

        # Linear Plot
        plt.subplot(1, 2, 1)
        plt.plot(e_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.title('Cumulative Rewards (Linear Scale)')
        plt.xlabel('Trials')
        plt.ylabel('Total Reward')
        plt.legend()

        # Log Plot
        plt.subplot(1, 2, 2)
        plt.plot(e_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.yscale('log')
        plt.title('Cumulative Rewards (Log Scale)')
        plt.xlabel('Trials')
        plt.ylabel('Total Reward')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, e_greedy_rewards, thompson_rewards, Bandit_Reward, NumberOfTrials):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        e_greedy_regrets = max(Bandit_Reward) * np.arange(1,
                                                          NumberOfTrials+1) - np.cumsum(e_greedy_rewards)
        thompson_regrets = max(Bandit_Reward) * np.arange(1,
                                                          NumberOfTrials+1) - np.cumsum(thompson_rewards)

        plt.plot(e_greedy_regrets, label="Epsilon-Greedy Regret")
        plt.plot(thompson_regrets, label="Thompson Sampling Regret")
        plt.title('Cumulative Regrets')
        plt.xlabel('Trials')
        plt.ylabel('Total Regret')
        plt.legend()
        plt.show()

    @staticmethod
    def save_to_csv(data, filename="rewards.csv"):
        df = pd.DataFrame(data, columns=["Bandit", "Reward", "Algorithm"])
        df.to_csv(filename, index=False)

#--------------------------------------#


class EpsilonGreedy(Bandit):

    def __init__(self, p):
        self.p = p  # probabilities of bandit's reward
        self.n = len(p)
        self.k = np.zeros(self.n)  # number of times arm was pulled
        self.reward = np.zeros(self.n)  # sum of rewards for each arm
        self.t = 0

    def __repr__(self):
        return f"EpsilonGreedy({self.p})"

    def pull(self):
        epsilon = 1 / (1 + self.t)
        if random.random() < epsilon:
            chosen_bandit = random.choice(range(self.n))
        else:
            chosen_bandit = np.argmax(self.reward / (self.k + 1e-5))  # exploit
        return chosen_bandit

    def update(self, chosen_bandit):
        self.t += 1
        self.k[chosen_bandit] += 1
        reward = self.p[chosen_bandit]
        self.reward[chosen_bandit] += reward
        return reward

    def experiment(self, NumberOfTrials):
        cumulative_rewards = []
        total_reward = 0
        for _ in range(NumberOfTrials):
            chosen_bandit = self.pull()
            reward = self.update(chosen_bandit)
            total_reward += reward
            cumulative_rewards.append(total_reward)
        return cumulative_rewards

    def report(self):
        avg_reward = np.sum(self.reward) / self.t
        optimal_reward = max(self.p) * self.t
        avg_regret = optimal_reward - np.sum(self.reward)
        # Saving to csv can be done using pandas
        print(f"Average Reward for EpsilonGreedy: {avg_reward}")
        print(f"Average Regret for EpsilonGreedy: {avg_regret}")

#--------------------------------------#


class ThompsonSampling(Bandit):

    def __init__(self, p):
        self.p = p
        self.n = len(p)
        self.alpha = np.ones(self.n)
        self.beta = np.ones(self.n)

    def __repr__(self):
        return f"ThompsonSampling({self.p})"

    def pull(self):
        sampled_probs = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        chosen_bandit = np.argmax(sampled_probs)
        return chosen_bandit

    def update(self, chosen_bandit):
        reward = self.p[chosen_bandit]
        self.alpha[chosen_bandit] += reward
        self.beta[chosen_bandit] += max(self.p) - reward
        return reward

    def experiment(self, NumberOfTrials):
        cumulative_rewards = []
        total_reward = 0
        for _ in range(NumberOfTrials):
            chosen_bandit = self.pull()
            reward = self.update(chosen_bandit)
            total_reward += reward
            cumulative_rewards.append(total_reward)
        return cumulative_rewards

    def report(self):
        total_reward = np.sum(self.alpha) - len(self.alpha)
        avg_reward = total_reward / \
            (np.sum(self.alpha) + np.sum(self.beta) - 2 * len(self.alpha))
        optimal_reward = max(self.p) * (np.sum(self.alpha) +
                                        np.sum(self.beta) - 2 * len(self.alpha))
        avg_regret = optimal_reward - total_reward
        # Saving to csv can be done using pandas
        print(f"Average Reward for ThompsonSampling: {avg_reward}")
        print(f"Average Regret for ThompsonSampling: {avg_regret}")


if __name__ == '__main__':

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
