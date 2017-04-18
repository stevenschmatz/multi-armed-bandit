'''
multi_armed_bandit.py
@stevenschmatz
18 April 2017
'''


import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


env = gym.make('BanditTenArmedGaussian-v0')
env.reset()


def run_simulation(get_action, num_trials=1000):
    '''Runs a 10-armed bandit simulation for multiple trials.
    
    Args:
    - get_action: a function which has input (t, means, count) and returns action
    - num_trials: number of iterations

    Returns:
    - A list of regret versus time 
    '''

    means = np.zeros(10)
    count = np.zeros(10)
    choices = []

    for t in range(num_trials):

        action = get_action(t, means, count)
        observation, reward, done, info = env.step(action)

        # Keep track of sample means for exploitation, and choices for regret calculation 
        count[action] += 1
        means[action] = (1 - 1/count[action]) * means[action] + (1/count[action]) * reward
        choices.append(action)

    def regret(t):
        best = np.argmax(means)
        return means[best] * t - sum([means[choices[i]] for i in range(t)])

    return [regret(t) for t in range(num_trials)]


def epsilon_greedy_action(t, means, count, epsilon=0.5):
    '''epsilon percent of the time, choose a random action.
    (1-epsilon) percent of the time, exploit by choosing the action
    with the highest mean reward.
    
    This is a suboptimal solution, achieving linear regret. 
    '''

    explore = np.random.uniform() < epsilon

    if explore:
        return env.action_space.sample()
    else:
        return np.argmax(means)


def upper_confidence_bounds_action(t, means, count, epsilon=0.0):
    '''Play each arm once, then choose according to the equation given
    by Auer, Cesa-Bianchi & Fisher (2002).
    
    This is said to achieve the most optimal solution, with logarithmic regret.
    '''

    if t < 10:
        return t
    else:
        return np.argmax(means + np.sqrt(2 * np.log(t) / count))


def main():
    '''Compares the epsilon-greedy approach to the upper confidence bounds
    approach for solving the multi-armed bandit problem.'''

    results_eps = run_simulation(epsilon_greedy_action)
    results_ucb = run_simulation(upper_confidence_bounds_action)

    plt.plot(results_eps, color='red')
    plt.plot(results_ucb, color='blue')
    plt.xlabel("Timestep")
    plt.ylabel("Regret")
    plt.title("Gaussian 10-armed-bandit regret vs time")
    greedy_patch = mpatches.Patch(color='red', label='epsilon-greedy')
    ucb_patch = mpatches.Patch(color='blue', label='upper confidence bounds')
    plt.legend(handles=[greedy_patch, ucb_patch])
    plt.show()


if __name__ == "__main__":
    main()
