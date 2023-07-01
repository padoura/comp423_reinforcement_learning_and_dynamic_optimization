''' Python script I used for understanding and manual tuning of Q Learning hyperparameters
'''

''' Training Q Learning Algorithm vs Threshold Agent & Analyzing Hyperparameters
'''

from env import Env
from q_learning_agent import QLearningAgent
from threshold_agent import ThresholdAgent
import json
import time
import numpy as npy


start_time = time.time()

initial_epsilon_values = [1.0]
initial_alpha_values = [0.1]
epsilon_decay_values = [-1/8]
alpha_decay_values = [-1/4]
hyperparam_set_name = ['q_threshold' + '_' + str(initial_epsilon_values[0]) + '_' + str(initial_alpha_values[0]) + '_' + str(epsilon_decay_values[0]) + '_' + str(alpha_decay_values[0])]
test_average_payoffs = [0.0]*len(initial_epsilon_values)
test_std_payoffs = [0.0]*len(initial_epsilon_values)

with open('random_agent_state_space.json') as json_file:
    unknown_agent_state_space = json.load(json_file)

print("Tuning Q Learning algorithm for Threshold Agent...")
for hyp_ind in range(len(initial_alpha_values)):
    # Make environment
    env = Env()
    threshold_agent = ThresholdAgent(False, agent_model_is_known = False)
    q_learning_agent = QLearningAgent(
        env.np_random, 
        False, 
        is_learning = True, 
        initial_epsilon = initial_epsilon_values[hyp_ind], 
        initial_alpha = initial_alpha_values[hyp_ind], 
        epsilon_decay = epsilon_decay_values[hyp_ind], 
        alpha_decay = alpha_decay_values[hyp_ind],
        state_space = unknown_agent_state_space
    )
    env.set_agents([
        q_learning_agent,
        threshold_agent,
    ])

    agent_payoffs = []
    num_of_games = 3*10**6
    print("Training session for hyperparameter set: ", hyperparam_set_name[hyp_ind])
    print("Progress (%)")
    for i in range(num_of_games):
        print(round(i/num_of_games*100, 1),"\r", end="")

        trajectories, payoffs = env.run()
        agent_payoffs.append(payoffs[0])

    print("Storing instance...")

    with open(hyperparam_set_name[hyp_ind] + '_threshold_model.json', 'w') as json_file:
        json.dump(q_learning_agent.model, json_file, indent=4, sort_keys=True)

    with open(hyperparam_set_name[hyp_ind] + '_threshold_payoffs_train.json', 'w') as json_file:
        json.dump(agent_payoffs, json_file, indent=4, sort_keys=True)


    q_learning_agent.is_learning = False
    
    agent_payoffs = []
    num_of_games = 2*10**6
    print("Testing session number for hyperparameter set: ", hyperparam_set_name[hyp_ind])
    print("Progress (%)")
    for i in range(num_of_games):
        print(round(i/num_of_games*100, 1),"\r", end="")

        trajectories, payoffs = env.run()
        agent_payoffs.append(payoffs[0])


    print("Storing instance...")

    with open(hyperparam_set_name[hyp_ind] + '_threshold_payoffs_test.json', 'w') as json_file:
        json.dump(agent_payoffs, json_file, indent=4, sort_keys=True)
    
    test_average_payoffs[hyp_ind] = npy.mean(agent_payoffs)
    test_std_payoffs[hyp_ind] = npy.std(agent_payoffs)
    print("Testing average payoffs for hyperparameter set ", hyperparam_set_name[hyp_ind], ": ", test_average_payoffs[hyp_ind])
    print("Testing standard deviation for hyperparameter set ", hyperparam_set_name[hyp_ind], ": ", test_std_payoffs[hyp_ind])

end_time = time.time()
print("Total time elapsed for code snippet: ",  end_time - start_time, " seconds")

###########################################################
# Figure Utilities
###########################################################

import matplotlib.pyplot as plt
from utils import get_moving_average
import json
import os
from IPython import display
import time
start_time = time.time()

SAVE_FIGURES = False # save figures for latex
DISPLAY_FIGURES = True # display figures in notebook
saved_figures = []

class Figure:
    def __init__(self, title, xlabel, ylabel, x, ys, legends, filename):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for y in ys:
            plt.plot(x,y)
        plt.legend(legends, loc = "best", frameon = False)
        self.filename = filename
        self.save_figure_or_continue()
    def save_figure_or_continue(self):
        plt.show
        if DISPLAY_FIGURES: 
            plt.savefig(self.filename + ".png", bbox_inches = "tight")
            saved_figures.append(self.filename + ".png")
        if SAVE_FIGURES: plt.savefig("../latex/figures/" + self.filename + ".png", bbox_inches = "tight")
        plt.close()

def display_figures():
    """ A utility to display saved figures inside this notebook
    """
    filelist=os.listdir()
    for file in saved_figures:
        display.display(display.Image(file))

def delete_figures():
    """ A utility to delete saved figures
    """
    for file in saved_figures:
        os.remove(file)

# ###########################################################
# # Figure generation
# ###########################################################


# hyperparam_set_name = ['q_high_varying', 'q_slower_alpha', 'q_slower_epsilon', 'q_low_varying']

# with open('q_high_varying_random_policy_evolution.json') as json_file:
#     q_high_varying_policy_evolution = json.load(json_file)

# with open('q_slower_alpha_random_policy_evolution.json') as json_file:
#     q_slower_alpha_policy_evolution = json.load(json_file)

# with open('q_slower_epsilon_random_policy_evolution.json') as json_file:
#     q_slower_epsilon_policy_evolution = json.load(json_file)

# with open('q_low_varying_random_policy_evolution.json') as json_file:
#     q_low_varying_policy_evolution = json.load(json_file)

    
# num_of_games = len(q_high_varying_policy_evolution) # <---- num_of_games of blocks above should be the same for all experiments in order for this code to finish figures successfully

# Figure(
#     title = "Q Learning Policy Convergence Rate vs Random Agent",
#     xlabel = "Episode t",
#     ylabel = "Same policy per state (%)",
#     x = range(1,num_of_games+1),
#     ys = [q_high_varying_policy_evolution, q_slower_alpha_policy_evolution, q_slower_epsilon_policy_evolution, q_low_varying_policy_evolution],
#     legends = tuple(hyperparam_set_name),
#     filename = "convergence_rate"
# )

# window_size = int(num_of_games/10)  # <---- CHANGE WINDOW SIZE HERE

# print("Calculating q_random_moving_averages...")

# with open('q_high_varying_random_payoffs_train.json') as json_file:
#     q_high_varying_random_payoffs = json.load(json_file)

# with open('q_slower_alpha_random_payoffs_train.json') as json_file:
#     q_slower_alpha_random_payoffs = json.load(json_file)

# with open('q_slower_epsilon_random_payoffs_train.json') as json_file:
#     q_slower_epsilon_random_payoffs = json.load(json_file)

# with open('q_low_varying_random_payoffs_train.json') as json_file:
#     q_low_varying_random_payoffs = json.load(json_file)

# q_high_varying_random_moving_averages = get_moving_average(q_high_varying_random_payoffs, window_size)
# q_slower_alpha_random_moving_averages = get_moving_average(q_slower_alpha_random_payoffs, window_size)
# q_slower_epsilon_random_moving_averages = get_moving_average(q_slower_epsilon_random_payoffs, window_size)
# q_low_varying_random_moving_averages = get_moving_average(q_low_varying_random_payoffs, window_size)

# with open('pi_random_payoffs.json') as json_file:
#     pi_random_payoffs = json.load(json_file)

# random_optimal_mean = sum(pi_random_payoffs)/len(pi_random_payoffs)

# Figure(
#     title = "Moving Average (per " + str(window_size) + " games) vs Random Agent",
#     xlabel = "Episode t",
#     ylabel = "Payoffs per game",
#     x = range(window_size,len(q_high_varying_random_moving_averages)+window_size),
#     ys = [q_high_varying_random_moving_averages, q_slower_alpha_random_moving_averages, q_slower_epsilon_random_moving_averages, q_low_varying_random_moving_averages, [random_optimal_mean]*len(q_high_varying_random_moving_averages)],
#     legends = ('q_high_varying', 'q_slower_alpha', 'q_slower_epsilon', 'q_low_varying', "mean optimal"),
#     filename = "moving_averages_random"
# )

print("Calculating q_threshold_moving_averages...")

threshold_payoffs = {}

for name in hyperparam_set_name:
    with open(name + '_threshold_payoffs_train.json') as json_file:
        threshold_payoffs[name] = json.load(json_file)

num_of_games = len(threshold_payoffs[hyperparam_set_name[0]]) # <---- num_of_games of blocks above should be the same for all experiments in order for this code to finish figures successfully
window_size = 10**5  # <---- CHANGE WINDOW SIZE HERE

threshold_moving_averages = {}

for name in hyperparam_set_name:
    threshold_moving_averages[name] = get_moving_average(threshold_payoffs[name], window_size)

with open('pi_threshold_payoffs.json') as json_file:
    pi_threshold_payoffs = json.load(json_file)

threshold_optimal_mean = sum(pi_threshold_payoffs)/len(pi_threshold_payoffs)

# ys = [ threshold_moving_averages[name] for name in threshold_moving_averages ]
# ys.append([threshold_optimal_mean]*len(threshold_moving_averages[hyperparam_set_name[0]]))

legends = hyperparam_set_name[:]
legends.append('mean optimal')

Figure(
    title = "Moving Average (per " + str(window_size) + " games) vs Threshold Agent",
    xlabel = "Episode t",
    ylabel = "Payoffs per game",
    x = range(window_size,len(threshold_moving_averages[hyperparam_set_name[0]])+window_size),
    ys = [threshold_moving_averages[hyperparam_set_name[0]], [threshold_optimal_mean]*len(threshold_moving_averages[hyperparam_set_name[0]])],
    legends = legends,
    filename = "moving_averages_threshold" + hyperparam_set_name[0]
)

end_time = time.time()
print("Total time elapsed for code snippet: ",  end_time - start_time, " seconds")

for name in hyperparam_set_name:
    with open(name + '_threshold_payoffs_test.json') as json_file:
        threshold_payoffs[name] = json.load(json_file)
        print(npy.mean(threshold_payoffs[name]))

