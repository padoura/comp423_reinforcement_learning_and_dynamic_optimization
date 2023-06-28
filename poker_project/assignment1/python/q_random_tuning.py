''' Python script I used for understanding and manual tuning of Q Learning hyperparameters
'''

###########################################################
# Figure Utilities
###########################################################

import matplotlib.pyplot as plt
from utils import get_moving_average
import json
import os
from IPython import display

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
        saved_figures.remove(file)

''' Q Learning Algorithm vs Random Agent
'''

from env import Env
from q_learning_agent import QLearningAgent
from random_agent import RandomAgent

# Make environment
env = Env()
random_agent = RandomAgent(env.np_random, False)
print("Running Q Learning algorithm for Random Agent...")
pretrained_model = None
with open('q_random_model2.json') as json_file:
    pretrained_model = json.load(json_file)
q_learning_agent = QLearningAgent(env.np_random, False, pretrained_model, is_learning = True, initial_epsilon = 1.0, initial_alpha = 1.0, epsilon_decay = -1/4, alpha_decay = -1/4)
env.set_agents([
    q_learning_agent,
    random_agent,
])

num_of_games = 10**5
window_size = 10**5
agent_payoffs = []
with open('q_random_payoffs2.json') as json_file:
    agent_payoffs = json.load(json_file)
print("Running ", num_of_games, " games \"Q Learning Agent vs Random Agent\"...")
print("Progress (%)")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"\r", end="")

    trajectories, payoffs = env.run()
    agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(agent_payoffs[-num_of_games:])/num_of_games)

print("Running ", num_of_games, " games \"Q Learning Agent vs Random Agent\"...")
print("Progress (%)")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"\r", end="")

    trajectories, payoffs = env.run()
    agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(agent_payoffs[-num_of_games:])/num_of_games)

print("Running ", num_of_games, " games \"Q Learning Agent vs Random Agent\"...")
print("Progress (%)")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"\r", end="")

    trajectories, payoffs = env.run()
    agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(agent_payoffs[-num_of_games:])/num_of_games)

print("Running ", num_of_games, " games \"Q Learning Agent vs Random Agent\"...")
print("Progress (%)")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"\r", end="")

    trajectories, payoffs = env.run()
    agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(agent_payoffs[-num_of_games:])/num_of_games)

q_random_moving_averages = get_moving_average(agent_payoffs, window_size)

Figure(
    title = "Payoffs vs Random Agent",
    xlabel = "Episode t",
    ylabel = "Moving Average (per 100000 games)",
    x = range(window_size,len(q_random_moving_averages)+window_size),
    ys = [q_random_moving_averages],
    legends = (),
    filename = "moving_averages_random2"
)

with open('q_random_model2.json', 'w') as json_file:
    json.dump(q_learning_agent.model, json_file, indent=4, sort_keys=True)
print("Stored trained model successfully!")

with open('q_random_payoffs2.json', 'w') as json_file:
    json.dump(agent_payoffs, json_file, indent=4, sort_keys=True)