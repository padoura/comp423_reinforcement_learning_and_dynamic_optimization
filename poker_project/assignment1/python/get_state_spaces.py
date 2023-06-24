import json
import os
from env import Env
from random_agent import RandomAgent
from threshold_agent import ThresholdAgent

with open(os.path.dirname(os.path.abspath(__file__))+'\\win_probabilities.json') as json_file:
    win_probabilities = json.load(json_file)

with open(os.path.dirname(os.path.abspath(__file__))+'\\loss_probabilities.json') as json_file:
    loss_probabilities = json.load(json_file)

with open(os.path.dirname(os.path.abspath(__file__))+'\\flop_probabilities.json') as json_file:
    flop_probabilities = json.load(json_file)

with open(os.path.dirname(os.path.abspath(__file__))+'\\range_probabilities.json') as json_file:
    range_probabilities = json.load(json_file)

env = Env()
random_agent = RandomAgent(env.np_random, True)
threshold_agent = ThresholdAgent(True)

state_space = threshold_agent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
print("Threshold Agent:")
print("len(state_space) = ", len(state_space))
print("len(state_space[]) = ", sum(len(v) for v in state_space.values()))
with open(os.path.dirname(os.path.abspath(__file__))+'\\threshold_agent_state_space.json', "w") as write_file:
    json.dump(state_space, write_file, indent=4, sort_keys=True)

state_space = random_agent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
print("Random Agent:")
print("len(state_space) = ", len(state_space))
print("len(state_space[]) = ", sum(len(v) for v in state_space.values()))
with open(os.path.dirname(os.path.abspath(__file__))+'\\random_agent_state_space.json', "w") as write_file:
    json.dump(state_space, write_file, indent=4, sort_keys=True)
