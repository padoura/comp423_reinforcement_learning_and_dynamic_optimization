import json
from random_agent import RandomAgent
from threshold_agent import ThresholdAgent

with open('poker_project//assignment1//python//win_probabilities.json') as json_file:
    win_probabilities = json.load(json_file)

with open('poker_project//assignment1//python//loss_probabilities.json') as json_file:
    loss_probabilities = json.load(json_file)

with open('poker_project//assignment1//python//flop_probabilities.json') as json_file:
    flop_probabilities = json.load(json_file)

with open('poker_project//assignment1//python//range_probabilities.json') as json_file:
    range_probabilities = json.load(json_file)

state_space = ThresholdAgent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
print("Threshold Agent:")
print("len(state_space) = ", len(state_space))
print("len(state_space[]) = ", sum(len(v) for v in state_space.values()))
with open("poker_project//assignment1//python//threshold_agent_state_space.json", "w") as write_file:
    json.dump(state_space, write_file, indent=4, sort_keys=True)

state_space = RandomAgent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
print("Random Agent:")
print("len(state_space) = ", len(state_space))
print("len(state_space[]) = ", sum(len(v) for v in state_space.values()))
with open("poker_project//assignment1//python//random_agent_state_space.json", "w") as write_file:
    json.dump(state_space, write_file, indent=4, sort_keys=True)
