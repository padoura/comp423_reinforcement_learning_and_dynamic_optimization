''' Q Learning Algorithm vs Threshold Agent
'''

from env import Env
# from human_agent import HumanAgent
from q_learning_agent import QLearningAgent
from random_agent import RandomAgent
from threshold_agent import ThresholdAgent
import json
import os

# q_trained_model = None
# print("Loading pretrained model...")
# with open(os.path.dirname(os.path.abspath(__file__))+'\\q_threshold_model.json') as json_file:
#     q_trained_model = json.load(json_file)

# Make environment
env = Env()
# human_agent1 = HumanAgent(env.num_actions)
# human_agent2 = HumanAgent(env.num_actions)
random_agent = RandomAgent(env.np_random, False)
threshold_agent = ThresholdAgent(False, False)
print("Running Q Learning algorithm for Threshold Agent...")
q_learning_agent = QLearningAgent(env.np_random, False, is_learning = True)
env.set_agents([
    # human_agent1,
    # human_agent2,
    # threshold_agent,
    # pi_random_agent,
    q_learning_agent,
    random_agent,
    # threshold_agent,
])

num_of_games = 100000
q_learning_agent_payoffs = []
print("Running ", num_of_games, " games \"Q Learning Agent vs Threshold Agent\"...")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"% complete\r", end="")

    trajectories, payoffs = env.run(is_training=False)
    q_learning_agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(q_learning_agent_payoffs)/num_of_games)

# print("Storing trained model...")

num_of_games = 100000
q_learning_agent_payoffs = []
print("Running ", num_of_games, " games \"Q Learning Agent vs Threshold Agent\"...")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"% complete\r", end="")

    trajectories, payoffs = env.run(is_training=False)
    q_learning_agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(q_learning_agent_payoffs)/num_of_games)

num_of_games = 100000
q_learning_agent_payoffs = []
print("Running ", num_of_games, " games \"Q Learning Agent vs Threshold Agent\"...")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"% complete\r", end="")

    trajectories, payoffs = env.run(is_training=False)
    q_learning_agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(q_learning_agent_payoffs)/num_of_games)

num_of_games = 100000
q_learning_agent_payoffs = []
print("Running ", num_of_games, " games \"Q Learning Agent vs Threshold Agent\"...")
for i in range(num_of_games):
    print(round(i/num_of_games*100, 1),"% complete\r", end="")

    trajectories, payoffs = env.run(is_training=False)
    q_learning_agent_payoffs.append(payoffs[0])

print("\nAverage payoffs:  ", sum(q_learning_agent_payoffs)/num_of_games)

with open(os.path.dirname(os.path.abspath(__file__))+'\\q_test.json', 'w') as json_file:
    json.dump(q_learning_agent.model, json_file, indent=4, sort_keys=True)
print("Stored trained model successfully!")

with open(os.path.dirname(os.path.abspath(__file__))+'\\q_payoffs_test.json', 'w') as json_file:
    json.dump(q_learning_agent_payoffs, json_file, indent=4, sort_keys=True)

print("Stored trained model payoff evolution!")