from env import Env
from policy_iteration_agent import PolicyIterationAgent
from random_agent import RandomAgent
from threshold_agent import ThresholdAgent
import json

# Make environment
env = Env()
# human_agent1 = HumanAgent(env.num_actions)
# human_agent2 = HumanAgent(env.num_actions)
random_agent = RandomAgent(env.np_random, False)
threshold_agent = ThresholdAgent(False)
print("Running Policy Iteration algorithm for Random Agent...")
pi_random_agent = PolicyIterationAgent(env.np_random, False, random_agent)
# print("Running Policy Iteration algorithm for Threshold Agent...")
# pi_threshold_agent = PolicyIterationAgent(env.np_random, False, threshold_agent)

with open("poker_project//assignment1//python//random_agent_optimal_policy.json", "w") as write_file:
    json.dump(pi_random_agent.P_opt, write_file, indent=4, sort_keys=True)

# with open("poker_project//assignment1//python//threshold_agent_optimal_policy.json", "w") as write_file:
#     json.dump(pi_threshold_agent.P_opt, write_file, indent=4, sort_keys=True)