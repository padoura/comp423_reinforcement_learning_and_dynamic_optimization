''' A toy example of playing simplified heads up limit hold'em
'''

from env import Env
from card import Card
from human_agent import HumanAgent
from q_learning_agent import QLearningAgent
from policy_iteration_agent import PolicyIterationAgent
from random_agent import RandomAgent
from threshold_agent import ThresholdAgent
import json
import os


with open(os.path.dirname(os.path.abspath(__file__))+'\\q_threshold_model.json') as json_file:
    q_trained_model = json.load(json_file)

print(len(q_trained_model['Q']))

# Make environment
env = Env()
# human_agent2 = HumanAgent(env.num_actions)
random_agent = RandomAgent(env.np_random, True)
threshold_agent = ThresholdAgent(True)
pi_random_agent = PolicyIterationAgent(env.np_random, True, random_agent)
pi_threshold_agent = PolicyIterationAgent(env.np_random, False, threshold_agent)
q_learning_agent = QLearningAgent(env.np_random, False, q_trained_model, is_learning = True)
human_agent1 = HumanAgent(env.num_actions, q_learning_agent)
env.set_agents([
    human_agent1,
    # human_agent2,
    q_learning_agent
    # pi_threshold_agent,
    # threshold_agent,
    # pi_random_agent,
    # random_agent,
])

print(">> Simplified Hold'em model")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    print('>> Player', action_record[-1][0], 'chooses', action_record[-1][1])

    # Let's take a look at what Player 1 card is
    print('===============     Hand of Player 1    ===============')
    Card.print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('Player 0 won {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('Player 0 lost {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
