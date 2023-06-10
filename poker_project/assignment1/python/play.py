''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''

from env import Env
from card import Card
from human_agent import HumanAgent
from random_agent import RandomAgent

# Make environment
env = Env()
human_agent1 = HumanAgent(env.num_actions)
# human_agent2 = HumanAgent(env.num_actions)
random_agent = RandomAgent(env.num_actions, env.np_random)
env.set_agents([
    human_agent1,
    # human_agent2,
    random_agent,
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
