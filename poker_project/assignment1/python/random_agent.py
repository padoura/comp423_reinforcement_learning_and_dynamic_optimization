from dealer import Dealer
from utils import try_key_initialization
class RandomAgent:
    ''' A random agent for benchmarking purposes
    '''

    def __init__(self, np_random, print_enabled):
        ''' Initilize the random agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.np_random = np_random
        self.use_raw = True
        self.print_enabled = print_enabled # TODO: obsolete, to be deleted

    def step(self, state):
        ''' Completely random agent

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The randomly chosen action
        '''
        if self.print_enabled: self._print_state(state['raw_obs'], state['action_record'])

        action = self.np_random.randint(0, len(state['raw_legal_actions']))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}

    def _print_state(self, state, action_record):
        ''' Print out the state

        Args:
            state (dict): A dictionary of the raw state
            action_record (list): A list of the historical actions
        '''
        if len(action_record) > 0:
            print('>> Player', action_record[-1][0], 'chooses', action_record[-1][1])

        # print('\n=============== Community Card ===============')
        # Card.print_card(state['public_cards'])
        # print('===============   Your Hand    ===============')
        # Card.print_card(state['hand'])
        print('===============     Chips      ===============')
        for i in range(len(state['all_chips'])):
            if i == state['current_player']:
                print('Random Agent (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions Random Agent Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')
    
    @staticmethod
    def calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities):


        # | Index            | # Enum |Meaning                                                                        |
        # | -----------------|--------|-------------------------------------------------------------------------------|
        # | 'position'       |   2    |Position of player 'first'/'second'                                            |
        # | 'my_chips'       |   5    |Chips placed by our agent so far                                               |
        # | 'other_chips'    |   3    |Difference in chips placed between adversary and our agent so far              |
        # | 'hand'           |   5    |Rank of hand: T ~ A as first public card                                       |
        # | 'public_cards'   |   16   |Rank of public cards in alphabetical order e.g. 'AK' or 'none' if not shown yet|
        # | 'opponent_range' |   <22  |Possible range of opponent's hand, meaningful for ThresholdAgent, else 'AJKQT' |

        # positions = {'first', 'second'}
        # chips = {'0.5', '1.5', '2.5', '3.5', '4.5'}
        # public_cards_set = {'none'}
        # for public_card1 in Dealer.RANK_LIST:
        #     for public_hand2 in Dealer.RANK_LIST:
        #         public_cards_set.add(''.join(sorted(public_card1 + public_hand2)))

        # legal_action_sequences = Judger.get_legal_sequences_of_actions()

        state_space = {}

        ############## position == 'first' #################
        # preflop @ chips [0.5, 0.5]
        # flop @ chips [0.5, 0.5], [1.5, 1.5], [2.5, 2.5]
        # my_legal_actions = ['bet', 'check']
        ############## position == 'second' #################
        # preflop @ chips [0.5, 0.5]
        # flop @ chips [0.5, 0.5]
        # my_legal_actions = ['raise', 'check']
        ############## position == 'first' #################
        # preflop @ chips [0.5, 1.5] or [1.5, 2.5]
        # flop @ chips [0.5, 1.5] or [1.5, 2.5] or [2.5, 3.5] or [3.5, 4.5]
        # my_legal_actions = ['fold', 'bet']
        ############## position == 'second' #################
        # preflop @ chips [1.5, 0.5]
        # flop @ chips [1.5, 0.5], [2.5, 1.5], [3.5, 2.5]
        # my_legal_actions = ['raise', 'bet', 'fold']
        for position in ['first', 'second']:
            for game_round in [1, 2]:
                for other_chips in [0, 1]:
                    if position == 'first' and other_chips == 1: 
                        my_starting_chips = [0.5, 1.5, 2.5, 3.5] if game_round == 2 else [0.5, 1.5]
                    else:
                        my_starting_chips = [0.5, 1.5, 2.5] if game_round == 2 else [0.5]
                    if other_chips == 0:
                        my_legal_actions = ['bet', 'check'] if position == 'first' else ['raise', 'check']
                    else: # other_chips = 1
                        my_legal_actions = ['fold', 'bet'] if position == 'first' else ['raise', 'bet', 'fold']
                    for my_chips in my_starting_chips:
                        for my_action in my_legal_actions:
                            RandomAgent._calculate_round_states(state_space, position, my_chips, other_chips, my_action, win_probabilities, loss_probabilities, flop_probabilities, game_round)

        return state_space
    

    @staticmethod
    def _calculate_round_states(state_space, position, my_chips, other_chips, my_action, win_probabilities, loss_probabilities, flop_probabilities, game_round):
        for hand in Dealer.RANK_LIST:
            key = position + '_' + str(my_chips) + '_' + str(other_chips) + '_' + hand + '_'
            if (position == 'first' and other_chips == 0 and my_action == 'bet') or (position == 'second' and my_action == 'raise'):
                new_my_chips = my_chips + 1 + other_chips
                action_prob = 1/3 if position == 'first' else 1/2 # first position -> 'fold', 'raise', 'bet', second position -> 'fold', 'bet'
                #### other_action == 'fold' ####
                is_terminal = True
                new_other_chips = -1
                reward = my_chips + other_chips
                RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                #### other_action == 'bet' ####
                is_terminal = False
                new_other_chips = 0
                reward = 0
                RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                if position == 'first':
                    #### other_action == 'raise' ####
                    is_terminal = False
                    new_other_chips = 1
                    reward = 0
                    RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (other_chips == 1 and my_action == 'bet'):
                new_my_chips = my_chips + 1
                action_prob = 1 # random agent has finished his move by raising
                is_terminal = False
                new_other_chips = 0
                reward = 0
                RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (my_action == 'fold'):
                new_my_chips = my_chips
                action_prob = 1 # random agent has finished his move by raising
                is_terminal = True
                new_other_chips = 1
                reward = -my_chips
                RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (my_action == 'check'):
                new_my_chips = my_chips
                action_prob = 0.5 if position == 'first' else 1 # first position -> randomly between 'check', 'raise', second position -> round done
                #### other_action == 'check' ####
                is_terminal = False
                new_other_chips = 0
                reward = 0
                RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                if position == 'first':
                    #### other_action == 'raise' ####
                    is_terminal = False
                    new_other_chips = 1
                    reward = 0
                    RandomAgent._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)

    @staticmethod
    def _calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round):
        if game_round == 1: # end of round 1
            full_key = key + 'none' + '_AJKQT'
            if new_other_chips != 0:
                public_cards = 'none'
                RandomAgent._add_or_update_key(state_space, full_key, action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
            else:
                for public_cards in flop_probabilities[hand]['AJKQT']:
                    RandomAgent._add_or_update_key(state_space, full_key, flop_probabilities[hand]['AJKQT'][public_cards]*action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
        else: # end of round 2
            for public_cards in flop_probabilities[hand]['AJKQT']:
                full_key = key + public_cards + '_AJKQT'
                if new_other_chips == 0:
                    # game finished, result based on players' hands 
                    is_terminal = True
                    win_prob = win_probabilities[hand][public_cards]['AJKQT']
                    loss_prob = loss_probabilities[hand][public_cards]['AJKQT']
                    tie_prob = 1 - win_prob - loss_prob
                    for result_prob in [win_prob, loss_prob, tie_prob]:
                        reward = new_my_chips if result_prob == win_prob else -new_my_chips if result_prob == loss_prob else 0
                        RandomAgent._add_or_update_key(state_space, full_key, result_prob*action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
                else:
                    RandomAgent._add_or_update_key(state_space, full_key, action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)

    @staticmethod
    def _add_or_update_key(state_space, key, prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards):
        if prob > 0:
            try_key_initialization(state_space, key, {})
            try_key_initialization(state_space[key], my_action, [])
            new_key = position + '_' + str(new_my_chips) + '_' + str(new_other_chips) + '_' + hand + '_' + public_cards + '_AJKQT'
            state_space[key][my_action].append( (prob, new_key, reward, is_terminal)  )

# import json

# with open('poker_project//assignment1//python//win_probabilities.json') as json_file:
#     win_probabilities = json.load(json_file)

# with open('poker_project//assignment1//python//loss_probabilities.json') as json_file:
#     loss_probabilities = json.load(json_file)

# with open('poker_project//assignment1//python//flop_probabilities.json') as json_file:
#     flop_probabilities = json.load(json_file)

# with open('poker_project//assignment1//python//range_probabilities.json') as json_file:
#     range_probabilities = json.load(json_file)

# state_space = RandomAgent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
# print("len(state_space) = ", len(state_space))
# print("len(state_space[]) = ", sum(len(v) for v in state_space.values()))
# with open("poker_project//assignment1//python//random_agent_state_space.json", "w") as write_file:
#     json.dump(state_space, write_file, indent=4, sort_keys=True)
