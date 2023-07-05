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
        self.print_enabled = print_enabled

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

    def eval_step(self, states, action_history, payoff = None):
        ''' Method only needed for online learning
        '''
        pass

    def _print_state(self, state, action_record):
        ''' Print out the state

        Args:
            state (dict): A dictionary of the raw state
            action_record (list): A list of the historical actions
        '''
        if len(action_record) > 0:
            print('>> Player', action_record[-1][0], 'chooses', action_record[-1][1])

        print('===============     Chips      ===============')
        for i in range(len(state['all_chips'])):
            if i == state['current_player']:
                print('Random Agent (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions Random Agent Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')
    
    def infer_card_range_from_action(self, action, game_round, current_range, other_chips, public_cards, position):
        return 'AJKQT' # range cannot be inferred by agent's actions


    def calculate_state_space(self, win_probabilities, loss_probabilities, flop_probabilities, range_probabilities):
        ''' Calculation of all possible states and their transitions (probability, reward, next state, is terminal state)
        
        See full description of state space representation in env._extract_state()
        
        '''

        state_space = {}
        ################# Possible States ##################

        ############## position == 'first' #################
        # preflop @ chips [0.5, 0.5]
        # flop @ chips [0.5, 0.5], [1.5, 1.5], [2.5, 2.5]
        # my_legal_actions = ['bet', 'check']
        ############## position == 'second' #################
        # preflop @ chips [0.5, 0.5]
        # flop @ chips [0.5, 0.5], [1.5, 1.5], [2.5, 2.5]
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
                    if other_chips == 0: # opponent has not place more chips yet
                        my_legal_actions = ['bet', 'check'] if position == 'first' else ['raise', 'check']
                    else: # other_chips = 1 -> opponent has placed (bet or raise) 1 more chip than us
                        my_legal_actions = ['fold', 'bet'] if position == 'first' else ['raise', 'bet', 'fold']
                    for my_chips in my_starting_chips:
                        for my_action in my_legal_actions:
                            self._calculate_round_states(state_space, position, my_chips, other_chips, my_action, win_probabilities, loss_probabilities, flop_probabilities, game_round)

        return state_space
    

    def _calculate_round_states(self, state_space, position, my_chips, other_chips, my_action, win_probabilities, loss_probabilities, flop_probabilities, game_round):
        ''' Given current state, calculate all possible transitions based on all possible actions of RandomAgent
        
        '''
        
        for hand in Dealer.RANK_LIST:
            key = position + '_' + str(my_chips) + '_' + str(other_chips) + '_' + hand + '_'
            if (position == 'first' and other_chips == 0 and my_action == 'bet') or (position == 'second' and my_action == 'raise'):
                new_my_chips = my_chips + 1 + other_chips
                action_prob = 1/3 if position == 'first' else 1/2 # first position -> 'fold', 'raise', 'bet', second position -> 'fold', 'bet'
                #### other_action == 'fold' ####
                is_terminal = True
                new_other_chips = -1
                reward = my_chips + other_chips
                self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                #### other_action == 'bet' ####
                is_terminal = False
                new_other_chips = 0
                reward = 0
                self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                if position == 'first':
                    #### other_action == 'raise' ####
                    is_terminal = False
                    new_other_chips = 1
                    reward = 0
                    self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (other_chips == 1 and my_action == 'bet'):
                new_my_chips = my_chips + 1
                action_prob = 1 # random agent has finished his move by raising
                is_terminal = False
                new_other_chips = 0
                reward = 0
                self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (my_action == 'fold'):
                new_my_chips = my_chips
                action_prob = 1 # random agent has finished his move by raising
                is_terminal = True
                new_other_chips = 1
                reward = -my_chips
                self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
            elif (my_action == 'check'):
                new_my_chips = my_chips
                action_prob = 0.5 if position == 'first' else 1 # first position -> randomly between 'check', 'raise', second position -> round done
                #### other_action == 'check' ####
                is_terminal = False
                new_other_chips = 0
                reward = 0
                self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)
                if position == 'first':
                    #### other_action == 'raise' ####
                    is_terminal = False
                    new_other_chips = 1
                    reward = 0
                    self._calculate_cards_states(state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round)

    def _calculate_cards_states(self, state_space, key, my_action, action_prob, position, new_my_chips, new_other_chips, is_terminal, reward, hand, win_probabilities, loss_probabilities, flop_probabilities, game_round):
        ''' Given current state and possible actions of RandomAgent, calculate all possible transitions based on public cards and judging rules
        
        '''
        
        if game_round == 1: # end of round 1
            full_key = key + 'none' + '_AJKQT' # AJKQT because we do not have any information about RandomAgent's possible hand based on his action
            if new_other_chips != 0: 
                public_cards = 'none' # game ended with a fold before opening public cards
                self._add_or_update_key(state_space, full_key, action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
            else:
                if position == 'second': # add all possible first actions of first position after flop (either 'bet' or 'check')
                    new_other_chips_round2_list = [0, 1]
                    action_prob = action_prob / len(new_other_chips_round2_list)
                else:
                    new_other_chips_round2_list = [0]
                for new_other_chips in new_other_chips_round2_list:
                    for public_cards in flop_probabilities[hand]['AJKQT']: # store state transition for each possible public card combination (only rank matters)
                        self._add_or_update_key(state_space, full_key, flop_probabilities[hand]['AJKQT'][public_cards]*action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
        else: # end of round 2
            for public_cards in flop_probabilities[hand]['AJKQT']:
                full_key = key + public_cards + '_AJKQT'
                if new_other_chips == 0: # game finished, result by judging both players' hands 
                    is_terminal = True
                    win_prob = win_probabilities[hand][public_cards]['AJKQT']
                    loss_prob = loss_probabilities[hand][public_cards]['AJKQT']
                    tie_prob = 1 - win_prob - loss_prob
                    for result_prob in [win_prob, loss_prob, tie_prob]: # store state transition for each possible game result
                        reward = new_my_chips if result_prob == win_prob else -new_my_chips if result_prob == loss_prob else 0
                        self._add_or_update_key(state_space, full_key, result_prob*action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)
                else: # game finished with a fold after opening public cards
                    self._add_or_update_key(state_space, full_key, action_prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards)

    def _add_or_update_key(self, state_space, key, prob, my_action, position, new_my_chips, new_other_chips, is_terminal, reward, hand, public_cards):
        if prob > 0: # no need to store impossible transitions
            try_key_initialization(state_space, key, {})
            try_key_initialization(state_space[key], my_action, [])
            new_key = position + '_' + str(new_my_chips) + '_' + str(new_other_chips) + '_' + hand + '_' + public_cards + '_AJKQT'
            state_space[key][my_action].append( (prob, new_key, reward, is_terminal)  )
