from dealer import Dealer
from player import Player
from judger import Judger

class RandomAgent:
    ''' A random agent for benchmarking purposes
    '''

    def __init__(self, num_actions, np_random):
        ''' Initilize the random agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.np_random = np_random
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        ''' Completely random agent

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The randomly chosen action
        '''
        self._print_state(state['raw_obs'], state['action_record'])

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
    def get_transition_probabilities_for_actions(legal_actions):
        ''' Return state transition probabilities due to known random opponent

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action_probabilities (dictionary): Probabilities for each action to be selected
        '''
        action_probabilities = {}
        for action in legal_actions:
            action_probabilities[action] = 1 / len(legal_actions)

        return action_probabilities
    
    @staticmethod
    def get_transition_probabilities_for_cards():
        ''' Calculates transition probabilities for pre- and post-flop state of cards
        To be used for state transitions of value/policy iteration algorithms

        Returns:
           [ win_probabilities, loss_probabilities, flop_probabilities ] (dictionaries): Transition probabilities for pre- and post-flop state of cards
        '''

        deck = Dealer.init_standard_deck()
        win_frequencies = {}
        win_probabilities = {}
        tie_frequencies = {}
        loss_frequencies = {}
        loss_probabilities = {}
        total_opposing_frequencies = {}

        flop_frequencies = {}
        flop_probabilities = {}
        
        my_player = Player(0)
        my_player.in_chips = 0.5
        opposing_player = Player(1)
        opposing_player.in_chips = 0.5
        # num_of_possible_opposing =  (len(deck) - 3) # 3 cards known, the rest are possible opposing_hands remaining

        for my_hand_idx, my_hand in enumerate(deck):
            my_player.hand = [my_hand]
            if my_hand.rank not in flop_frequencies:
                flop_frequencies[my_hand.rank] = {}
            for public_card1_idx, public_card1 in enumerate(deck):
                if my_hand_idx != public_card1_idx:
                    for public_card2_idx, public_card2 in enumerate(deck):
                        if my_hand_idx != public_card2_idx and public_card1_idx != public_card2_idx:
                            public_hands = [ public_card1, public_card2 ]
                            key = my_hand.rank + ''.join(sorted(public_card1.rank + public_card2.rank))
                            if key not in tie_frequencies:
                                tie_frequencies[key] = 0
                                win_frequencies[key] = 0
                                loss_frequencies[key] = 0
                                total_opposing_frequencies[key] = 0
                            if key not in flop_frequencies[my_hand.rank]:
                                flop_frequencies[my_hand.rank][key] = 0
                            flop_frequencies[my_hand.rank][key] += 1
                            for opposing_hand_idx, opposing_hand in enumerate(deck):
                                if my_hand_idx != opposing_hand_idx and public_card1_idx != opposing_hand_idx and public_card2_idx != opposing_hand_idx:
                                    opposing_player.hand = [opposing_hand]
                                    players = [ my_player, opposing_player ]
                                    payoffs = Judger.judge_game(players, public_hands)
                                    total_opposing_frequencies[key] += 1
                                    # if key == 'QJQ': # DEBUG
                                    #     print('my hand: ', players[0].hand[0], ', opposing hand: ', players[1].hand[0], ', public 1: ', public_hands[0], ', public 2: ', public_hands[1], ', payoffs: ', payoffs)
                                    if payoffs[0] == 0.5:
                                        win_frequencies[key] += 1
                                    elif payoffs[0] == 0:
                                        tie_frequencies[key] += 1
                                    else:
                                        loss_frequencies[key] += 1
        
        for key in total_opposing_frequencies:
            win_probabilities[key] = win_frequencies[key] / total_opposing_frequencies[key]
            loss_probabilities[key] = loss_frequencies[key] / total_opposing_frequencies[key]

        for key in flop_frequencies:
            flop_probabilities[key] = {}
            for public_key in flop_frequencies[key]:
                flop_probabilities[key][public_key] = flop_frequencies[key][public_key] / sum(flop_frequencies[key].values())

        return [ win_probabilities, loss_probabilities, flop_probabilities ]