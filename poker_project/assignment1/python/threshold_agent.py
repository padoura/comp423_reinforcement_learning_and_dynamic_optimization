from dealer import Dealer
from player import Player
from judger import Judger


class ThresholdAgent:
    ''' Threshold ("static") agent for benchmarking purposes
    '''

    def __init__(self, print_enabled):
        ''' Initilize the agent
        '''
        self.use_raw = True
        self.print_enabled = print_enabled

    def step(self, state):
        ''' Threshold ("static") agent

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (str): The rule-based chosen action
        '''
        if self.print_enabled: self._print_state(state['raw_obs'], state['action_record'])

        is_round_1 = (state['raw_obs']['public_cards'][0] is None) and (state['raw_obs']['public_cards'][1] is None)

        if is_round_1:
            action = self._choose_action_round_1(state)
        else:
            action = self._choose_action_round_2(state)
        return action

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
                print('Threshold Agent (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions Threshold Agent Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')

    def _choose_action_round_1(self, state):
        '''Maximum bet/raise with K or A
        Check/bet with Q or J
        Check/fold with 10
        '''
        if state['raw_obs']['hand'][0].rank_to_index() >= 13:
            if 'raise' in state['raw_legal_actions']:
                action = 'raise'
            else:
                action = 'bet'
        else:
            if 'check' in state['raw_legal_actions']:
                action = 'check'
            elif state['raw_obs']['hand'][0].rank_to_index() > 10:
                action = 'bet'
            else:
                action = 'fold'

        return action
    
    def _choose_action_round_2(self, state):
        '''Maximum bet/raise with at least a pair
        Check/bet with A, K, Q
        Check/fold with J, T
        '''

        has_at_least_a_pair = (
            state['raw_obs']['hand'][0].rank == state['raw_obs']['public_cards'][0].rank
            ) or (
            state['raw_obs']['hand'][0].rank == state['raw_obs']['public_cards'][1].rank
            )


        if has_at_least_a_pair:
            if 'raise' in state['raw_legal_actions']:
                action = 'raise'
            else:
                action = 'bet'
        else:
            if 'check' in state['raw_legal_actions']:
                action = 'check'
            elif state['raw_obs']['hand'][0].rank_to_index() < 12:
                action = 'fold'
            else:
                action = 'bet'

        return action
    
    @staticmethod
    def infer_card_range_from_action(action, game_round, current_range, other_chips, public_cards):
        # Ignoring fold and final bet of the game since after these actions it is over
        if game_round == 1:
            if action == 'raise' or (other_chips == 0 and action == 'bet'):
                current_range.remove(card for card in ['T', 'J', 'Q'])
            elif action == 'bet':
                current_range.remove(card for card in ['T', 'K', 'A'])
            else:
                current_range.remove(card for card in ['K', 'A'])
        else:
            if action == 'raise' or (other_chips == 0 and action == 'bet'):
                current_range = [public_cards[0], public_cards[1]]
            else:
                current_range.remove(card for card in [public_cards[0], public_cards[1]])

    @staticmethod
    def get_transition_probabilities_for_cards():
        ''' Calculates transition probabilities for pre- and post-flop state of cards
        To be used for state transitions of value/policy iteration algorithms
        Part of the Agent classes because "Threshold" agents range of hands can be inferred from 

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
            flop_frequencies[my_hand.rank] = {}
            for preflop_opponent_range in ['AK', 'JQ', 'JQT']:
                if preflop_opponent_range not in flop_frequencies:
                    flop_frequencies[my_hand.rank][preflop_opponent_range] = {}
                    for opponent_hand in preflop_opponent_range:
                        possibly_removed_cards = list(filter(lambda card: card.rank == opponent_hand, deck))
                        for removed_card in possibly_removed_cards:
                            remaining_deck = list(filter(lambda card: card != removed_card, deck))
                            for public_card1_idx, public_card1 in enumerate(remaining_deck):
                                if my_hand_idx != public_card1_idx:
                                    for public_card2_idx, public_card2 in enumerate(remaining_deck):
                                        if my_hand_idx != public_card2_idx and public_card1_idx != public_card2_idx:
                                            public_hands = [ public_card1, public_card2 ]
                                            key = ''.join(sorted(public_card1.rank + public_card2.rank))
                                            if key not in tie_frequencies:
                                                tie_frequencies[my_hand.rank+key] = 0
                                                win_frequencies[my_hand.rank+key] = 0
                                                loss_frequencies[my_hand.rank+key] = 0
                                                total_opposing_frequencies[my_hand.rank+key] = 0
                                            if key not in flop_frequencies[my_hand.rank][preflop_opponent_range]:
                                                flop_frequencies[my_hand.rank][preflop_opponent_range][key] = 0
                                            flop_frequencies[my_hand.rank][preflop_opponent_range][key] += 1
                                            for opposing_hand_idx, opposing_hand in enumerate(deck):
                                                if my_hand_idx != opposing_hand_idx and public_card1_idx != opposing_hand_idx and public_card2_idx != opposing_hand_idx:
                                                    opposing_player.hand = [opposing_hand]
                                                    players = [ my_player, opposing_player ]
                                                    payoffs = Judger.judge_game(players, public_hands)
                                                    total_opposing_frequencies[my_hand.rank+key] += 1
                                                    # if key == 'QJQ': # DEBUG
                                                    #     print('my hand: ', players[0].hand[0], ', opposing hand: ', players[1].hand[0], ', public 1: ', public_hands[0], ', public 2: ', public_hands[1], ', payoffs: ', payoffs)
                                                    if payoffs[0] == 0.5:
                                                        win_frequencies[my_hand.rank+key] += 1
                                                    elif payoffs[0] == 0:
                                                        tie_frequencies[my_hand.rank+key] += 1
                                                    else:
                                                        loss_frequencies[my_hand.rank+key] += 1
        
        for key in total_opposing_frequencies:
            win_probabilities[key] = win_frequencies[key] / total_opposing_frequencies[key]
            loss_probabilities[key] = loss_frequencies[key] / total_opposing_frequencies[key]

        for key in flop_frequencies:
            flop_probabilities[key] = {}
            for preflop_opponent_range in flop_frequencies[key]:
                flop_probabilities[key][preflop_opponent_range] = {}
                for public_key in flop_frequencies[key][preflop_opponent_range]:
                    flop_probabilities[key][preflop_opponent_range][public_key] = flop_frequencies[key][preflop_opponent_range][public_key] / sum(flop_frequencies[key][preflop_opponent_range].values())

        return [ win_probabilities, loss_probabilities, flop_probabilities ]


# [ win_probabilities, loss_probabilities, flop_probabilities ] = ThresholdAgent.get_transition_probabilities_for_cards()

# import json
# with open('flop_probabilities.json', 'w') as json_file:
#     json.dump(flop_probabilities, json_file, indent=4)