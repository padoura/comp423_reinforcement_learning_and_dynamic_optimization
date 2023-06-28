from card import Card


class HumanAgent:
    ''' A human agent adapted from rlcard library. It can be used to play against trained models
    '''

    def __init__(self, num_actions, modelled_opponent):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
            modelled_opponent (Agent class): 
        '''
        self.use_raw = True
        self.num_actions = num_actions
        self.modelled_opponent = modelled_opponent # Set the opposing agent so that the appropriate infer_card_range_from_action() method is used

    def step(self, state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        self._print_state(state['raw_obs'], state['action_record'])
        action = int(input('>> You (Player {}) choose action (integer): '.format(state['raw_obs']['current_player'])))
        while action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
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

        print('\n=============== Community Card ===============')
        Card.print_card(state['public_cards'])
        print('===============   Your Hand    ===============')
        Card.print_card(state['hand'])
        print('===============     Chips      ===============')
        for i in range(len(state['all_chips'])):
            if i == state['current_player']:
                print('Yours (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions You Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')

    def infer_card_range_from_action(self, action, game_round, current_range, other_chips, public_cards, position):
        # use what the opponent expects, to play against policy iteration agents 
        # WARNING: KeyError may occur when playing against the optimal policy for ThresholdAgent if unexpected moves are selected
        # (e.g. raising without A or K in hand for round 1, not raising in flop with AK as public cards in round 2)
        return self.modelled_opponent.infer_card_range_from_action(action, game_round, current_range, other_chips, public_cards, position)