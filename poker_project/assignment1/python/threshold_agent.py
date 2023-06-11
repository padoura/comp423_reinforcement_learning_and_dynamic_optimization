from card import Card


class ThresholdAgent:
    ''' Threshold ("static") agent for benchmarking purposes
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        ''' Threshold ("static") agent

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (str): The rule-based chosen action
        '''
        self._print_state(state['raw_obs'], state['action_record'])

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
        Check/bet with Q
        Check/fold with 10 or J
        '''
        if state['raw_obs']['hand'][0].rank_to_index() >= 13:
            if 'raise' in state['raw_legal_actions']:
                action = 'raise'
            else:
                action = 'bet'
        else:
            if 'check' in state['raw_legal_actions']:
                action = 'check'
            elif state['raw_obs']['hand'][0].rank_to_index() == 12:
                action = 'bet'
            else:
                action = 'fold'

        return action
    
    def _choose_action_round_2(self, state):
        '''Maximum bet/raise with at least a pair
        Check/fold without
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
            else:
                action = 'fold'

        return action
