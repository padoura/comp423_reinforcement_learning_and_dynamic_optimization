from card import Card


class HumanAgent:
    ''' A human agent adapted from rlcard library. It can be used to play against trained models
    '''

    def __init__(self, num_actions):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        HumanAgent._print_state(state['raw_obs'], state['action_record'])
        action = int(input('>> You (Player {}) choose action (integer): '.format(state['raw_obs']['current_player'])))
        while action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}

    @staticmethod
    def _print_state(state, action_record):
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
