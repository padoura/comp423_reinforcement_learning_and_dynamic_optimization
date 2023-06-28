import numpy as np
import json
import os
from utils import try_key_initialization

class QLearningAgent:
    ''' An agent following the optimal policy returned by Q-Learning algorithm
    '''

    def __init__(self, np_random, print_enabled, pretrained_model = None, is_learning = True, initial_epsilon = 1.0, initial_alpha = 1.0, epsilon_decay = -1/4, alpha_decay = -1/4):
        self.np_random = np_random
        self.print_enabled = print_enabled
        self.use_raw = True
        self.model = pretrained_model if pretrained_model != None else { 'Q': {}, 'episode_num': 0, 'policy': {}}
        self.is_learning = is_learning
        self.gamma = 1.0
        # self.initial_epsilon = 1.0 if self.slow_decay else 0.1
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self._update_epsilon()
        # self.initial_alpha = 1.0 if self.slow_decay else 0.1
        self.initial_alpha = initial_alpha
        self.alpha_decay = alpha_decay
        self._update_alpha()

    def _update_epsilon(self):
        ''' Exponential decay over time
        '''
        self.epsilon = self.initial_epsilon if self.model['episode_num'] == 0 else self.initial_epsilon*self.model['episode_num']**(self.epsilon_decay)
    def _update_alpha(self):
        ''' Exponential decay over time
        '''
        self.alpha = self.initial_alpha if self.model['episode_num'] == 0 else self.initial_alpha*self.model['episode_num']**(self.alpha_decay)

    def step(self, state):
        ''' Choose action for next step of Q Learning Algorithm using an e-greedy approach

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): the optimal action
        '''
        if self.print_enabled: self._print_state(state['raw_obs'], state['action_record'])
        state_key = state['obs']['position'] + '_' + str(state['obs']['my_chips']) + '_' + str(state['obs']['other_chips']) + '_' + state['obs']['hand'] + '_' + state['obs']['public_cards'] + '_' + state['obs']['opponent_range']
        Q = self.model['Q']
        ## Choose action from state using policy derived from Q and e-greedy
        action = self.np_random.choice(state['raw_legal_actions']) if self.np_random.binomial(1, self.epsilon) == 1 else self.model['policy'][state_key]
        return action

    def eval_step(self, states, action_history, payoff = None):
        ''' Evaluation of Q values and best policy per state so far
        '''
        new_state = states[-1]
        if len(states) > 1:
            old_state = states[-2]
            old_state_key = old_state['obs']['position'] + '_' + str(old_state['obs']['my_chips']) + '_' + str(old_state['obs']['other_chips']) + '_' + old_state['obs']['hand'] + '_' + old_state['obs']['public_cards'] + '_' + old_state['obs']['opponent_range']
        else:
            old_state = None
            
        new_state_key = new_state['obs']['position'] + '_' + str(new_state['obs']['my_chips']) + '_' + str(new_state['obs']['other_chips']) + '_' + new_state['obs']['hand'] + '_' + new_state['obs']['public_cards'] + '_' + new_state['obs']['opponent_range']

        ## initialize Q for a newly observed state
        Q = self.model['Q']
        if payoff == None: # there is no need to store terminal states
            try_key_initialization(Q, new_state_key, {action: 0.0 for action in new_state['raw_legal_actions']})
            try_key_initialization(self.model['policy'], new_state_key, new_state['raw_legal_actions'][0])

        ## Update Q if learning is enabled and action was performed
        if self.is_learning:
            if old_state != None: # new state is not an initial state
                latest_action = action_history[-1]
                if payoff == None: # no reward received yet, considered as 0 and is thus omitted
                    best_next_action = max(Q[new_state_key], key=Q[new_state_key].get)
                    Q[old_state_key][latest_action] = Q[old_state_key][latest_action] + self.alpha * (self.gamma*Q[new_state_key][best_next_action] - Q[old_state_key][latest_action])
                else: # reached terminal state, maximization term for further actions is 0 and is thus omitted
                    Q[old_state_key][latest_action] = Q[old_state_key][latest_action] + self.alpha * (payoff - Q[old_state_key][latest_action])
                    self.model['episode_num'] += 1
                    self._update_epsilon()
                    self._update_alpha()
                self.model['policy'][old_state_key] = max(Q[old_state_key], key=Q[old_state_key].get) # update policy for previous state based on new Q values

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
                print('Policy Iteration Agent (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions Policy Iteration Agent Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')

    def infer_card_range_from_action(self, action, game_round, current_range, other_chips, public_cards, position):
        return 'AJKQT' # range cannot be inferred by agent's actions