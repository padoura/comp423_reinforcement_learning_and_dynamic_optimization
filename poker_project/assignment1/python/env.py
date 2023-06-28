''' Environment class adapted from rlcard
'''
from collections import OrderedDict
import numpy as np
from game import Game
import seeding

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class Env(object):
    '''
    The base Env class. For all the environments in RLCard,
    we should base on this class and implement as many functions
    as we can.
    '''
    def __init__(self, config = { 'allow_step_back': False, 'seed': None }):
        ''' Initialize the Limitholdem environment
        '''
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        # Set random seed, default is None
        self.seed(config['seed'])
        self.allow_step_back = self.game.allow_step_back = config['allow_step_back']
        self.action_recorder = []

        _game_config = self.default_game_config.copy()
        for key in config:
            if key in _game_config:
                _game_config[key] = config[key]
        self.game.configure(_game_config)

        # Get the number of players/actions in this game
        self.num_players = self.game.num_players
        self.num_actions = self.game.num_players

        # A counter for the timesteps
        self.timestep = 0
        
        self.actions = ['bet', 'raise', 'fold', 'check']


    def reset(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (list): The beginning state of the game
                (int): The beginning player
        '''
        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    def step(self, action, raw_action=False):
        ''' Step forward

        Args:
            action (int): The action taken by the current player
            raw_action (boolean): True if the action is a raw action

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        '''
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        # Record the action for human interface
        self.action_recorder.append((self.get_player_id(), action))
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def step_back(self):
        ''' Take one step backward.

        Returns:
            (tuple): Tuple containing:

                (dict): The previous state
                (int): The ID of the previous player

        Note: Error will be raised if step back from the root node.
        '''
        if not self.allow_step_back:
            raise Exception('Step back is off. To use step_back, please set allow_step_back=True in rlcard.make')

        if not self.game.step_back():
            return False

        player_id = self.get_player_id()
        state = self.get_state(player_id)

        return state, player_id

    def set_agents(self, agents):
        '''
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        '''
        self.agents = agents

    def run(self):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list of payoffs. Each entry corresponds to one player.

        Note: The trajectories are 2-dimension lists. The first dimension is for different players,
              while the second dimension is for the contents of each transition
        '''
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent learns
            player_action_history = [action_entry[1] for action_entry in trajectories[player_id][-1]['action_record'] if action_entry[0] == player_id]
            self.agents[player_id].eval_step(trajectories[player_id], player_action_history)

            # Agent plays
            action = self.agents[player_id].step(state)         

            # Get new opponent range based on action (only applicable vs ThresholdAgent as known opponent)
            new_opponent_range = self.agents[player_id].infer_card_range_from_action(action, self.game.round_counter+1, self.game.players[1 if player_id == 0 else 0].opponent_range, state['obs']['other_chips'], state['obs']['public_cards'], state['obs']['position'])
            # Update new opponent range based on action
            self.game.players[1 if player_id == 0 else 0].opponent_range = new_opponent_range[:]

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()
        
        # Agent learns from terminal state
        for player_id in range(self.num_players):
            player_action_history = [action_entry[1] for action_entry in trajectories[player_id][-1]['action_record'] if action_entry[0] == player_id]
            self.agents[player_id].eval_step(trajectories[player_id], player_action_history, payoffs[player_id])

        return trajectories, payoffs

    def is_over(self):
        ''' Check whether the current game is over

        Returns:
            (boolean): True if current game is over
        '''
        return self.game.is_over()

    def get_player_id(self):
        ''' Get the current player id

        Returns:
            (int): The id of the current player
        '''
        return self.game.game_pointer


    def get_state(self, player_id):
        ''' Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (list): The observed state of the player
        '''
        return self._extract_state(self.game.get_state(player_id))

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_cards'] = self.game.public_cards
        state['hand_cards'] = [self.game.players[i].hand for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.round.get_legal_actions()
        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.np_random = self.np_random
        return seed

    def _extract_state(self, state):
        ''' Extract the state representation for learning 'obs'.
        I chose descriptive representation over optimized memory usage.

        # | Index            | # Enum |Meaning                                                                        |
        # | -----------------|--------|-------------------------------------------------------------------------------|
        # | 'position'       |   2    |Position of player 'first'/'second'                                            |
        # | 'my_chips'       |   5    |Chips placed by our agent so far                                               |
        # | 'other_chips'    |   3    |Difference in chips placed between adversary and our agent so far (-1, 0, 1)   |
        # | 'hand'           |   5    |Rank of hand: T ~ A as first public card                                       |
        # | 'public_cards'   |   16   |Rank of public cards in alphabetical order e.g. 'AK' or 'none' if not shown yet|
        # | 'opponent_range' |   17   |Possible range of opponent's hand, meaningful for ThresholdAgent, else 'AJKQT' |

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        public_cards = state['public_cards']
        obs = {}
        obs['position'] = state['position']
        obs['hand'] = state['hand'][0].rank
        obs['my_chips'] = state['my_chips']
        obs['other_chips'] = int(sum(state['all_chips'])-2*state['my_chips'])
        if (public_cards[0] is not None) and (public_cards[1] is not None):
            obs['public_cards'] = ''.join(sorted(public_cards[0].rank + public_cards[1].rank))
        else:
            obs['public_cards'] = 'none'
        obs['opponent_range'] = state['opponent_range'] # state feature useful only for PolicyIterationAgent vs ThresholdAgent
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.round.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.round.get_legal_actions()
