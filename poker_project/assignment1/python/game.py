from copy import deepcopy, copy
import numpy as np

from dealer import Dealer
from player import Player
from judger import Judger
from round import Round


class Game:
    ''' The Game class adapted from rlcard
    '''
    def __init__(self, allow_step_back=False, num_players=2):
        ''' Initialize the class Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        ''' big/small blind
        # Some configarations of the game

        # Raise amount and allowed times
        '''
        # Some configarations of the game
        # These arguments can be specified for creating new games

        # Small blind and big blind
        self.small_blind = 0.5
        self.big_blind = self.small_blind

        # Raise amount and allowed times
        self.raise_amount = 1
        self.allowed_raise_num = 1

        self.num_players = num_players

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']

    def init_game(self):
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initilize two players to play the game
        self.players = [Player(i) for i in range(self.num_players)]

        # Prepare for the first round
        for i in range(self.num_players):
            self.players[i].hand.append(self.dealer.deal_card())
        # Randomly choose a small blind and a big blind
        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind
        self.public_cards = [None, None]
        # The player with small blind plays first
        self.game_pointer = s
        self.starting_game_pointer = s

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(raise_amount=self.raise_amount,
                           allowed_raise_num=self.allowed_raise_num,
                           num_players=self.num_players,
                           np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, starting_game_pointer=self.starting_game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 2 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''
        if self.allow_step_back:
            # First snapshot the current state
            r = copy(self.round)
            r_raised = copy(self.round.raised)
            gp = self.game_pointer
            r_c = self.round_counter
            d_deck = copy(self.dealer.deck)
            p = copy(self.public_cards)
            ps = [copy(self.players[i]) for i in range(self.num_players)]
            ps_hand = [copy(self.players[i].hand) for i in range(self.num_players)]
            self.history.append((r, r_raised, gp, r_c, d_deck, p, ps, ps_hand))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 2 cards as public cards.
            if self.round_counter == 0:
                self.public_cards[0] = self.dealer.deal_card()
                self.public_cards[1] = self.dealer.deal_card()

            self.round_counter += 1
            self.game_pointer = self.starting_game_pointer # unlike real heads up rules, keep player order pre-flop and post-flop the same
            self.round.start_new_round(self.game_pointer, self.starting_game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_state(self, player):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.round.get_legal_actions()
        state = self.players[player].get_state(self.public_cards, chips, legal_actions)
        state['current_player'] = self.game_pointer

        return state

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        alive_players = [1 if p.status=='alive' else 0 for p in self.players]
        # If only one player is alive, the game is over.
        if sum(alive_players) == 1:
            return True

        # If all rounds are finshed
        if self.round_counter >= 2:
            return True
        return False

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player, returned values normalized by big blind
        '''
        chips_payoffs = Judger.judge_game(self.players, self.public_cards)
        payoffs = np.array(chips_payoffs) / (self.big_blind)
        return payoffs

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, r_raised, self.game_pointer, self.round_counter, d_deck, self.public_cards, self.players, ps_hand = self.history.pop()
            self.round.raised = r_raised
            self.dealer.deck = d_deck
            for i, hand in enumerate(ps_hand):
                self.players[i].hand = hand
            return True
        return False
    
    @staticmethod
    def get_winning_frequency_if_cards_shown():
        ''' Calculates frequency of winning and tieing after (final) round 2 (flop) if 'fold' is not a possible action
        To be used for state transitions of value/policy iteration algorithms

        Returns:
           [ win_frequencies, tie_frequencies ] (2 dictionaries): Frequencies of all combinations of 1 hand + 2 public cards or 'tie' if it's a guarranteed tie
        '''

        deck = Dealer.init_standard_deck()
        win_frequencies = {}
        tie_frequencies = {}
        loss_frequencies = {}
        
        my_player = Player(0)
        my_player.in_chips = 0.5
        opposing_player = Player(1)
        opposing_player.in_chips = 0.5
        # num_of_possible_opposing =  (len(deck) - 3) # 3 cards known, the rest are possible opposing_hands remaining

        for my_hand_idx, my_hand in enumerate(deck):
            my_player.hand = [my_hand]
            for public_card1_idx, public_card1 in enumerate(deck):
                if my_hand_idx != public_card1_idx:
                    public_hands = []
                    public_hands.append(public_card1)
                    for public_card2_idx, public_card2 in enumerate(deck):
                        if my_hand_idx != public_card2_idx and public_card1_idx != public_card2_idx:
                            public_hands.append(public_card2)
                            key = my_hand.rank + ''.join(sorted(public_card1.rank + public_card2.rank))
                            if key not in tie_frequencies:
                                tie_frequencies[key] = 0
                                win_frequencies[key] = 0
                                loss_frequencies[key] = 0
                            for opposing_hand_idx, opposing_hand in enumerate(deck):
                                if my_hand_idx != opposing_hand_idx and public_card1_idx != opposing_hand_idx and public_card2_idx != opposing_hand_idx:
                                    opposing_player.hand = [opposing_hand]
                                    players = [ my_player, opposing_player ]
                                    payoffs = Judger.judge_game(players, public_hands)
                                    if payoffs[0] == 0.5:
                                        win_frequencies[key] += 1
                                    elif payoffs[0] == 0:
                                        tie_frequencies[key] += 1
                                    else:
                                        loss_frequencies[key] += 1
        

        return [ win_frequencies, tie_frequencies, loss_frequencies ]