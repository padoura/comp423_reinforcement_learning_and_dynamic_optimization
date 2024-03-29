from copy import copy
import numpy as np

from dealer import Dealer
from player import Player
from judger import Judger
from round import Round
from utils import try_key_initialization


class Game:
    ''' The Game class adapted from rlcard
    '''

    POSSIBLE_OPPONENT_RANGES = [
        'A',
        'AJ',
        # 'AJK', # commented out impossible opponent ranges
        # 'AJKQ',
        'AJKQT',
        # 'AJKT',
        # 'AJQ',
        # 'AJQT',
        # 'AJT',
        'AK',
        # 'AKQ',
        # 'AKQT',
        # 'AKT',
        'AQ',
        # 'AQT',
        'AT',
        'J',
        'JK',
        # 'JKQ',
        # 'JKQT',
        # 'JKT',
        'JQ',
        'JQT',
        'JT',
        'K',
        'KQ',
        # 'KQT',
        'KT',
        'Q',
        'QT',
        'T'
    ]


    def __init__(self, allow_step_back=False, num_players=2):
        ''' Initialize the class Game
        '''
        self.allow_step_back = allow_step_back
        # self.np_random = np.random.RandomState() # commented out because it is initialized externally by Env class

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
        # Randomly choose a small blind (first) and a big blind (second player)
        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].position = 'second'
        self.players[b].in_chips = self.big_blind
        self.players[s].position = 'first'
        self.players[s].in_chips = self.small_blind
        self.public_cards = [None, None]
        # The player with small blind plays first
        self.game_pointer = s
        self.starting_game_pointer = s

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(raise_amount=self.raise_amount,
                           allowed_raise_num=self.allowed_raise_num,
                           num_players=self.num_players)

        self.round.start_new_round(game_pointer=self.game_pointer, starting_game_pointer=self.starting_game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 2 rounds in each game.
        self.round_counter = 0

        # Save the history for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (bet, raise, fold, or check)

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

        # If all rounds are finished
        if self.round_counter >= 2:
            return True
        return False

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        chips_payoffs = Judger.judge_game(self.players, self.public_cards)
        payoffs = chips_payoffs
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

        range_frequencies = {}
        range_probabilities = {}
        
        # initializing players and mocking chips for simulated rewards
        my_player = Player(0)
        my_player.in_chips = 0.5
        opposing_player = Player(1)
        opposing_player.in_chips = 0.5

        for my_hand in deck: # check all states for each possible card in hand
            my_player.hand = [my_hand]
            flop_frequencies[my_hand.rank] = {}
            tie_frequencies[my_hand.rank] = {}
            win_frequencies[my_hand.rank] = {}
            loss_frequencies[my_hand.rank] = {}
            total_opposing_frequencies[my_hand.rank] = {}

            for preflop_opponent_range in Game.POSSIBLE_OPPONENT_RANGES: # calculation of conditional probabilities given our current knowledge of the opponent's hand
                if preflop_opponent_range not in flop_frequencies[my_hand.rank]:
                    flop_frequencies[my_hand.rank][preflop_opponent_range] = {}
                    for opponent_hand in preflop_opponent_range: # for each possible rank in opponent's hand
                        possibly_removed_cards = list(filter(lambda card: card.rank == opponent_hand, deck))
                        for removed_card in possibly_removed_cards: # we know the rank of the opponent's hand, but not the suit, calculate for all possible suits
                            remaining_deck = list(filter(lambda card: card != removed_card, deck))
                            for public_card1 in remaining_deck:
                                if my_hand != public_card1:
                                    for public_card2 in remaining_deck: # count all possible public card ranks to calculate their conditional probabilities of appearing
                                        if my_hand != public_card2 and public_card1 != public_card2:
                                            public_hands = [ public_card1, public_card2 ]
                                            hand = ''.join(sorted(public_card1.rank + public_card2.rank))
                                            try_key_initialization(flop_frequencies[my_hand.rank][preflop_opponent_range], hand, 0)
                                            flop_frequencies[my_hand.rank][preflop_opponent_range][hand] += 1

            for public_card1 in deck: # Similarly as above, given our hand, opponent's range, and public cards, count all possible game judging scenarios (if no fold happens)
                if my_hand != public_card1:
                    for public_card2 in deck:
                        if my_hand != public_card2 and public_card1 != public_card2:
                            public_hands = [ public_card1, public_card2 ]
                            hand = ''.join(sorted(public_card1.rank + public_card2.rank))
                            try_key_initialization(tie_frequencies[my_hand.rank], hand, {})
                            try_key_initialization(win_frequencies[my_hand.rank], hand, {})
                            try_key_initialization(loss_frequencies[my_hand.rank], hand, {})
                            try_key_initialization(total_opposing_frequencies[my_hand.rank], hand, {})
                            for possible_hand_range in Game.POSSIBLE_OPPONENT_RANGES:
                                try_key_initialization(tie_frequencies[my_hand.rank][hand], possible_hand_range, 0)
                                try_key_initialization(win_frequencies[my_hand.rank][hand], possible_hand_range, 0)
                                try_key_initialization(loss_frequencies[my_hand.rank][hand], possible_hand_range, 0)
                                try_key_initialization(total_opposing_frequencies[my_hand.rank][hand], possible_hand_range, 0)
                                for opponent_hand in possible_hand_range:
                                    remaining_opposing_deck = list(filter(lambda card: card.rank == opponent_hand, deck))
                                    for opposing_hand in remaining_opposing_deck:
                                        if my_hand != opposing_hand and public_card1 != opposing_hand and public_card2 != opposing_hand:
                                            opposing_player.hand = [opposing_hand]
                                            players = [ my_player, opposing_player ]
                                            payoffs = Judger.judge_game(players, public_hands)
                                            total_opposing_frequencies[my_hand.rank][hand][possible_hand_range] += 1
                                            if payoffs[0] == 0.5:
                                                win_frequencies[my_hand.rank][hand][possible_hand_range] += 1
                                            elif payoffs[0] == 0:
                                                tie_frequencies[my_hand.rank][hand][possible_hand_range] += 1
                                            else:
                                                loss_frequencies[my_hand.rank][hand][possible_hand_range] += 1
        
        # convert absolute frequencies to probabilities
        for hand in total_opposing_frequencies:
            win_probabilities[hand] = {}
            loss_probabilities[hand] = {}
            for public_cards in win_frequencies[hand]:
                win_probabilities[hand][public_cards] = {}
                loss_probabilities[hand][public_cards] = {}
                for possible_hand_range in win_frequencies[hand][public_cards]:
                    win_probabilities[hand][public_cards][possible_hand_range] = win_frequencies[hand][public_cards][possible_hand_range] / total_opposing_frequencies[hand][public_cards][possible_hand_range]
                    loss_probabilities[hand][public_cards][possible_hand_range] = loss_frequencies[hand][public_cards][possible_hand_range] / total_opposing_frequencies[hand][public_cards][possible_hand_range]

        # convert absolute frequencies to probabilities
        for hand in flop_frequencies:
            flop_probabilities[hand] = {}
            for preflop_opponent_range in flop_frequencies[hand]:
                flop_probabilities[hand][preflop_opponent_range] = {}
                for public_key in flop_frequencies[hand][preflop_opponent_range]:
                    flop_probabilities[hand][preflop_opponent_range][public_key] = flop_frequencies[hand][preflop_opponent_range][public_key] / sum(flop_frequencies[hand][preflop_opponent_range].values())

        # Given our hand, no public cards, and current opponent range, calculate frequency of belonging to a new opponent range
        # Probablity is 1 if new range is same/superset of the old one, and <1 if it is a subset, 0 for incompatible ranges
        for my_hand in deck:
            range_frequencies[my_hand.rank] = {}
            range_frequencies[my_hand.rank]['none'] = {}
            for preflop_opponent_range in Game.POSSIBLE_OPPONENT_RANGES:
                range_frequencies[my_hand.rank]['none'][preflop_opponent_range] = {}
                for new_opponent_range in Game.POSSIBLE_OPPONENT_RANGES:
                    range_frequencies[my_hand.rank]['none'][preflop_opponent_range][new_opponent_range] = 0
                    for opponent_hand in deck:
                        if opponent_hand != my_hand and opponent_hand.rank in preflop_opponent_range and opponent_hand.rank in new_opponent_range:
                            range_frequencies[my_hand.rank]['none'][preflop_opponent_range][new_opponent_range] += 1

        # Do the same as above for all possible public hand combinations
        for my_hand in deck:
            for public_card1 in deck:
                if my_hand != public_card1:
                    for public_card2 in deck:
                        if my_hand != public_card2 and public_card1 != public_card2:
                            public_cards = ''.join(sorted(public_card1.rank + public_card2.rank))
                            try_key_initialization(range_frequencies[my_hand.rank], public_cards, {})
                            for preflop_opponent_range in Game.POSSIBLE_OPPONENT_RANGES:
                                range_frequencies[my_hand.rank][public_cards][preflop_opponent_range] = {}
                                for new_opponent_range in Game.POSSIBLE_OPPONENT_RANGES:
                                    range_frequencies[my_hand.rank][public_cards][preflop_opponent_range][new_opponent_range] = 0
                                    for opponent_hand in deck:
                                        if opponent_hand != my_hand and opponent_hand != public_card1 and opponent_hand != public_card2 and opponent_hand.rank in preflop_opponent_range and opponent_hand.rank in new_opponent_range:
                                            range_frequencies[my_hand.rank][public_cards][preflop_opponent_range][new_opponent_range] += 1        

        # convert absolute frequencies to probabilities
        for hand in range_frequencies:
            range_probabilities[hand] = {}
            for public_cards in range_frequencies[hand]:
                range_probabilities[hand][public_cards] = {}
                for opponent_range in range_frequencies[hand][public_cards]:
                    range_probabilities[hand][public_cards][opponent_range] = {}
                    for new_opponent_range in range_frequencies[hand][public_cards][opponent_range]:
                        range_probabilities[hand][public_cards][opponent_range][new_opponent_range] = range_frequencies[hand][public_cards][opponent_range][new_opponent_range] / range_frequencies[hand][public_cards][opponent_range][opponent_range]

        return [ win_probabilities, loss_probabilities, flop_probabilities, range_probabilities ]