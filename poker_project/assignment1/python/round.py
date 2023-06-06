''' Round class adapted from rlcard
'''
class Round:
    """Round can call other Classes' functions to keep the game running"""

    def __init__(self, bet_amount, np_random):
        """
        Initialize the round class

        Args:
            bet_amount (int): the raise amount for each raise
            allowed_raise_num (int): The number of allowed raise num
            starting_player_index (int): retains player with small blind
        """
        self.np_random = np_random
        self.player_index = None
        self.starting_player_index = None
        self.bet_amount = bet_amount
        self.allowed_raise_num = 1

        self.num_players = 2

        # Count the number of raise
        self.have_raised_num = 0

        # Count the number without raise
        # If every player agree to not raise, the round is over
        self.not_raise_num = 0

        # Raised amount for each player
        self.raised = [0 for _ in range(self.num_players)]
        self.player_folded = None

    def start_new_round(self, player_index, raised=None):
        """
        Start a new bidding round

        Args:
            player_index (int): The player_index that indicates the next player
            raised (list): Initialize the chips for each player

        Note: For the first round of the game, we need to setup the big/small blind
        """
        self.player_index = player_index
        self.have_raised_num = 0
        self.not_raise_num = 0
        if raised:
            self.raised = raised
        else:
            self.raised = [0 for _ in range(self.num_players)]

    def proceed_round(self, players, action):
        """
        Call other classes functions to keep one round running

        Args:
            players (list): The list of players that play the game
            action (str): An legal action taken by the player

        Returns:
            (int): The player_index that indicates the next player
        """
        if action not in self.get_legal_actions():
            raise Exception('{} is not legal action. Legal actions: {}'.format(action, self.get_legal_actions()))

        if action == 'bet':
            self.raised[self.player_index] += self.bet_amount
            players[self.player_index].in_chips += self.bet_amount
            self.not_raise_num += 1

        elif action == 'raise':
            diff = max(self.raised) - self.raised[self.player_index] + self.bet_amount
            self.raised[self.player_index] += diff
            players[self.player_index].in_chips += diff
            self.have_raised_num += 1
            self.not_raise_num = 1

        elif action == 'fold':
            players[self.player_index].status = 'folded'
            self.player_folded = True

        elif action == 'check':
            self.not_raise_num += 1

        self.player_index = (self.player_index + 1) % self.num_players

        # Skip the folded players
        while players[self.player_index].status == 'folded':
            self.player_index = (self.player_index + 1) % self.num_players

        return self.player_index

    def get_legal_actions(self):
        """
        Obtain the legal actions for the current player

        Returns:
           (list):  A list of legal actions
        """
        full_actions = ['bet', 'raise', 'fold', 'check']

        # If the the number of raises already reaches the maximum number raises, we can not raise any more
        # Moreover, player with small blind cannot raise
        if self.player_index == self.starting_player_index or self.have_raised_num >= self.allowed_raise_num:
            full_actions.remove('raise')

        # If the current chips are less than that of the highest one in the round, we can not check
        if self.raised[self.player_index] < max(self.raised):
            full_actions.remove('check')

        # Player with big blind cannot bet unless other player bet first
        if self.player_index != self.starting_player_index and self.raised[self.player_index] == max(self.raised):
            full_actions.remove('bet')

        # A Player cannot fold if current chips are the highest one in the round
        if self.raised[self.player_index] == max(self.raised):
            full_actions.remove('fold')

        return full_actions

    def is_over(self):
        """
        Check whether the round is over

        Returns:
            (boolean): True if the current round is over
        """
        if self.not_raise_num >= self.num_players:
            return True
        return False
