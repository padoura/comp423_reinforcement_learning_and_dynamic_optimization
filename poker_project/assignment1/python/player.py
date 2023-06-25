''' Player class adapted from rlcard
'''
class Player:

    def __init__(self, player_id):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
        """
        self.player_id = player_id
        self.hand = []
        self.status = 'alive'
        self.position = None
        self.opponent_range = 'AK'

        # The chips that this player has put in until now
        self.in_chips = 0

    def get_state(self, public_cards, all_chips, legal_actions):
        """
        Encode the state for the player

        Args:
            public_cards (list): A list of public cards that seen by all the players
            all_chips (int): The chips that all players have put in

        Returns:
            (dict): The state of the player
        """
        return {
            'hand': [c for c in self.hand],
            'public_cards': [c for c in public_cards],
            'all_chips': all_chips,
            'my_chips': self.in_chips,
            'legal_actions': legal_actions,
            'position': self.position,
            'opponent_range': self.opponent_range
        }

    def get_player_id(self):
        return self.player_id
