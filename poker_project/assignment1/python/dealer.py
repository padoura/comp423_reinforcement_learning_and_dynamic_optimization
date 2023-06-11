from card import Card
''' Dealer class adapted from rlcard
'''
class Dealer:
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = Dealer.init_standard_deck()
        self.shuffle()
        self.pot = 0

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_card(self):
        """
        Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        """
        return self.deck.pop()
    
    @staticmethod
    def init_standard_deck():
        ''' Initialize limited cards for assignment 1

        Returns:
            (list): A list of Card object
        '''
        suit_list = ['S', 'H', 'D', 'C']
        rank_list = ['A', 'T', 'J', 'Q', 'K']
        res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
        return res
