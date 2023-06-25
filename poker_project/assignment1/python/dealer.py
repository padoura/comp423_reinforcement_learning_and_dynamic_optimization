from card import Card
''' Dealer class adapted from rlcard
'''
class Dealer:

    SUIT_LIST = ['S', 'H', 'D', 'C']
    RANK_LIST = ['A', 'K']

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
        res = [Card(suit, rank) for suit in Dealer.SUIT_LIST for rank in Dealer.RANK_LIST]
        return res
