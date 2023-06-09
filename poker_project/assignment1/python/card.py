''' Card class adapted from rlcard
'''
class Card:
    '''
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C] meaning [Spades, Hearts, Diamonds, Clubs]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    '''
    suit = None
    rank = None
    valid_suit = ['S', 'H', 'D', 'C']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return self.rank + self.suit

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit+self.rank
    
    def rank_to_index(rank):
        ''' Get the coresponding number of a rank.

        Args:
            rank(str): rank stored in Card object

        Returns:
            (int): the number corresponding to the rank

        Note:
            1. If the input rank is an empty string, the function will return -1.
            2. If the input rank is not valid, the function will return None.
        '''
        if rank == '':
            return -1
        elif rank.isdigit():
            if int(rank) >= 2 and int(rank) <= 10:
                return int(rank)
            else:
                return None
        elif rank == 'A':
            return 14
        elif rank == 'T':
            return 10
        elif rank == 'J':
            return 11
        elif rank == 'Q':
            return 12
        elif rank == 'K':
            return 13
        return None
    
    @staticmethod
    def print_card(cards):
        ''' Nicely print a card or list of cards

        Args:
            card (string or list): The card(s) to be printed
        '''
        if cards is None:
            cards = [None]
        if isinstance(cards, str):
            cards = [cards]

        lines = [[] for _ in range(9)]

        for card in cards:
            if card is None:
                lines[0].append('┌─────────┐')
                lines[1].append('│░░░░░░░░░│')
                lines[2].append('│░░░░░░░░░│')
                lines[3].append('│░░░░░░░░░│')
                lines[4].append('│░░░░░░░░░│')
                lines[5].append('│░░░░░░░░░│')
                lines[6].append('│░░░░░░░░░│')
                lines[7].append('│░░░░░░░░░│')
                lines[8].append('└─────────┘')
            else:
                if isinstance(card, Card):
                    elegent_card = Card.elegent_form(card.suit + card.rank)
                else:
                    elegent_card = Card.elegent_form(card)
                suit = elegent_card[0]
                rank = elegent_card[1]
                if len(elegent_card) == 3:
                    space = elegent_card[2]
                else:
                    space = ' '

                lines[0].append('┌─────────┐')
                lines[1].append('│{}{}       │'.format(rank, space))
                lines[2].append('│         │')
                lines[3].append('│         │')
                lines[4].append('│    {}    │'.format(suit))
                lines[5].append('│         │')
                lines[6].append('│         │')
                lines[7].append('│       {}{}│'.format(space, rank))
                lines[8].append('└─────────┘')

    @staticmethod
    def elegent_form(card):
        ''' Get a elegent form of a card string

        Args:
            card (string): A card string

        Returns:
            elegent_card (string): A nice form of card
        '''
        suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣','s': '♠', 'h': '♥', 'd': '♦', 'c': '♣' }
        rank = '10' if card[1] == 'T' else card[1]
