class Judger:
    ''' The Judger class adapted from rlcard
    '''

    @staticmethod
    def judge_game(players, public_cards):
        ''' Judge the winner of the game.

        Args:
            players (list): The list of players who play the game
            public_cards (object): The public cards seen by all players

        Returns:
            (list): Each entry of the list corresponds to one entry of the
        '''
        # Judge who are the winners
        winners = [0] * len(players)
        fold_count = 0
        high_card_ranks = []
        # If every player folds except one, the alive player is the winner
        for idx, player in enumerate(players):
            high_card_ranks.append(Judger.rank2int(player.hand[0].rank))
            if player.status == 'folded':
               fold_count += 1
            else:
                alive_idx = idx
        if fold_count == (len(players) - 1):
            winners[alive_idx] = 1
        
        # Winning condition if both public cards have the same rank (only possibility of a 3 of a kind)
        if sum(winners) < 1 and public_cards[0].rank == public_cards[1].rank:
            for idx, player in enumerate(players):
                if player.hand[0].rank == public_cards[0].rank:
                    winners[idx] = 1

        # Winning condition for 1 pair
        if sum(winners) < 1 and public_cards[0].rank != public_cards[1].rank:
            one_pair_ranks = [0] * len(players)
            for idx, player in enumerate(players):
                if player.hand[0].rank == public_cards[0].rank or player.hand[0].rank == public_cards[1].rank:
                    one_pair_ranks[idx] = Judger.rank2int(player.hand[0].rank)
            if one_pair_ranks[0] != one_pair_ranks[1]:
                max_rank = max(one_pair_ranks)
                max_index = [i for i, j in enumerate(one_pair_ranks) if j == max_rank]
                if len(max_index) == 1:
                    winners[max_index[0]] = 1
                    
        
        # If non of the above conditions, the winner player is the one with the highest card rank
        if sum(winners) < 1:
            max_rank = max(high_card_ranks)
            max_index = [i for i, j in enumerate(high_card_ranks) if j == max_rank]
            for idx in max_index:
                winners[idx] = 1

        # Compute the total chips
        total = 0
        for p in players:
            total += p.in_chips

        each_win = float(total) / sum(winners)

        payoffs = []
        for i, _ in enumerate(players):
            if winners[i] == 1:
                payoffs.append(each_win - players[i].in_chips)
            else:
                payoffs.append(float(-players[i].in_chips))

        return payoffs
    
    @staticmethod
    def rank2int(rank):
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
