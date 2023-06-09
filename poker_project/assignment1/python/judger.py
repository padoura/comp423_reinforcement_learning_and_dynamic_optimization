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
            high_card_ranks.append(player.hand[0].rank_to_index())
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
                    one_pair_ranks[idx] = player.hand[0].rank_to_index()
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
