from game import Game
import os

[ win_probabilities, loss_probabilities, flop_probabilities, range_probabilities ] = Game.get_transition_probabilities_for_cards()

import json
with open(os.path.dirname(os.path.abspath(__file__))+'\\flop_probabilities.json', 'w') as json_file:
    json.dump(flop_probabilities, json_file, indent=4)


import json
with open(os.path.dirname(os.path.abspath(__file__))+'\\loss_probabilities.json', 'w') as json_file:
    json.dump(loss_probabilities, json_file, indent=4)

import json
with open(os.path.dirname(os.path.abspath(__file__))+'\\win_probabilities.json', 'w') as json_file:
    json.dump(win_probabilities, json_file, indent=4)

import json
with open(os.path.dirname(os.path.abspath(__file__))+'\\range_probabilities.json', 'w') as json_file:
    json.dump(range_probabilities, json_file, indent=4)

