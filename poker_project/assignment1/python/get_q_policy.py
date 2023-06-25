import json
import os

q_trained_model = None
print("Loading pretrained model...")
with open(os.path.dirname(os.path.abspath(__file__))+'\\q_test.json') as json_file:
    q_trained_model = json.load(json_file)

Q = q_trained_model['Q']

Q_policy = {state: max(Q[state], key=Q[state].get) for state in Q}

with open(os.path.dirname(os.path.abspath(__file__))+'\\q_policy_test.json', 'w') as json_file:
    json.dump(Q_policy, json_file, indent=4, sort_keys=True)