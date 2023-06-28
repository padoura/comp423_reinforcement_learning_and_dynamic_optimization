import numpy as np
import json

class PolicyIterationAgent:
    ''' An agent following the optimal policy returned by Policy Iteration algorithm
    '''

    def __init__(self, np_random, print_enabled, opponent):
        self.np_random = np_random
         # preloading probabilities for using them in state transitions
        with open('win_probabilities.json') as json_file:
            win_probabilities = json.load(json_file)

        with open('loss_probabilities.json') as json_file:
            loss_probabilities = json.load(json_file)

        with open('flop_probabilities.json') as json_file:
            flop_probabilities = json.load(json_file)

        with open('range_probabilities.json') as json_file:
            range_probabilities = json.load(json_file)
        self.state_space = opponent.calculate_state_space(win_probabilities, loss_probabilities, flop_probabilities, range_probabilities)
        self.print_enabled = print_enabled # to prevent printing of cli for no-human games
        self.use_raw = True
        self.V_opt,self.P_opt = self.policy_iteration(self.state_space, gamma = 1.0)

    def step(self, state):
        ''' Given current state, choose the optimal action based on Policy Iteration algorithm

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): the optimal action
        '''
        if self.print_enabled: self._print_state(state['raw_obs'], state['action_record'])
        state_key = state['obs']['position'] + '_' + str(state['obs']['my_chips']) + '_' + str(state['obs']['other_chips']) + '_' + state['obs']['hand'] + '_' + state['obs']['public_cards'] + '_' + state['obs']['opponent_range']
        action = self.P_opt[state_key]
        return action

    def eval_step(self, states, action_history, payoff = None):
        ''' Method only needed for online learning
        '''
        pass

    def _print_state(self, state, action_record):
        ''' Print out the state

        Args:
            state (dict): A dictionary of the raw state
            action_record (list): A list of the historical actions
        '''
        if len(action_record) > 0:
            print('>> Player', action_record[-1][0], 'chooses', action_record[-1][1])

        print('===============     Chips      ===============')
        for i in range(len(state['all_chips'])):
            if i == state['current_player']:
                print('Policy Iteration Agent (Player {}): '.format(i), state['my_chips'])
            else:
                print('Player {}: '.format(i) , state['all_chips'][i])
        print('\n=========== Actions Policy Iteration Agent Can Choose ===========')
        print(', '.join([str(index) + ': ' + action for index, action in enumerate(state['legal_actions'])]))
        print('')

    def infer_card_range_from_action(self, action, game_round, current_range, other_chips, public_cards, position):
        return 'AJKQT' # range cannot be inferred by agent's actions

    ############################################################################
    # Policy Iteration Algorithm
    # 
    # Algorithm adapted from class' Frozen Lake example implementation
    # The most important change I had to perform was conversion of numpy arrays
    # into dictionaries for my choice of state representation.
    ############################################################################

    def policy_evaluation(self, pi, P, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
        # t = 0   #there's more elegant ways to do this
        prev_V = dict.fromkeys(P.keys(),0) # use as "cost-to-go", i.e. for V(s')
        while True:
            V = dict.fromkeys(P.keys(),0) # current value function to be learnerd
            for s in P.keys():  # do for every state
                for prob, next_state, reward, done in P[s][pi[s]]:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                    if done:
                        V[s] += prob * reward
                    else:
                        V[s] += prob * (reward + gamma * prev_V[next_state])
            if np.max([np.abs(prev_V[s] - V[s]) for s in V.keys()]) < epsilon: #check if the new V estimate is close enough to the previous one; 
                break # if yes, finish loop
            prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
            # t += 1
            # Vplot[:,t] = prev_V  # accounting for GUI  
        return V

    def policy_improvement(self, V, P, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
        Q = {key: dict.fromkeys(P[key],0) for key in P}
        for s in P.keys():        # for every state in the environment/model
            for a in P[s].keys():  # and for every action in that state
                for prob, next_state, reward, done in P[s][a]:  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                    if done:
                        Q[s][a] += prob * reward
                    else:
                        Q[s][a] += prob * (reward + gamma * V[next_state])
        new_pi = {s:max(Q[s], key=lambda k: Q[s][k]) for s in Q.keys()} # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
        # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
        # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself
        
        return new_pi

    # policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

    def policy_iteration(self, P, gamma = 1.0, epsilon = 1e-10):
        t = 0
        random_actions = { state: self.np_random.choice(tuple(P[state].keys())) for state in P.keys()}  # start with random actions for each state
        pi = random_actions     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
        
        while True:
            old_pi = {s: pi[s] for s in P.keys()}  #keep the old policy to compare with new
            V = self.policy_evaluation(pi,P,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
            pi = self.policy_improvement(V,P,gamma)          #get a better policy using the value function of the previous one just calculated 
            
            t += 1
            # Pplot[:,t]= [pi[s] for s in P.keys()]  #keep track of the policy evolution
            # Vplot[:,t] = V                              #and the value function evolution (for the GUI)
        
            if old_pi == {s:pi[s] for s in P.keys()}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        return V,pi