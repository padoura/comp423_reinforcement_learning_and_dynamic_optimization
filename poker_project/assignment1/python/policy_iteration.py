from random_agent import RandomAgent
import numpy as np
import json

P = RandomAgent.calculate_game_tree()
t = 0

def policy_evaluation(pi, P, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
    t = 0   #there's more elegant ways to do this
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
        t += 1
        # Vplot[:,t] = prev_V  # accounting for GUI  
    return V

def policy_improvement(V, P, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
    Q = {key: dict.fromkeys(P[key],0) for key in P}
    for s in P.keys():        # for every state in the environment/model
        for a in P[s].keys():  # and for every action in that state
            for prob, next_state, reward, done in P[s][a]:  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                if done:
                    Q[s][a] += prob * reward
                else:
                    Q[s][a] += prob * (reward + gamma * V[next_state])
    new_pi = {s:max(Q[s], key=lambda k: Q[s][k]) for s in Q.keys()}
    print(new_pi)  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
    # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
    # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself
    
    return new_pi

# policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

def policy_iteration(P, gamma = 1.0, epsilon = 1e-10):
    t = 0
    random_actions = { state: np.random.choice(tuple(P[state].keys())) for state in P.keys()}  # start with random actions for each state
    pi = random_actions     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    
    while True:
        old_pi = {s: pi[s] for s in P.keys()}  #keep the old policy to compare with new
        V = policy_evaluation(pi,P,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
        pi = policy_improvement(V,P,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1
        # Pplot[:,t]= [pi[s] for s in P.keys()]  #keep track of the policy evolution
        # Vplot[:,t] = V                              #and the value function evolution (for the GUI)
    
        if old_pi == {s:pi[s] for s in P.keys()}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
    return V,pi


V_opt,P_opt = policy_iteration(P)   #just example of calling the various new functions we created.


with open("P_opt.json", "w") as write_file:
    json.dump(P_opt, write_file, indent=4, sort_keys=True)
with open("V_opt.json", "w") as write_file:
    json.dump(V_opt, write_file, indent=4, sort_keys=True)