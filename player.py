import pickle
import numpy as np
import random 
from Counter import Counter
import time

class Player:
    def __init__(self, name, exploration_rho=0.3, lr_alpha=0.2, discount_rate_gamma=0.9, walk_len_nu=0.2):
        self.name = name
        self.exploration_rho = exploration_rho
        self.lr_alpha = lr_alpha
        self.discount_rate_gamma = discount_rate_gamma
        self.walk_len_nu = walk_len_nu
        
        # Q-table
        self.states_value = Counter()  # state -> value
        # current score
        self.old_score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []
        self.isDead = False
        self.isFreight = False

    # Get Q(s,a).
    def getQValue(self, state, action):
        return self.states_value[str([state,action])]

    # Return the maximum Q value of a given state.
    def getMaxQ(self, state, possible_directions):
        q_list = []
        for action in possible_directions:
            q = self.getQValue(state,action)
            q_list.append(q)
        if len(q_list) ==0:
            return 0
        return max(q_list)

    # update Q value
    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state,action)
        self.states_value[str([state,action])] = (1 - self.lr_alpha)*q + self.lr_alpha*(reward + self.discount_rate_gamma*qmax - q)
    
    # Return the action that maximises Q of state.
    def takeBestAction(self, state, possible_directions):
        tmp = Counter()
        for action in possible_directions:
            tmp[action] = self.getQValue(state, action)
        # print(tmp)
        return tmp.argMax()
    
    # The main method required by the game. Called every time that
    # Pacman is expected to move.
    def getAction(self, state, possible_directions, score):
        # print("___________________________")
        # print(self.states_value)

        # Update Q-value
        reward = score - self.old_score

        # xe start
        # initialize: is pacman dead
        is_dead = state[3]
        # initialize: are any ghosts in freight mode
        is_freight = state[4]
        # initialize: manhattan distance to closest ghost
        closest_ghost_dist = state[5]
        # initialize: manhattan distance to closest pellet
        closest_pellet_dist = state[6]
        # initialize: manhattan distance to closest ghost's goal
        hunt_distance = state[7]

        # reward pacman for being near a ghost when in freight mode
        if is_freight and hunt_distance <= 30:
            reward += 5
            self.isFreight = False
            print("is_freight" + str(is_freight))
            print("reward" + str(reward))
        
        # punish pacman for being near a ghost when not in freight mode
        elif not is_freight and closest_ghost_dist <= 150 and not is_dead:
            reward -= 50
            # print("pacman too near ghost")
            # print("reward" + str(reward))
        
        # reward pacman if near a pellet, punish if not near
        if closest_pellet_dist < 15 and not is_dead:
            reward += 1
            # print("closest_pellet_dist" + str(closest_pellet_dist))
            # print("reward" + str(reward))
        else:
            reward -= 1

        # punish pacman for losing lives
        if (not self.isDead) and is_dead:
            reward = -150
            print("isdead" + str(is_dead))
            print("reward" + str(reward))
            self.isDead = False
        if is_dead:
            self.isDead = True
        
        # xe end

        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.getMaxQ(state, possible_directions)
            self.updateQ(last_state, last_action, reward, max_q)

        # (Explore vs Exploit)
        # Check if random action should be taken.
        rand_rho = random.uniform(0,1)
        if rand_rho < self.exploration_rho:
            # take random action
            action = np.random.choice(possible_directions) 
        else:
            # take the best action
            action =  self.takeBestAction(state, possible_directions)

        # Update attributes.
        self.old_score = score
        self.lastState.append(state)
        self.lastAction.append(action)

        return action


    # This is called by the game after a win or a loss.
    def final(self, state, score):
        # Update Q-values.
        reward = score - self.old_score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, 0)

        # Reset attributes.
        self.old_score = 0
        self.lastState = []
        self.lastAction = []

    # change the file name after every time you edit the code? so that 
    # it wont be overwritten???
    # Saves the Q-table.
    def savePolicy(self):
        fw = open('trained_controller', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    # Loads a Q-table.
    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()