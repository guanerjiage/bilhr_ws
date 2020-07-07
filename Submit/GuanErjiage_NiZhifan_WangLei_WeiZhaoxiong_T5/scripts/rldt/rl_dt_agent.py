#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import json
from decision_tree import DecisionTree, Node, unicode_convert


class RLDTAgent(object):

    def __init__(self, num_feature, action_space, gamma=0.9, epsilon=0.1, desired_reward=15, step_max=10, gain_threshold=0.0, exploration_rate=0.4):
        self.num_feature = num_feature # number of features (number of columns in feature dataset)
        self.action_space = action_space # all actions
        self.num_action = len(action_space) # number of actions
        self.action_map = {action_space[i]: i for i in range(self.num_action)} # map an action to an index        
        self.gamma = gamma # reward decay
        self.epsilon = epsilon # exploration rate
        self.gain_threshold = gain_threshold # threshold for decision tree to cut unnecessary branch
        self.desired_reward = desired_reward # value to determine exploration
        self.exploration_rate = exploration_rate # rate to determine the exploration mode, 40% of desired reward default
        self.step_max = step_max # maximum relative moves 

        print("Init RLDT Agent")
        print("Action space: " + str(self.action_space))
        print("Action map: " + str(self.action_map))

        # initialize all decision trees
        self.dt_list = []
        for index_tree in range(self.num_feature + 1): # plus 1 for reward tree
            self.dt_list.append(DecisionTree(threshold=self.gain_threshold))
        
        # initialization model
        self.state_list = [] # store all states
        self.not_terminal = [] # store state that are not terminal
        self.action_value_function = {} # action value function, dict, key=state s, value=array of Q(s, a)
        self.fake_action_value_function = {} # fake Q function for exploration mode
        self.policy_function = {} # policy, dict, key=state s, value=action name
        self.reward_function = {} # reward model, dict, key=state s, value=array of R(s, a) 
        self.state_transition = {} # state transition model, dict, key=state s, value=dict of P(s_|s, a) 
        self.state_action_visit = {} # visited times of a state-action pair, dict, key=state s, value=array of visit(s,a)
        self.exploration_mode = True # True: exploration, False: exploitation
    
    # s should be a list
    def check_exist_state(self, s):
        if s not in self.state_list:
            self.state_list.append(s)
            # self.action_value_function[str(s)] = np.zeros(self.num_action)
            self.policy_function[str(s)] = self.action_space[0]
            self.reward_function[str(s)] = np.zeros(self.num_action)
            self.state_transition[str(s)] = {}
            self.state_action_visit[str(s)] = np.zeros(self.num_action)
            return False
        return True

    # epsilon-greedy action choice
    def choose_action(self, s, exploit=False):
        if not self.check_exist_state(s):
            print("State not exist, return init action")
        if s not in self.not_terminal:
            self.not_terminal.append(s)

        if not self.exploration_mode:
            a = self.policy_function[str(s)]
        else:
            if str(s) not in self.fake_action_value_function:
                a = np.random.choice(self.action_space) 
            else:
                a_index = np.argmax(self.fake_action_value_function[str(s)])
                a = self.action_space[a_index]
        if exploit:
            return a
        else:
            if np.random.rand() > self.epsilon:
                return a
            else:
                return np.random.choice(self.action_space)

    # determine whether should explore other actions in given state
    def determine_exploration(self):
        for s in self.state_list:
            if s not in self.not_terminal:
                continue
            a = self.policy_function[str(s)]
            a_index = self.action_map[a]
            if self.action_value_function[str(s)][a_index] < self.desired_reward * self.exploration_rate:
                self.exploration_mode = True
                return True
        self.exploration_mode = False
        return False
    
    # s, s_ should be lists (list with one element also ok), a should be a string or number, r should be a number 
    def add_exp_to_tree(self, s, a, r, s_):
        self.check_exist_state(s)
        if s not in self.not_terminal:
            self.not_terminal.append(s)
        self.check_exist_state(s_)
        # increment visit(s, a)
        a_index = self.action_map[a]
        self.state_action_visit[str(s)][a_index] += 1
        # concatenate feature vector
        feature_vec = s + [a]
        # add to reward tree
        exp_r = feature_vec + [r]
        self.dt_list[-1].append_data(exp_r)
        # add to relative feature change trees
        for i in range(len(s_)):
            exp_s_ = feature_vec + [s_[i] - s[i]] # relative change in feature as label
            self.dt_list[i].append_data(exp_s_)

    # generate state transition probability function P(s_|s, a) for all states and actions using decision trees
    def generate_transition(self):
        num_state = len(self.state_list)
        # all states
        for s_index in range(num_state):
            s = self.state_list[s_index]
            # all actions
            for a_index in range(self.num_action):
                if self.state_action_visit[str(s)][a_index] == 0:
                    continue
                a = self.action_space[a_index]
                # concatenate feature
                feature = s + [a]
                state_next_list = [[]] # all possible s_
                state_next_prob_list = [1.0] # P(s_|s,a)
                for feature_index in range(len(s)):
                    # get possible relative changes
                    change_list, change_prob_list = self.dt_list[feature_index].predict(feature)
                    # there is relative change in this feature
                    if change_list is not None and change_prob_list is not None:
                        # combine changes to states
                        temp_state_list = []
                        temp_state_prob_list = []
                        for i in range(len(state_next_list)):
                            for j in range(len(change_list)):
                                temp_state_list.append(state_next_list[i] + [s[feature_index] + change_list[j]])
                                temp_state_prob_list.append(state_next_prob_list[i] * change_prob_list[j])
                        state_next_list = temp_state_list
                        state_next_prob_list = temp_state_prob_list
                    # no relative change possible, append current feature
                    else:
                        for state_next in state_next_list:
                            state_next.append(s[i])
                # assign data to transition function dict      
                self.state_transition[str(s)][a] = [state_next_list, state_next_prob_list]
                for state_next in state_next_list:
                    self.check_exist_state(state_next)                   

    # generate the reward function R(s, a) for all states and actions using decision trees
    def generate_reward(self):
        for s in self.state_list:
            for a_index in range(self.num_action):
                if self.state_action_visit[str(s)][a_index] == 0:
                    continue
                a = self.action_space[a_index]
                feature = s + [a]
                reward_list, reward_prob = self.dt_list[-1].predict(feature)
                # if reward_list is not None and reward_prob is not None:
                mean_reward = np.sum(np.array(reward_list, dtype=np.float) * np.array(reward_prob, dtype=np.float))
                self.reward_function[str(s)][a_index] = mean_reward
                
    # value iteration as in the paper
    def value_iteration(self, eps=1e-7, verbose=False):
        K = {}
        current_q = {str(s): np.zeros(self.num_action) for s in self.state_list}
        old_q_mat = np.array(current_q.values())
        for s in self.state_list:
            if np.sum(self.state_action_visit[str(s)]) > 0:
                K[str(s)] = 0
            else:
                K[str(s)] = np.inf
        converge = False
        iter_step = 0
        while not converge:
            for s in self.state_list:
                for a_index in range(self.num_action):
                    if K[str(s)] > self.step_max:
                        # state out of reach
                        current_q[str(s)][a_index] = self.desired_reward
                    else:
                        # update remaining state's action-values
                        current_q[str(s)][a_index] = self.reward_function[str(s)][a_index]
                        a = self.action_space[a_index]
                        if a in self.state_transition[str(s)]:
                            temp = self.state_transition[str(s)][a]
                            s_next_list = temp[0]
                            s_next_prob_list = temp[1]
                            for i in range(len(s_next_list)):                            
                                s_ = s_next_list[i]
                                # update steps to this state
                                if K[str(s)] + 1 < K[str(s_)]:
                                    K[str(s_)] = K[str(s)] + 1
                                # bellman equation
                                current_q[str(s)][a_index] += self.gamma * s_next_prob_list[i] * np.max(current_q[str(s_)])
            iter_step += 1
            # calculate MSE to the old Q
            current_q_mat = np.array(current_q.values())
            mse = np.sum((old_q_mat - current_q_mat) ** 2)
            if mse <= eps:
                converge = True
            old_q_mat = current_q_mat
        self.action_value_function = current_q
        if verbose:
            print("Value iteration finish with %d steps" % iter_step)
        
        # maintain another action-value function to generate action in exploration mode
        if self.exploration_mode:
            K_fake = {}
            current_q_fake = {str(s): np.zeros(self.num_action) for s in self.state_list}
            old_q_mat_fake = np.array(current_q_fake.values())
            for s in self.state_list:
                if np.sum(self.state_action_visit[str(s)]) > 0:
                    K_fake[str(s)] = 0
                else:
                    K_fake[str(s)] = np.inf
            # state_visit = np.sum(self.state_action_visit.values(), axis=1)
            min_visit_fake = 999999999
            for s in self.state_list:
                if s in self.not_terminal:
                    min_visit_fake = min(np.min(self.state_action_visit[str(s)]), min_visit_fake)
            converge_fake = False
            iter_step_fake = 0
            while not converge_fake:
                for s in self.state_list:
                    for a_index in range(self.num_action):
                        if np.sum(self.state_action_visit[str(s)][a_index]) == min_visit_fake and s in self.not_terminal:
                            # unknown states are given exploration bonus
                            current_q_fake[str(s)] = np.ones(self.num_action) * self.desired_reward
                            current_q_fake[str(s)][a_index] = self.desired_reward * 1.5
                        elif K_fake[str(s)] > self.step_max and s in self.not_terminal:
                            # state out of reach
                            current_q_fake[str(s)][a_index] = self.desired_reward
                        else:
                            # update remaining state's action-values
                            current_q_fake[str(s)][a_index] = self.reward_function[str(s)][a_index]
                            a = self.action_space[a_index]
                            if a in self.state_transition[str(s)]:
                                temp_fake = self.state_transition[str(s)][a]
                                s_next_list_fake = temp_fake[0]
                                s_next_prob_list_fake = temp_fake[1]
                                for i in range(len(s_next_list_fake)):                            
                                    s_ = s_next_list_fake[i]
                                    # update steps to this state
                                    if K_fake[str(s)] + 1 < K_fake[str(s_)]:
                                        K_fake[str(s_)] = K_fake[str(s)] + 1
                                    # bellman equation
                                    current_q_fake[str(s)][a_index] += self.gamma * s_next_prob_list_fake[i] * np.max(current_q_fake[str(s_)])
                iter_step_fake += 1
                # calculate MSE to the old Q
                current_q_mat_fake = np.array(current_q_fake.values())
                mse_fake = np.sum((old_q_mat_fake - current_q_mat_fake) ** 2)
                if mse_fake <= eps:
                    converge_fake = True
                old_q_mat_fake = current_q_mat_fake
            self.fake_action_value_function = current_q_fake
            if verbose:
                print("Fake Value Iteration finish with %d steps" % iter_step_fake)

    # derive policy function from action-value function
    def generate_policy(self):
        for s in self.state_list:
            a_index = np.argmax(self.action_value_function[str(s)])
            a = self.action_space[a_index]
            self.policy_function[str(s)] = a

    # build tree, build model, value iteration, extract policy
    def learn(self):
        for dt in self.dt_list:
            dt.build_tree()
        self.generate_transition()
        self.generate_reward()
        self.value_iteration()
        self.generate_policy()

    # save decision trees and some dataset deependent variables
    def save_agent(self, filename_prefix):
        for i in range(len(self.dt_list)):
            filename = filename_prefix + "_tree_" + str(i) + ".json"
            self.dt_list[i].save_tree(filename)
        filename = filename_prefix + "_agent.json"
        temp_visit = {key: value.tolist() for key, value in self.state_action_visit.items()}
        agent_dict = {"state_list": self.state_list, "not_terminal": self.not_terminal, "state_action_visit": temp_visit}
        agent_json = json.dumps(agent_dict, indent=4)
        with open(filename, "w") as f:
            f.write(agent_json)
            f.write("\n")
        print("Agent saved")
    
    # load all decision trees and rebuild the MDP model
    def load_agent(self, filename_prefix):
        for i in range(len(self.dt_list)):
            filename = filename_prefix + "_tree_" + str(i) + ".json"
            self.dt_list[i].load_tree(filename)
        filename = filename_prefix + "_agent.json"
        with open(filename, "r") as f:
            agent_json = f.read()
            agent_dict = unicode_convert(json.loads(agent_json))
            temp_state_list = agent_dict["state_list"]
            for state in temp_state_list:
                self.check_exist_state(state)
            self.not_terminal = agent_dict["not_terminal"]
            temp_visit = agent_dict["state_action_visit"]
            for key, value in temp_visit.items():
                self.state_action_visit[key] = np.array(value)
            # self.state_action_visit = {key: np.array(value) for key, value in temp_visit.items()}
        self.learn()
        self.determine_exploration()
        print("Agent loaded")

        
        


if __name__ == "__main__":
    agent = RLDTAgent(2, ["left", "right", "up", "down"], gamma=0.99, epsilon=0.1, desired_reward=15, step_max=10, gain_threshold=0.0)
    agent.add_exp_to_tree([0, 0], "left", -1, [0, 0])
    agent.add_exp_to_tree([0, 0], "right", -1, [0, 1])
    agent.add_exp_to_tree([0, 1], "left", -1, [0, 0])
    agent.add_exp_to_tree([0, 0], "down", -1, [1, 0])
    agent.add_exp_to_tree([1, 0], "right", -1, [1, 1])
    agent.add_exp_to_tree([1, 1], "right", -1, [1, 2])
    agent.add_exp_to_tree([1, 2], "down", -1, [2, 2])
    agent.add_exp_to_tree([2, 2], "up", -1, [1, 2])
    agent.add_exp_to_tree([1, 2], "right", 30, [1, 3])
    agent.learn()
    
    print("Transition")
    print(agent.state_transition)
    print("Reward")
    print(agent.reward_function)
    print("Q")
    print(agent.action_value_function)
    for dt in agent.dt_list:
        print(dt)