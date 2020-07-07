#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np


class QLearningAgent(object):

    def __init__(self, num_state, num_action, gamma=0.9, alpha=0.3, epsilon=0.1, q_table_initializer="zeros"):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table_initializer = q_table_initializer
        if q_table_initializer.lower() == "zeros":
            self.q_table = np.zeros((num_state, num_action))
        elif q_table_initializer.lower() == "normal" or q_table_initializer.lower() == "gaussian":
            self.q_table = np.random.randn(num_state, num_action)
        else:
            self.q_table = np.zeros((num_state, num_action))
        
        print("Create new Agent, %d states, %d actions, gamma=%.4f" % (num_state, num_action, gamma))
    
    def check_exist_state(self, s):
        if s >= self.num_state:
            state_to_add = s - self.num_state + 1
            self.num_state += state_to_add
            if self.q_table_initializer.lower() == "zeros":
                self.q_table = np.row_stack((self.q_table, np.zeros((state_to_add, self.num_action))))
            elif self.q_table_initializer.lower() == "normal" or self.q_table_initializer.lower() == "gaussian":
                self.q_table = np.row_stack((self.q_table, np.random.randn(state_to_add, self.num_action)))
            else:
                self.q_table = np.row_stack((self.q_table, np.zeros((state_to_add, self.num_action))))
                
            print("Add new states, current Q table size", self.q_table.shape)

    def check_exist_action(self, a):
        if a >= self.num_action:
            action_to_add = a - self.num_action + 1
            self.num_action += action_to_add
            if self.q_table_initializer.lower() == "zeros":
                self.q_table = np.column_stack((self.q_table, np.zeros((self.num_state, action_to_add))))
            elif self.q_table_initializer.lower() == "normal" or self.q_table_initializer.lower() == "gaussian":
                self.q_table = np.column_stack((self.q_table, np.random.randn(state_to_add, self.num_action)))
            else:
                self.q_table = np.column_stack((self.q_table, np.zeros((self.num_state, action_to_add))))

            print("Add new actions, current Q table size", self.q_table.shape)

    def choose_action(self, s, exploit=False):
        self.check_exist_state(s)
        if self.num_action == 0:
            self.check_exist_action(0)
        a = np.argmax(self.q_table[s, :])
        print(self.q_table[s, :])
        if exploit:
            return a
        else:
            if np.random.rand() > self.epsilon:
                return a
            else:
                return np.random.choice(self.num_action)

    def learn(self, s, a, r, s_):
        self.check_exist_state(s)
        self.check_exist_state(s_)
        self.check_exist_action(a)
        q_pred = self.q_table[s, a]
        q_target = r + self.gamma * np.max(self.q_table[s_, :])
        q_error = q_target - q_pred
        self.q_table[s, a] += self.alpha * q_error


if __name__ == "__main__":
    agent = QLearningAgent(6, 3, q_table_initializer="gaussian")
    print(agent.choose_action(4))
    print(agent.choose_action(8))
    print(agent.q_table)
    