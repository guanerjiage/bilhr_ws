#!/usr/bin/env python
import numpy as np
from enum import Enum
import random
from sklearn import tree


class State:
    def __init__(self, j, b=0, k=0, unexplore=0):
        self.joint = j
        self.ball_position = b
        self.goal_keeper = k
        # the q value of there actions
        self.q = np.full(3, unexplore)
        self.visit = np.full(3, 0)
        # the step value of joint change in different states
        self.step = 0.1
        self.unexplore = unexplore
        # the offset is used to shift relative joint change into positive class labels
        self.offset = 10
        self.label = j / self.step + self.offset

    def __eq__(self, other):
        return self.label==other.label
    def __le__(self, other):
        return self.label<other.label

    def printstate(self):
        print("state: l= "+self.l+"q= "+self.q[0]+self.q[1]+self.q[2])

    def get_best_action(self): # get the action with largest Q value
        # if all actions are unexplored randomly take one
        if self.q[0]==self.unexplore and self.q[1]==self.unexplore and  self.q[2]==self.unexplore:
            return random.randint(0,3)
        best_action = 0
        if self.q[1] > self.q[0]:
            best_action = 1
        if self.q[2] > self.q[1]:
            best_action = 2
        return best_action

    def execute_action(self, action):
        if(action == Action.LEFT):
            return self.move_left()
        elif (action == Action.RIGHT):
            return self.move_right()
        elif (action == Action.KICK):
            return self.kick()

    # the three actions should return the new states from reading the joint
    def move_left(self):
        s = State(self.joint - 0.1)
        return s
    def move_right(self):
        s = State(self.joint + 0.1)
        return s
    def kick(self):
        return self


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    KICK = 2


def RL_DT(s, RMax=20):
    # two decision tree to learn rewards and relative state change(only joint changed)
    RM = tree.DecisionTreeRegressor(max_depth=4)
    PM = tree.DecisionTreeClassifier(max_depth=4)
    # the reward
    reward_data = []
    # the relative change in joint
    transition_data = []
    # the input data of decision tree including state and action
    input_data = []
    # set of states
    states = [s]

    while True:
        s.printstate()
        action = s.get_best_action()
        print("After applying action "+action+" we are in ")
        s_new = s.execute_action(action)
        s_new.printstate()
        # if s_new is a new state, add it into state set
        if s!=s_new:
            states.append(s_new)
        # get the reward from user input
        txt = raw_input("Give reward: f for fall, s for success")
        if txt=='f':
            reward = -RMax
        elif txt=='s':
            reward = RMax
        else:
            reward = -1
        PM, RM, CH = update_model(PM, RM, s, action, reward, s_new, reward_data, transition_data, input_data)
        # exp=0 for explore 1 for exploit
        exp = check_policy(states, RM, RMax)
        if CH:
            compute_values(PM, RM, states, exp)
            print("check the new value: ")
            for s in states:
                s.printstate()

        if txt=='f':
            s = State(0)
        else:
            s = s_new



def update_model(PM, RM, s, action, reward, s_new,  reward_data, transition_data, input_data):
    # append new data
    input_data.append(np.asarray([s.label, s.ball_position, s.goal_keeper, action]))
    transition_data.append((s_new.label-s.label)/s.step+s.offset)
    reward_data.append(reward)

    # check if the model should be different
    CH = False
    trans_pred = PM.predict(input_data[-1])
    reward_pred = RM.predict(input_data[-1])
    if trans_pred==transition_data[-1] and abs(reward_pred-reward_data[-1])>2:
        CH = True

    # if so update the model
    if CH:
        PM = PM.fit(input_data, transition_data)
        RM = RM.fit(input_data, reward_data)
    return PM, RM, CH

def check_policy(states, RMax):
    # if any q value that already explored larger than 0.4 RMax return exploit
    exp = 0 # for explore
    for s in states:
        for r in s.q:
            if r!=s.unexplore and r>=0.4*RMax:
                exp = 1 # for exploit
    print("In the mode "+ exp)
    return exp


def one_step_lookahead(s, V, PM, RM, states):
    discount_factor = 0.9
    A = np.zeros(3)
    for a in range(3):
        input = np.asarray([s.label, s.ball_position, s.goal_keeper, a])
        reward = RM.predict(input)
        A[a] += reward
        # the vector prob[i] contains the probability to go to state has relative change i in label
        prob = PM.predict_proba(input)
        for i in range(len(prob)):
            l = s.label+i
            for k in range(len(states)):
                if states[k].label == l:
                    break
            A[a] += prob[i] * discount_factor * V[k]
    s.q = A
    return A

def compute_values(PM, RM, states, exp, Rmax=20):
    theta = 0.01
    # the V vector is used to remember the best q value of a state i in i position of states vector
    V = np.zeros(states)
    # get the minimul visite count among all states
    minvisit = 10000
    for s in states:
        state_visit = s.q[0]+s.q[1]+s.q[2]
        if  state_visit> minvisit:
            minvisit = state_visit
    while True:
        # Stopping condition
        delta = theta
        # Update each state...
        for s in states:
            if exp == 0 and s.q[0] + s.q[1] + s.q[2] == minvisit:  # if in explore mode less visit states are given Rmax
                s.q[0] = Rmax
                s.q[1] = Rmax
                s.q[2] = Rmax
            else:
                for a in range(3):
                    input = np.asarray([s.label, s.ball_position, s.goal_keeper, a])
                    reward = RM.predict(input)
                    s.q[a] = reward
                    prob = PM.predict_proba(input)





        '''
        for i in range(len(states)):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(states[i], V, PM, RM)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[i]))
            V[i] = best_action_value
            # Check if we can stop
        '''
        if delta < theta:
            break



if __name__=='__main__':
    init_state = State(0)
    RL_DT(init_state)
