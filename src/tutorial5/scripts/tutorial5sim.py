#!/usr/bin/env python
import numpy as np
from enum import Enum
import random
from sklearn import tree
import csv
from pathlib import Path
import os


class State:
    def __init__(self, l, b=0, k=0, unexplore=10.0, Rmax=False):
        #self.joint = j
        self.ball_position = b
        self.goal_keeper = k
        # the q value of there actions
        self.q = np.full(3, 0.0)
        if Rmax:
            self.q = np.full(3, unexplore)
        self.visit = np.full(3, 0.0)
        # the step value of joint change in different states
        self.step = 0.1
        self.unexplore = unexplore
        # the offset is used to shift relative joint change into positive class labels
        self.offset = 1
        self.label = l
        #self.robot = robot

    def __eq__(self, other):
        return self.label==other.label
    def __le__(self, other):
        return self.label<other.label

    def printstate(self):
        print("state: l="+str(self.label)+" q= "+str(self.q[0])+" "+str(self.q[1])+" "+str(self.q[2]))

    def get_best_action(self): # get the action with largest Q value
        # if all actions are unexplored randomly take one
        best_action = random.randint(0,2)
        if self.q[best_action]<self.q[0]:
            best_action = 0
        if self.q[1] > self.q[best_action]:
            best_action = 1
        if self.q[2] > self.q[best_action]:
            best_action = 2
        print("best action: "+str(best_action))
        return best_action

    def execute_action(self, action):
        self.visit[action]+=1
        if action == Action.LEFT.value:
            new_l = self.move_left_sim()
        elif action == Action.RIGHT.value:
            new_l = self.move_right_sim()
        elif action == Action.KICK.value:
            new_l = self.kick_sim()
        return new_l

    
    # simulate the model without using real robot
    def move_left_sim(self):
        print("left")
        l = self.label + 1
        return l
    def move_right_sim(self):
        print("right")
        l = self.label - 1
        return l
    def kick_sim(self):
        print("kick")
        l = self.label
        return l


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    KICK = 2


def RL_DT(s, RMax=20):
    # two decision tree to learn rewards and relative state change(only joint changed)
    RM = tree.DecisionTreeRegressor(max_depth=4)
    PM = tree.DecisionTreeClassifier(max_depth=4)
    # the reward
    # the relative change in joint
    # the input data of decision tree including state and action
    # set of states
    # load the previous model
    states, reward_data, transition_data, input_data = load_model()

    # inite first state when model is empty
    if len(states)==0:
        states = [s]
    else:
        # reshape the data to work with fit function
        PM = PM.fit(input_data, transition_data)
        RM = RM.fit(input_data, reward_data)
        s = states[0]

    while True:
        print("new loop")
        s.printstate()
        action = s.get_best_action()
        print("After applying action "+str(action))
        s_new_label = s.execute_action(action)
        s_new = None
        # find whether the new state exist in states vector already
        for si in states:
            if s_new_label==si.label:
                s_new=si
        # if s_new is a new state, add it into state set
        if s_new is None:
            print("append new state")
            s_new = State(s_new_label)
            states.append(s_new)

        print("entering state")
        s_new.printstate()
        
        # get the reward from user input
        txt = raw_input("Give reward: f for fall, s for success, m for miss the shoot:   ")
        if txt=='f':
            reward = -RMax
            #robot.prepose()
        elif txt=='s':
            reward = RMax
            #robot.prepose()
        elif txt=='m':
            reward = -5
            #robot.prepose()
        elif txt=='q':
            save_model(states, reward_data, transition_data, input_data)
            break
        else:
            reward = -1
        PM, RM, CH = update_model(PM, RM, s, action, reward, s_new, reward_data, transition_data, input_data)
        # exp=0 for explore 1 for exploit
        exp = check_policy(states, RMax)
        if True:
            compute_values(PM, RM, states, exp)
            print("check the new value: ")
            for s in states:
                s.printstate()

        if txt=='f' or txt=='s' or txt=='m':
            save_model(states, reward_data, transition_data, input_data)
            s = states[0]
        else:
            s = s_new

def save_model(states, reward_data, transition_data, input_data):
    # remove the old file if exist
    my_file = Path("./reward_data.csv")
    if my_file.is_file():
        os.remove('./reward_data.csv')
    my_file = Path("./transition_data.csv")
    if my_file.is_file():
        os.remove('./transition_data.csv')
    my_file = Path("./input_data.csv")
    if my_file.is_file():
        os.remove('./input_data.csv')
    my_file = Path("./states.csv")
    if my_file.is_file():
        os.remove('./states.csv')

    np.savetxt('input_data.csv', input_data, delimiter=",")
    np.savetxt('transition_data.csv', transition_data, delimiter=",")
    np.savetxt('reward_data.csv', reward_data, delimiter=",")

    state_data = []
    for s in states:
        state = [s.label, s.ball_position, s.goal_keeper, s.q[0], s.q[1], s.q[2], s.visit[0], s.visit[1], s.visit[2]]
        state_data.append(state)
    np.savetxt('states.csv', state_data, delimiter=",")
    print "save model"
    print state_data

def load_model():
    reward_data = []
    my_file = Path("./reward_data.csv")
    if my_file.is_file():
        with open('reward_data.csv', 'rb') as f:
            reader = csv.reader(f)
            reward_data = [float(row[0]) for row in reader if row]
    transition_data = []
    my_file = Path("./transition_data.csv")
    if my_file.is_file():
        with open('transition_data.csv', 'rb') as f:
            reader = csv.reader(f)
            transition_data = [int(float(row[0])) for row in reader if row]
    input_data = []
    my_file = Path("./input_data.csv")
    if my_file.is_file():
        with open('input_data.csv', 'rb') as f:
            reader = csv.reader(f)
            input_data = [np.asarray([int(float(row[0])), int(float(row[1])),\
                                      int(float(row[2])), int(float(row[3]))]) for row in reader if row]
    state_data = []
    my_file = Path("./states.csv")
    if my_file.is_file():
        with open('states.csv', 'rb') as f:
            reader = csv.reader(f)
            state_data = [[int(float(row[0])), int(float(row[1])), int(float(row[2])),\
                           float(row[3]), float(row[4]), float(row[5]),\
                           int(float(row[6])), int(float(row[7])), int(float(row[8]))] for row in reader if row]
    print "load model"
    print state_data
    states = []
    for state in state_data:
        s = State(l=state[0], b=state[1],  k=state[2])
        s.q[0] = state[3]
        s.q[1] = state[4]
        s.q[2] = state[5]
        s.visit[0] = state[6]
        s.visit[1] = state[7]
        s.visit[2] = state[8]
        s.printstate()
        states.append(s)
    return states, reward_data, transition_data, input_data



def update_model(PM, RM, s, action, reward, s_new,  reward_data, transition_data, input_data):
    # append new data
    input_data.append(np.asarray([s.label, s.ball_position, s.goal_keeper, action]))
    transition_data.append(s_new.label-s.label+s.offset)
    reward_data.append(reward)

    # check if the model should be different
    CH = False
    if len(input_data)==1:
        CH = True
    else:
        input = input_data[-1]
        input = input.reshape(1, -1)
        trans_pred = PM.predict(input)
        reward_pred = RM.predict(input)
        print("predict relative change in label = "+str(trans_pred)+" predict rewards = "+str(reward_pred))
        if trans_pred!=transition_data[-1] or abs(reward_pred - reward)>0.1:
            CH = True
            print("The model is changed")

    # if so update the model
    if CH:
        print input_data
        print  transition_data
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
    print("check_policy, In the mode "+ str(exp))
    return exp


def compute_values(PM, RM, states, exp, Rmax=10):
    theta = 0.01
    discount_factor = 0.9
    # get the minimul visite count among all states
    minvisit = 10000
    for s in states:
        state_visit = s.visit[0]+s.visit[1]+s.visit[2]
        if  state_visit< minvisit:
            minvisit = state_visit
    print("least visit states with visit number of "+str(minvisit))
    while True:
        delta = theta
        # Update each state...
        for s in states:
            print("in state")
            s.printstate()
            if s.visit[0]+s.visit[1]+s.visit[2] == 0: # if the states is never explored, do not predict and expend statues
                print("the state is never visit and will not be expend")
                continue
            if exp == 0 and s.visit[0]+s.visit[1]+s.visit[2] == minvisit:  # if in explore mode less visit states are given Rmax
                s.q[0] = Rmax
                s.q[1] = Rmax
                s.q[2] = Rmax
                print("the state is least visit set to Rmax")
            else:
                print("updating value")
                for a in range(3):
                    input = np.asarray([s.label, s.ball_position, s.goal_keeper, a])
                    input = input.reshape(1, -1)
                    reward = RM.predict(input)
                    last_value = s.q[a]
                    s.q[a] = reward
                    print("predicted reward is"+str(reward))
                    prob = PM.predict_proba(input)
                    print(prob)
                    for i in range(len(prob[0])):
                        if prob[0][i]!=0:
                            print("after action "+str(a)+" transite to state "+str(s.label+i-s.offset)+" with probability "+str(prob[0][i]))
                            exist_flag = False
                            l = s.label+i-s.offset # calculate the next possible states label
                            for k in range(len(states)):
                                if states[k].label == l: # if it already exist in states vector, do value iteration
                                    print("STATE "+str(s.label)+" ACTION "+str(a)+" reward "+str(s.q[a] ))
                                    states[k].printstate()
                                    tmp = prob[0][i] * discount_factor * np.max(states[k].q)
                                    print("prob "+str(prob[0][i])+" max "+str(np.max(states[k].q))+" tmp "+ str(tmp))
                                    s.q[a] = s.q[a] + tmp
                                    print("update result is " + str(s.q[a]))
                                    
                                    exist_flag = True
                                    print("exist in states vector")
                                    break
                            if not exist_flag: # if not exist in the vector add it to states vector with bonues 
                                print("not exit in states vector")
                                s_new = State(l, Rmax=True)
                                states.append(s_new)
                                s.q[a] += prob[0][i] * discount_factor * Rmax
                    # check the difference between two times of iteration
                    delta = max(delta, abs(last_value - s.q[a]))   
        # Stopping condition, if all differences are less than theta
        if delta <= theta:
            break



if __name__=='__main__':
    #robot = Central()
    #rospy.init_node('central_node',anonymous=True) #initilizes node, sets name
    # create several topic subscribers
    #rospy.Subscriber("key", String, self.key_cb)
    #rospy.Subscriber("joint_states",JointState,robot.joints_cb)
    #rate = rospy.Rate(10) # sets the sleep time to 10ms
    init_state = State(5)
   
    #robot.prepose()
    RL_DT(init_state)
