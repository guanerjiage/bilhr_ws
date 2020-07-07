#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import rospy
from std_msgs.msg import String
import numpy as np
import json
from rldt.rl_dt_agent import RLDTAgent, unicode_convert
import scipy.io as sio

state_vec = [0, 0, 0]
state_value = [0.0, 0, 0]

# read keyboard input
# def read_input():
    

def state_cb(data):
    global state_vec
    global state_value
    print(data.data)
    state_str = data.data.split()
    # RHipRoll from -0.4 to -0.6 in 0.02 interval
    RHipRoll = float(state_str[0])
    state_value[0] = RHipRoll
    RHipRoll = int(round((RHipRoll + 0.4) / -0.02))
    RHipRoll = max(0, RHipRoll)
    RHipRoll = min(10, RHipRoll)
    # ball x coordinate from 225 to 295 in 5 intervals
    ball = int(state_str[1])
    state_value[1] = ball
    ball = int(round((ball - 225) / 14))
    ball = max(0, ball)
    ball = min(4, ball)
    # goalkeeper x coordinate from 135 to 235 in 5 intervals
    gk = int(state_str[2])
    state_value[2] = gk
    gk = int(round((gk - 135) / 20))
    gk = max(0, gk)
    gk = min(4, gk) 
    state_vec = [RHipRoll, ball, gk]
    print(state_vec)

if __name__ == "__main__":
    s_trace = []
    s_next_trace = []
    a_trace = []
    r_trace = []
    r_cummulative_trace = []
    transition_list = []
    # agent = QLearningAgent(200, 3, gamma=0.99, epsilon=0.3, q_table_initializer="zero")
    agent = RLDTAgent(3, ["l", "r", "kick"], gamma=0.99, epsilon=0.0, desired_reward=14, step_max=10, gain_threshold=0.0, exploration_rate=0.3)

    rospy.Subscriber("rl_state", String, state_cb)
    pub = rospy.Publisher("learning", String, queue_size=10)
    rospy.init_node("RLNode", anonymous=True)
    rate = rospy.Rate(10)

    help_msg = "Instructions\n" + \
        "set joint value: set joint position to value, e.g. set LShoulderRoll 0.3: set left shoulder roll to 0.3 \n" + \
        "setr joint value: set joint position relatively to value, e.g. setr LShoulderRoll 0.3: set left shoulder roll +0.3 \n" + \
        "penaltyready: prepare for penalty kick\n" + \
        "kick: execute kick action\n" + \
        "start: start RL sampling\n" + \
        "next: step to the next episode\n" + \
        "save: save agent's trees and parameters\n" + \
        "store: store current traces into files\n" + \
        "load: load traces from files\n" + \
        "state: get current state and store it\n" + \
        "print: print agent decision trees\n" + \
        "restart: restart\n" + \
        "l: right leg a bit to left\n" + \
        "r: right leg a bit to right\n" + \
        "h: display this help\n" + \
        "q: quit\n"
    print(help_msg)

    while not rospy.is_shutdown():        
        msg = raw_input("Type a control signal: ")

        if msg[0:3] != "set":
            msg = msg.lower()

        if msg == "start":
            print("Start RL Learning")
            print("Set to initial position")
            pub.publish("penaltyready")
            rospy.sleep(14)
            print("Type next to continue learning")

        # perform one episode of learning
        elif msg == "next":
            pub.publish("state")
            rospy.sleep(1)
        
            reward_cum = 0
            terminate = False
            while not terminate:
                # generate an action
                a = agent.choose_action(state_vec)
                act = "No Command"
                if a == "l":
                    if state_value[0] <= -0.41:
                        act = "setr RHipRoll 0.02"
                elif a == "r":
                    if state_value[0] >= -0.59:
                        act = "setr RHipRoll -0.02"
                elif a == "kick":
                    act = "kick"
                else:
                    print("Action Invalid")

                print(a, act)
                s_trace.append(state_vec)
                a_trace.append(a)
                
                pub.publish(act)
                rospy.sleep(2)                
                
                pub.publish("state")
                rospy.sleep(1)                

                # give a reward to this action
                reward = raw_input("Give a reward: ")
                reward = int(reward)
                if reward != -1:
                    terminate = True
                    fake_state_vec = [state_vec[0], -1, state_vec[2]]
                    s_next_trace.append(fake_state_vec)
                else:
                    s_next_trace.append(state_vec)
                r_trace.append(reward)
                reward_cum += reward
                
                data = {"s": s_trace[-1], "a": a, "r": reward, "s_": s_next_trace[-1]}
                print(data)
                transition_list.append(data)
                # store transitions in a temp file
                transition_json = json.dumps(transition_list, indent=4)
                with open("RL_transitions_temp.json", "w") as f:
                    f.write(transition_json)
                    f.write("\n")

                agent.add_exp_to_tree(s_trace[-1], a, reward, s_next_trace[-1])
                agent.learn()
                print("Exploration: ", agent.determine_exploration())
                
            r_cummulative_trace.append(reward_cum)

        elif msg == "save":
            agent.save_agent("RLDT_penalty")
        
        elif msg == "store":
            # avoid mistype
            if len(r_cummulative_trace) == 0:
                check = raw_input("Reward empty, are you sure?")
                if check != "y":
                    continue
            # only need to store the transitions
            reward_dict = {"r_cum": r_cummulative_trace}
            reward_json = json.dumps(reward_dict, indent=4)
            with open("RL_reward.json", "w") as f:
                f.write(reward_json)
                f.write("\n")
                print("Cummulated Reward saved")

            transition_json = json.dumps(transition_list, indent=4)
            with open("RL_transitions.json", "w") as f:
                f.write(transition_json)
                f.write("\n")
                print("Transitions saved")

        elif msg == "load":
            with open("RL_reward.json", "r") as f:
                reward_json = f.read()
                reward_dict = unicode_convert(json.loads(reward_json))
            r_cummulative_trace = reward_dict["r_cum"]
            print(r_cummulative_trace)
            print("Cummulated Reward loaded")

            # load transitions
            with open("RL_transitions.json", "r") as f:
                transition_json = f.read()
                transition_list = unicode_convert(json.loads(transition_json))
            print("Transition loaded")
            # learn again
            for transition in transition_list:
                s_trace.append(transition["s"])
                a_trace.append(transition["a"])
                r_trace.append(transition["r"])
                s_next_trace.append(transition["s_"])
                agent.add_exp_to_tree(transition["s"], transition["a"], transition["r"], transition["s_"])
                agent.learn()
                agent.determine_exploration()
            agent.learn()
            print(agent.determine_exploration())
            print("Transitions learning complete")

        elif msg == "print":
            for dt in agent.dt_list:
                print(dt)
            print(agent.fake_action_value_function)
            print(agent.action_value_function)

        # no use
        elif msg == "restart":
            rospy.Subscriber("rl_state", String, state_cb)
            pub = rospy.Publisher("learning", String, queue_size=10)
            rospy.init_node("RLNode", anonymous=True)
                

        # l, r, h and q are not sent to the subscriber
        elif msg == "h":
            print(help_msg)
        elif msg == "q":
            break 
        elif msg == "l":
            pub.publish("setr RHipRoll 0.02")
        elif msg == "r":
            pub.publish("setr RHipRoll -0.02")
        else:
            pub.publish(msg)
        rate.sleep()

    # try:
    #     read_input()
    # except rospy.ROSInterruptException:
    #     pass
