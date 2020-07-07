#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import time
from rl_dt_agent import RLDTAgent
from treasure1D_env import Treasure1D


if __name__ == "__main__":
    env = Treasure1D(10)
    agent = RLDTAgent(1, env.action_space, 0.99, 0.0, 10.0, 10, 0.0)
    
    old_mode = True
    for i in range(30):
        print("Episode %d" % i)
        s = env.reset()
        while s != 9:
            a = agent.choose_action([s])
            s_, r = env.update(a)
            agent.add_exp_to_tree([s], a, r, [s_])
            agent.learn()
            new_mode = agent.determine_exploration()
            if new_mode ^ old_mode:
                print("New exploration mode " + str(new_mode))
                old_mode = new_mode
            if i == 0:
                print(agent.state_list)
                print(agent.action_value_function)
            s = s_        
    
    print("Test")
    print(agent.action_value_function)
    print(agent.reward_function)
    print(agent.policy_function)
    
    s = env.reset()
    print(env)
    while s != 9:
        a = agent.choose_action([s], True)
        s_, r = env.update(a)
        s = s_
        print(env)

    print(agent.dt_list[0])
    print(agent.dt_list[1])

    
    
