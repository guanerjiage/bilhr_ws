#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import time
from rl_dt_agent import RLDTAgent
from env_grid_world import GridWorld


if __name__ == "__main__":
    env = GridWorld(5, 10, init_reward=-1)
    env.define_reward(9, 4, 50)
    print(env)

    agent = RLDTAgent(2, env.action_space, 0.99, 0.15, 50.0, 20, 0.0)
    agent.load_agent("rldt_gridworld")

    old_mode = True
    for i in range(10):
        print("Episode %d" % i)
        s = env.reset()
        terminal = False
        while not terminal:
            a = agent.choose_action(s)
            s_, r, terminal = env.update(a)
            agent.add_exp_to_tree(s, a, r, s_)
            agent.learn()
            new_mode = agent.determine_exploration()
            if new_mode ^ old_mode:
                print("New exploration mode " + str(new_mode))
                old_mode = new_mode
            s = s_       
            
    print("Test")
    
    print("State Transition")
    print(agent.state_transition)
    print("Rewards")
    print(agent.reward_function)
    print("QQQ")
    print(agent.action_value_function)
    print("Policy")
    
    s = env.reset()
    env.print_state()
    terminal = False
    while not terminal:
        a = agent.choose_action(s, True)
        s_, r, terminal = env.update(a)
        s = s_
        env.print_state()

    print(agent.dt_list[0])
    print(len(agent.dt_list[0].node_list))
    policy_mat = np.zeros((5, 10), dtype=np.str)
    for i in range(5):
        for j in range(10):
            if str([j, i]) in agent.policy_function:
                policy_mat[i, j] = agent.policy_function[str([j, i])][0]
    print(policy_mat)
  
    

